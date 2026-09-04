"""
Background Text-to-Speech engine for ISL Sign-to-Speech.

On Windows, pyttsx3 (SAPI5) produces audible speech on the main
thread but is unreliable from worker threads and child processes:
runAndWait() often finishes on time while no audio is heard.

This engine uses Windows System.Speech through a persistent
PowerShell worker instead:

    main webcam loop  ->  queue  ->  background thread
                                      |
                                      v
                         PowerShell System.Speech worker

The OpenCV / prediction loop only enqueues text and never blocks
on TTS.  No extra pip packages are required.
"""

from __future__ import annotations

import base64
import queue
import subprocess
import threading
import traceback


_SHUTDOWN = "__SPEECH_ENGINE_SHUTDOWN__"

# Persistent PowerShell worker:
# - reads one Base64 line at a time (safe for punctuation / quotes)
# - speaks with System.Speech
# - prints OK after each utterance so Python can stay in sync
# - exits on __SPEECH_ENGINE_SHUTDOWN__
_POWERSHELL_WORKER = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Volume = 100
$synth.Rate = 0
function Write-Ack([string]$status) {
    [Console]::Out.WriteLine($status)
    [Console]::Out.Flush()
}
try {
    while ($true) {
        $line = [Console]::In.ReadLine()
        if ($null -eq $line) { break }
        if ($line -eq '__SPEECH_ENGINE_SHUTDOWN__') { break }
        try {
            $bytes = [Convert]::FromBase64String($line)
            $text = [System.Text.Encoding]::UTF8.GetString($bytes)
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $synth.Speak($text)
            }
            Write-Ack 'OK'
        } catch {
            [Console]::Error.WriteLine("TTS worker speak error: $_")
            [Console]::Error.Flush()
            Write-Ack 'ERR'
        }
    }
} finally {
    $synth.Dispose()
}
"""


def _encode_utterance(text: str) -> str:
    """Encode spoken text as a single Base64 line for PowerShell."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


class SpeechEngine:
    """
    Queue-based speech engine using Windows System.Speech.

    Public API:

        speech = SpeechEngine()
        speech.speak_word("HELLO")
        speech.speak_sentence("I am going to school tomorrow.")
        speech.shutdown()
        speech.is_enabled
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, str] | str] = queue.Queue()
        self._enabled = False
        self._process: subprocess.Popen[str] | None = None
        self._io_lock = threading.Lock()

        try:
            self._process = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    _POWERSHELL_WORKER,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception:
            print(
                "[SpeechEngine] ERROR: could not start "
                "Windows TTS worker (PowerShell).",
                flush=True,
            )
            traceback.print_exc()
            self._process = None
            self._enabled = False
            return

        self._enabled = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="SpeechEngine",
            daemon=True,
        )
        self._thread.start()

        print(
            "[SpeechEngine] Windows System.Speech TTS ready.",
            flush=True,
        )

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def speak_word(self, word: str) -> None:
        """Queue a recognized sign/word for speech."""
        self._enqueue("word", word)

    def speak_sentence(self, sentence: str) -> None:
        """Queue a complete generated sentence for speech."""
        self._enqueue("sentence", sentence)

    def shutdown(self) -> None:
        """Shut down the TTS worker cleanly."""
        if not self._enabled and self._process is None:
            return

        print(
            "[SpeechEngine] Sending shutdown request...",
            flush=True,
        )

        try:
            self._queue.put(_SHUTDOWN)
        except Exception:
            pass

        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=20)

        self._stop_process()
        self._enabled = False

        print(
            "[SpeechEngine] Shutdown complete.",
            flush=True,
        )

    @property
    def is_enabled(self) -> bool:
        """Return whether the speech engine is enabled."""
        return self._enabled

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _enqueue(self, kind: str, text: str) -> None:
        if not self._enabled:
            return

        if not text:
            return

        cleaned = str(text).strip()
        if not cleaned:
            return

        try:
            self._queue.put((kind, cleaned))
        except Exception:
            traceback.print_exc()

    def _worker_loop(self) -> None:
        """Background thread: drain queue into the PowerShell worker."""
        try:
            while True:
                item = self._queue.get()

                if item == _SHUTDOWN:
                    print(
                        "[SpeechEngine] Shutting down...",
                        flush=True,
                    )
                    self._send_line(_SHUTDOWN)
                    break

                if not isinstance(item, tuple) or len(item) != 2:
                    continue

                kind, text = item
                if not text:
                    continue

                print(
                    f"[SpeechEngine] Speaking {kind}: {text}",
                    flush=True,
                )

                if self._speak(text):
                    print(
                        f"[SpeechEngine] Finished {kind}: {text}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SpeechEngine] Failed {kind}: {text!r}",
                        flush=True,
                    )
                    self._enabled = False
                    break
        finally:
            self._stop_process()

    def _speak(self, text: str) -> bool:
        """Send one utterance and wait for the OK acknowledgement."""
        if not self._send_line(_encode_utterance(text)):
            return False
        return self._wait_for_ack()

    def _send_line(self, line: str) -> bool:
        process = self._process
        if process is None or process.stdin is None:
            return False

        if process.poll() is not None:
            self._report_process_failure()
            return False

        try:
            with self._io_lock:
                process.stdin.write(line + "\n")
                process.stdin.flush()
            return True
        except Exception:
            traceback.print_exc()
            self._report_process_failure()
            return False

    def _wait_for_ack(self) -> bool:
        process = self._process
        if process is None or process.stdout is None:
            return False

        try:
            line = process.stdout.readline()
        except Exception:
            traceback.print_exc()
            return False

        if not line:
            self._report_process_failure()
            return False

        status = line.strip()
        if status == "OK":
            return True

        print(
            f"[SpeechEngine] Unexpected TTS ack: {status!r}",
            flush=True,
        )
        return False

    def _report_process_failure(self) -> None:
        process = self._process
        detail = ""
        if process is not None and process.stderr is not None:
            try:
                # Non-blocking-ish: only used after failure.
                detail = process.stderr.read()
            except Exception:
                detail = ""

        print(
            "[SpeechEngine] ERROR: Windows TTS worker exited "
            f"unexpectedly. {detail}",
            flush=True,
        )

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return

        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except Exception:
            pass

        try:
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

        self._process = None
