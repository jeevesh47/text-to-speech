"""
Diagnostic test for the Windows TTS SpeechEngine.

Run from the project root:

    .\\.venv\\Scripts\\python.exe ml\\test_tts.py

You should HEAR two utterances.  Console messages alone are not enough.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `python ml/test_tts.py` from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.speech import SpeechEngine


def main() -> None:
    print("Starting SpeechEngine diagnostic...", flush=True)

    speech = SpeechEngine()

    if not speech.is_enabled:
        print("FAIL: SpeechEngine did not enable.", flush=True)
        sys.exit(1)

    # Give the PowerShell worker a moment to finish startup.
    time.sleep(1.0)

    print(
        "Speaking diagnostic phrase "
        "(you should HEAR this)...",
        flush=True,
    )
    speech.speak_sentence(
        "TTS DIAGNOSTIC TEST. IF YOU CAN HEAR THIS, AUDIO IS WORKING."
    )

    print(
        "Speaking multi-word sentence "
        "(you should HEAR this)...",
        flush=True,
    )
    speech.speak_sentence(
        "I am going to school tomorrow."
    )

    # speak_* returns immediately; wait for the worker queue.
    print(
        "Waiting for queued speech to finish...",
        flush=True,
    )
    time.sleep(18)

    speech.shutdown()

    print(
        "Diagnostic finished. "
        "If you heard both sentences, TTS is working.",
        flush=True,
    )


if __name__ == "__main__":
    main()
