"""
Modern OpenCV Dashboard User Interface for ISL Sign-to-Speech.

Renders a 1280x720 broadcast-quality interface using Pillow for
crisp antialiased TrueType typography and OpenCV for high-speed display.
"""

from __future__ import annotations

import math
import os
import time
from typing import Sequence
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Hand connections for MediaPipe 21 landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]


def _get_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font from Windows Fonts directory with fallback."""
    windows_fonts = os.environ.get("WINDIR", "C:\\Windows") + "\\Fonts\\"
    candidates = [
        os.path.join(windows_fonts, name),
        os.path.join(windows_fonts, "segoeuib.ttf" if "b.ttf" in name else "segoeui.ttf"),
        os.path.join(windows_fonts, "arialbd.ttf" if "b.ttf" in name else "arial.ttf"),
        name,
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


class ISLDashboard:
    """
    Renders an elegant, modern 1280x720 dashboard for real-time ISL Sign-to-Speech.
    """

    # Color Palette (RGB tuples for PIL)
    BG_COLOR = (15, 23, 42)          # Slate 900
    CARD_BG = (30, 41, 59)           # Slate 800
    CARD_BORDER = (51, 65, 85)       # Slate 700
    HEADER_BG = (11, 17, 32)         # Deep Slate
    FOOTER_BG = (11, 17, 32)
    ACCENT_CYAN = (56, 189, 248)     # Sky 400
    ACCENT_BLUE = (99, 102, 241)     # Indigo 500
    SUCCESS_GREEN = (16, 185, 129)   # Emerald 500
    WARNING_AMBER = (245, 158, 11)   # Amber 500
    ERROR_RED = (239, 68, 68)        # Red 500
    TEXT_PRIMARY = (248, 250, 252)   # Slate 50
    TEXT_MUTED = (148, 163, 184)     # Slate 400
    TEXT_DIM = (100, 116, 139)       # Slate 500
    CHIP_BG = (45, 55, 72)           # Slate 700 / Gray

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

        # Typography
        self.font_title = _get_font("segoeuib.ttf", 19)
        self.font_badge = _get_font("segoeui.ttf", 11)
        self.font_section = _get_font("segoeuib.ttf", 12)
        self.font_sign = _get_font("segoeuib.ttf", 32)
        self.font_sign_sub = _get_font("segoeui.ttf", 13)
        self.font_conf = _get_font("segoeuib.ttf", 15)
        self.font_chip = _get_font("segoeuib.ttf", 13)
        self.font_sentence = _get_font("segoeui.ttf", 16)
        self.font_controls = _get_font("segoeui.ttf", 12)
        self.font_controls_bold = _get_font("segoeuib.ttf", 12)

        # Pulse animation state
        self._start_time = time.monotonic()

    def draw_hand_landmarks_on_camera(
        self,
        frame: np.ndarray,
        mediapipe_result,
    ) -> np.ndarray:
        """
        Draw clean, modern hand skeletons onto the camera frame.
        Mirrored frame: left hand is Cyan, right hand is Warm Amber.
        """
        if not mediapipe_result or not getattr(mediapipe_result, "hand_landmarks", None):
            return frame

        h, w = frame.shape[:2]
        handedness_list = getattr(mediapipe_result, "handedness", [])

        for idx, hand_lms in enumerate(mediapipe_result.hand_landmarks):
            # Determine color by hand label if available
            is_left = True
            if idx < len(handedness_list) and handedness_list[idx]:
                label = handedness_list[idx][0].category_name
                # On mirrored selfie camera, physical left is labeled 'Left'
                is_left = (label == "Left")

            line_color = (235, 206, 56) if is_left else (56, 140, 245)   # BGR
            dot_color = (255, 255, 255)                                   # White

            # Convert normalized landmarks to pixel coords
            pts = []
            for lm in hand_lms:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pts.append((px, py))

            # Draw bones
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if p1_idx < len(pts) and p2_idx < len(pts):
                    cv2.line(frame, pts[p1_idx], pts[p2_idx], line_color, 2, cv2.LINE_AA)

            # Draw joints
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, line_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 2, dot_color, -1, cv2.LINE_AA)

        return frame

    def render(
        self,
        camera_frame: np.ndarray,
        current_prediction: str,
        current_confidence: float,
        confidence_threshold: float,
        stable_prediction: str | None,
        stable_count: int,
        stable_count_required: int,
        accumulated_words: Sequence[str],
        generated_sentence: str,
        speech_status: str,
        is_speaking: bool,
        buffer_len: int,
        buffer_max: int,
        fps: float,
        llm_provider: str,
    ) -> np.ndarray:
        """
        Produce a finished 1280x720 BGR frame ready for cv2.imshow.
        """
        # Create base PIL canvas
        canvas = Image.new("RGB", (self.width, self.height), self.BG_COLOR)
        draw = ImageDraw.Draw(canvas)

        now = time.monotonic() - self._start_time

        # ─────────────────────────────────────────────────────────────
        # 1. HEADER BAR (Y: 0 to 58)
        # ─────────────────────────────────────────────────────────────
        draw.rectangle([(0, 0), (self.width, 58)], fill=self.HEADER_BG)
        draw.line([(0, 58), (self.width, 58)], fill=self.CARD_BORDER, width=1)

        # Title
        title_text = "ISL SIGN -> ENGLISH SPEECH"
        draw.text((25, 17), title_text, fill=self.TEXT_PRIMARY, font=self.font_title)

        # Badge: REAL-TIME AI
        badge_text = "REAL-TIME AI"
        b_box = draw.textbbox((0, 0), badge_text, font=self.font_badge)
        b_w = b_box[2] - b_box[0]
        badge_x = 345
        draw.rounded_rectangle(
            [(badge_x, 18), (badge_x + b_w + 18, 42)],
            radius=4,
            fill=(30, 41, 59),
            outline=self.ACCENT_CYAN,
            width=1,
        )
        draw.text((badge_x + 9, 22), badge_text, fill=self.ACCENT_CYAN, font=self.font_badge)

        # Right Telemetry Pills
        pulse = int(128 + 127 * math.sin(now * 5))
        live_dot_color = (16, pulse, 129)
        draw.ellipse([(830, 24), (840, 34)], fill=live_dot_color)
        draw.text((848, 22), "LIVE", fill=self.TEXT_PRIMARY, font=self.font_badge)

        # FPS Pill
        draw.text((905, 22), f"FPS: {fps:.1f}", fill=self.TEXT_MUTED, font=self.font_badge)

        # Sequence Buffer Pill
        buf_color = self.SUCCESS_GREEN if buffer_len >= buffer_max else self.WARNING_AMBER
        draw.text((990, 22), f"BUFFER: {buffer_len}/{buffer_max}", fill=buf_color, font=self.font_badge)

        # LLM Provider Pill
        provider_name = llm_provider.upper()
        draw.text((1115, 22), f"LLM: {provider_name}", fill=self.ACCENT_CYAN, font=self.font_badge)

        # ─────────────────────────────────────────────────────────────
        # 2. LEFT PANEL: CAMERA VIEWPORT (X: 20 to 730, Y: 70 to 520)
        # ─────────────────────────────────────────────────────────────
        cam_card_x1, cam_card_y1 = 20, 70
        cam_card_x2, cam_card_y2 = 730, 520
        draw.rounded_rectangle(
            [(cam_card_x1, cam_card_y1), (cam_card_x2, cam_card_y2)],
            radius=8,
            fill=self.CARD_BG,
            outline=self.CARD_BORDER,
            width=1,
        )

        # Camera Header text
        draw.text((cam_card_x1 + 16, cam_card_y1 + 12), "LIVE CAMERA PREVIEW", fill=self.TEXT_MUTED, font=self.font_section)
        draw.text((cam_card_x2 - 130, cam_card_y1 + 12), "MIRRORED (SELFIE)", fill=self.TEXT_DIM, font=self.font_badge)

        # Insert Camera Frame inside viewport
        vp_x, vp_y = cam_card_x1 + 12, cam_card_y1 + 36
        vp_w, vp_h = (cam_card_x2 - cam_card_x1) - 24, (cam_card_y2 - cam_card_y1) - 48

        cam_h, cam_w = camera_frame.shape[:2]
        scale = min(vp_w / cam_w, vp_h / cam_h)
        new_w, new_h = int(cam_w * scale), int(cam_h * scale)
        resized_cam = cv2.resize(camera_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB for PIL pasting
        rgb_cam = cv2.cvtColor(resized_cam, cv2.COLOR_BGR2RGB)
        cam_pil = Image.fromarray(rgb_cam)

        offset_x = vp_x + (vp_w - new_w) // 2
        offset_y = vp_y + (vp_h - new_h) // 2
        canvas.paste(cam_pil, (offset_x, offset_y))

        # Viewport outline
        draw.rectangle(
            [(offset_x - 1, offset_y - 1), (offset_x + new_w, offset_y + new_h)],
            outline=self.CARD_BORDER,
            width=1,
        )

        # ─────────────────────────────────────────────────────────────
        # 3. BUFFER BAR (BELOW CAMERA, Y: 532 to 558)
        # ─────────────────────────────────────────────────────────────
        buf_y1, buf_y2 = 532, 558
        draw.rounded_rectangle(
            [(cam_card_x1, buf_y1), (cam_card_x2, buf_y2)],
            radius=6,
            fill=self.CARD_BG,
            outline=self.CARD_BORDER,
            width=1,
        )

        draw.text((cam_card_x1 + 12, buf_y1 + 5), "TEMPORAL BUFFER", fill=self.TEXT_MUTED, font=self.font_badge)

        bar_x1 = cam_card_x1 + 130
        bar_x2 = cam_card_x2 - 100
        bar_y_center = (buf_y1 + buf_y2) // 2
        bar_w = bar_x2 - bar_x1

        # Track background
        draw.rounded_rectangle([(bar_x1, bar_y_center - 4), (bar_x2, bar_y_center + 4)], radius=4, fill=(15, 23, 42))

        # Track fill
        progress = max(0.0, min(1.0, buffer_len / max(1, buffer_max)))
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            fill_color = self.SUCCESS_GREEN if progress >= 1.0 else self.WARNING_AMBER
            draw.rounded_rectangle([(bar_x1, bar_y_center - 4), (bar_x1 + fill_w, bar_y_center + 4)], radius=4, fill=fill_color)

        draw.text((bar_x2 + 15, buf_y1 + 5), f"{buffer_len}/{buffer_max} frames", fill=self.TEXT_MUTED, font=self.font_badge)

        # ─────────────────────────────────────────────────────────────
        # 4. RIGHT PANEL - CARD 1: REAL-TIME SIGN RECOGNITION (Y: 70 to 305)
        # ─────────────────────────────────────────────────────────────
        r_card_x1, r_card_x2 = 750, 1260
        c1_y1, c1_y2 = 70, 305

        draw.rounded_rectangle(
            [(r_card_x1, c1_y1), (r_card_x2, c1_y2)],
            radius=8,
            fill=self.CARD_BG,
            outline=self.CARD_BORDER,
            width=1,
        )

        # Header
        draw.text((r_card_x1 + 18, c1_y1 + 14), "REAL-TIME SIGN RECOGNITION", fill=self.TEXT_MUTED, font=self.font_section)

        # Stability Pill Badge (Centered text within pill)
        if stable_prediction:
            status_badge = f"* STABLE ({stable_count_required}/{stable_count_required})"
            badge_bg = (6, 78, 59)     # Dark green
            badge_fg = (52, 211, 153)  # Emerald light
        elif stable_count > 0:
            status_badge = f"o ACCUMULATING ({stable_count}/{stable_count_required})"
            badge_bg = (120, 53, 15)   # Dark amber
            badge_fg = (251, 191, 36)  # Amber light
        else:
            status_badge = "o LISTENING"
            badge_bg = (30, 41, 59)
            badge_fg = self.TEXT_MUTED

        sb_box = draw.textbbox((0, 0), status_badge, font=self.font_badge)
        sb_w = sb_box[2] - sb_box[0]
        sb_h = sb_box[3] - sb_box[1]
        pill_w = sb_w + 24
        pill_x2 = r_card_x2 - 18
        pill_x1 = pill_x2 - pill_w
        pill_y1 = c1_y1 + 11
        pill_y2 = pill_y1 + 24

        draw.rounded_rectangle(
            [(pill_x1, pill_y1), (pill_x2, pill_y2)],
            radius=12,
            fill=badge_bg,
            outline=badge_fg,
            width=1,
        )
        draw.text((pill_x1 + 12, pill_y1 + 4), status_badge, fill=badge_fg, font=self.font_badge)

        # Active Recognized Sign Text (Large & Bold)
        disp_sign = current_prediction.strip() if current_prediction else "Waiting..."
        if disp_sign.lower() in {"waiting...", "uncertain"}:
            sign_color = self.TEXT_DIM
        elif stable_prediction:
            sign_color = self.SUCCESS_GREEN
        else:
            sign_color = self.ACCENT_CYAN

        draw.text((r_card_x1 + 20, c1_y1 + 52), disp_sign, fill=sign_color, font=self.font_sign)

        # Helper note
        if stable_prediction:
            note_text = f"Sign recognized: {stable_prediction} (holding steady)"
            note_color = self.SUCCESS_GREEN
        elif current_confidence >= confidence_threshold:
            note_text = f"Confidence {current_confidence:.1%} >= {confidence_threshold:.0%} threshold"
            note_color = self.TEXT_MUTED
        else:
            note_text = f"Below {confidence_threshold:.0%} threshold (needs clearer gesture)"
            note_color = self.TEXT_DIM

        draw.text((r_card_x1 + 22, c1_y1 + 104), note_text, fill=note_color, font=self.font_sign_sub)

        # Confidence Bar
        conf_y = c1_y1 + 145
        draw.text((r_card_x1 + 20, conf_y), "CONFIDENCE", fill=self.TEXT_MUTED, font=self.font_section)
        conf_pct_text = f"{current_confidence:.1%}"
        draw.text((r_card_x2 - 75, conf_y - 2), conf_pct_text, fill=sign_color, font=self.font_conf)

        c_bar_x1 = r_card_x1 + 20
        c_bar_x2 = r_card_x2 - 20
        c_bar_w = c_bar_x2 - c_bar_x1
        c_bar_y = conf_y + 24

        # Track background
        draw.rounded_rectangle([(c_bar_x1, c_bar_y), (c_bar_x2, c_bar_y + 12)], radius=6, fill=(15, 23, 42))

        # Filled portion
        c_fill_w = int(c_bar_w * max(0.0, min(1.0, current_confidence)))
        if c_fill_w > 0:
            if current_confidence >= confidence_threshold:
                c_fill_color = self.SUCCESS_GREEN
            elif current_confidence >= 0.40:
                c_fill_color = self.WARNING_AMBER
            else:
                c_fill_color = self.ERROR_RED
            draw.rounded_rectangle([(c_bar_x1, c_bar_y), (c_bar_x1 + c_fill_w, c_bar_y + 12)], radius=6, fill=c_fill_color)

        # 70% threshold marker line
        thresh_x = c_bar_x1 + int(c_bar_w * confidence_threshold)
        draw.line([(thresh_x, c_bar_y - 3), (thresh_x, c_bar_y + 15)], fill=(255, 255, 255), width=2)
        draw.text((thresh_x - 30, c_bar_y + 18), "70% THRESHOLD", fill=self.TEXT_DIM, font=self.font_badge)

        # ─────────────────────────────────────────────────────────────
        # 5. RIGHT PANEL - CARD 2: RECOGNIZED SIGN SEQUENCE (Y: 320 to 560)
        # ─────────────────────────────────────────────────────────────
        c2_y1, c2_y2 = 320, 560
        draw.rounded_rectangle(
            [(r_card_x1, c2_y1), (r_card_x2, c2_y2)],
            radius=8,
            fill=self.CARD_BG,
            outline=self.CARD_BORDER,
            width=1,
        )

        draw.text((r_card_x1 + 18, c2_y1 + 14), "ACCUMULATED SIGN SEQUENCE", fill=self.TEXT_MUTED, font=self.font_section)

        # Word count badge
        w_count = len(accumulated_words)
        count_str = f"{w_count} sign{'s' if w_count != 1 else ''}"
        draw.text((r_card_x2 - 80, c2_y1 + 14), count_str, fill=self.ACCENT_CYAN, font=self.font_badge)

        # Render chips inside Card 2
        chip_start_x = r_card_x1 + 20
        chip_start_y = c2_y1 + 45
        max_chip_x = r_card_x2 - 25

        if not accumulated_words:
            draw.text(
                (chip_start_x, chip_start_y + 30),
                "No signs captured yet.",
                fill=self.TEXT_MUTED,
                font=self.font_sign_sub,
            )
            draw.text(
                (chip_start_x, chip_start_y + 55),
                "Hold any verified sign steadily to add it to the sequence.",
                fill=self.TEXT_DIM,
                font=self.font_badge,
            )
        else:
            cur_x = chip_start_x
            cur_y = chip_start_y

            for i, word in enumerate(accumulated_words):
                word_text = word.upper()
                bbox = draw.textbbox((0, 0), word_text, font=self.font_chip)
                t_w = bbox[2] - bbox[0]
                chip_w = t_w + 20
                chip_h = 28

                # Check line wrap
                if cur_x + chip_w + 28 > max_chip_x and cur_x > chip_start_x:
                    cur_x = chip_start_x
                    cur_y += 38

                if cur_y + chip_h > c2_y2 - 10:
                    draw.text((cur_x, cur_y + 5), "...", fill=self.TEXT_MUTED, font=self.font_chip)
                    break

                # Draw chip pill
                draw.rounded_rectangle(
                    [(cur_x, cur_y), (cur_x + chip_w, cur_y + chip_h)],
                    radius=6,
                    fill=(45, 55, 72),
                    outline=self.ACCENT_CYAN,
                    width=1,
                )
                draw.text((cur_x + 10, cur_y + 5), word_text, fill=self.TEXT_PRIMARY, font=self.font_chip)
                cur_x += chip_w + 8

                # Draw arrow if not last word
                if i < len(accumulated_words) - 1:
                    draw.text((cur_x, cur_y + 5), "->", fill=self.ACCENT_CYAN, font=self.font_chip)
                    cur_x += 24

        # ─────────────────────────────────────────────────────────────
        # 6. BOTTOM PANEL: GENERATED ENGLISH SENTENCE (Y: 575 to 665)
        # ─────────────────────────────────────────────────────────────
        bot_y1, bot_y2 = 575, 665
        draw.rounded_rectangle(
            [(cam_card_x1, bot_y1), (r_card_x2, bot_y2)],
            radius=8,
            fill=self.CARD_BG,
            outline=self.ACCENT_BLUE if generated_sentence else self.CARD_BORDER,
            width=1,
        )

        draw.text((cam_card_x1 + 18, bot_y1 + 12), "GENERATED ENGLISH SENTENCE", fill=self.TEXT_MUTED, font=self.font_section)

        # TTS Status Badge (Right side of sentence card)
        if is_speaking:
            tts_badge = "((o)) SPEAKING AUDIO..."
            tts_bg = (120, 53, 15)     # Dark Amber
            tts_fg = (251, 191, 36)    # Warm Amber
        elif speech_status and "Generating" in speech_status:
            tts_badge = "... RECONSTRUCTING"
            tts_bg = (30, 41, 59)
            tts_fg = self.ACCENT_CYAN
        else:
            tts_badge = "o TTS READY"
            tts_bg = (6, 78, 59)      # Dark Emerald
            tts_fg = (52, 211, 153)   # Green

        tb_box = draw.textbbox((0, 0), tts_badge, font=self.font_badge)
        tb_w = tb_box[2] - tb_box[0]
        tp_w = tb_w + 24
        tp_x2 = r_card_x2 - 18
        tp_x1 = tp_x2 - tp_w
        tp_y1 = bot_y1 + 10
        tp_y2 = tp_y1 + 24

        draw.rounded_rectangle(
            [(tp_x1, tp_y1), (tp_x2, tp_y2)],
            radius=12,
            fill=tts_bg,
            outline=tts_fg,
            width=1,
        )
        draw.text((tp_x1 + 12, tp_y1 + 4), tts_badge, fill=tts_fg, font=self.font_badge)

        # Sentence Text
        if generated_sentence:
            quote_text = f'"{generated_sentence}"'
            draw.text((cam_card_x1 + 20, bot_y1 + 40), quote_text, fill=self.TEXT_PRIMARY, font=self.font_sentence)
        else:
            draw.text(
                (cam_card_x1 + 20, bot_y1 + 42),
                "(Press ENTER to construct natural English sentence and speak aloud)",
                fill=self.TEXT_DIM,
                font=self.font_sentence,
            )

        # ─────────────────────────────────────────────────────────────
        # 7. FOOTER BAR: KEYBOARD SHORTCUTS (Y: 675 to 720)
        # ─────────────────────────────────────────────────────────────
        draw.rectangle([(0, 675), (self.width, 720)], fill=self.FOOTER_BG)
        draw.line([(0, 675), (self.width, 675)], fill=self.CARD_BORDER, width=1)

        # Key Cap 1: ENTER
        k1_x = 35
        draw.rounded_rectangle([(k1_x, 686), (k1_x + 65, 708)], radius=4, fill=(30, 41, 59), outline=self.ACCENT_CYAN, width=1)
        draw.text((k1_x + 10, 689), "ENTER", fill=self.ACCENT_CYAN, font=self.font_controls_bold)
        draw.text((k1_x + 75, 689), "Complete & Speak Sentence", fill=self.TEXT_PRIMARY, font=self.font_controls)

        # Key Cap 2: C
        k2_x = 420
        draw.rounded_rectangle([(k2_x, 686), (k2_x + 35, 708)], radius=4, fill=(30, 41, 59), outline=self.WARNING_AMBER, width=1)
        draw.text((k2_x + 12, 689), "C", fill=self.WARNING_AMBER, font=self.font_controls_bold)
        draw.text((k2_x + 45, 689), "Clear Accumulated Signs", fill=self.TEXT_PRIMARY, font=self.font_controls)

        # Key Cap 3: Q
        k3_x = 750
        draw.rounded_rectangle([(k3_x, 686), (k3_x + 35, 708)], radius=4, fill=(30, 41, 59), outline=self.ERROR_RED, width=1)
        draw.text((k3_x + 11, 689), "Q", fill=self.ERROR_RED, font=self.font_controls_bold)
        draw.text((k3_x + 45, 689), "Quit Application", fill=self.TEXT_PRIMARY, font=self.font_controls)

        # Right status string
        if speech_status:
            draw.text((self.width - 240, 689), f"Status: {speech_status}", fill=self.TEXT_MUTED, font=self.font_controls)

        # Convert canvas from RGB to BGR for OpenCV
        canvas_rgb = np.array(canvas, dtype=np.uint8)
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

        return canvas_bgr
