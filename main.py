# -*- coding: utf-8 -*-
"""
타이머 출력창(평상시) + ⚙️ 설정창(모든 설정) + 화면캡처(ROI) + OCR(닉네임) + 색상인식 + (옵션) 소리 트리거
- 지금은 "돌아가는 뼈대"를 먼저 만든 코드야.
- 너가 말한 자동화 로직(트리거 -> 닉 OCR / 소리 -> 딜레이 후 색인식) 구조까지 포함.
- PC 환경에 따라 OCR(Tesseract) / 루프백 오디오 설정은 추가 설치가 필요할 수 있어.

필수 설치(최소):
    pip install pyqt6 mss opencv-python numpy pillow rapidfuzz pytesseract

옵션(소리 트리거까지 쓰려면):
    pip install sounddevice scipy

※ Windows에서 pytesseract는 Tesseract OCR 프로그램도 설치돼 있어야 동작함.
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- GUI ---
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage, QColor, QAction
from PyQt6.QtWidgets import (
    QApplication, QWidget, QDialog, QMainWindow, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QGridLayout, QTabWidget, QComboBox, QSpinBox, QLineEdit, QSlider,
    QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox
)

# --- Screen capture / image ---
import mss
import cv2
from PIL import Image

# --- OCR & fuzzy match (optional-ish) ---
try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

# --- Audio (optional) ---
try:
    import sounddevice as sd
    from scipy.signal import butter, lfilter, spectrogram
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False


# -----------------------------
# Config / Data
# -----------------------------
@dataclass
class Rect:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def valid(self) -> bool:
        return self.w > 0 and self.h > 0


@dataclass
class TriggerConfig:
    enabled: bool = True
    target_bgr: Tuple[int, int, int] = (0, 255, 0)  # default green
    tolerance: int = 35
    consecutive_needed: int = 6
    window_frames: int = 8
    cooldown_sec: float = 2.0


@dataclass
class OCRConfig:
    samples: int = 2
    allow_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    unknown_label: str = "UNKNOWN"
    sim_threshold: int = 90
    require_gap: int = 5  # top1 - top2 gap


@dataclass
class ColorConfig:
    delay_after_event_sec: float = 2.0
    sample_frames: int = 5
    vote_enabled: bool = True


@dataclass
class SoundTemplate:
    name: str
    wav_path: str
    thr: float = 0.85
    cooldown: float = 3.0
    delay: float = 0.0
    action: str = "SCHEDULE_COLOR_READ"  # or START_TIMER / STOP_TIMER / OCR_NAMES


@dataclass
class AppConfig:
    monitor_index: int = 1  # mss: 1..N
    # ROIs on selected monitor
    roi_trigger: Rect = field(default_factory=Rect)
    roi_blue_name: Rect = field(default_factory=Rect)
    roi_red_name: Rect = field(default_factory=Rect)
    roi_blue_color: Rect = field(default_factory=Rect)
    roi_red_color: Rect = field(default_factory=Rect)

    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    color: ColorConfig = field(default_factory=ColorConfig)

    # GameID -> DisplayName
    players: Dict[str, str] = field(default_factory=dict)

    # Sound
    sound_enabled: bool = False
    sound_device: Optional[str] = None
    sound_templates: List[SoundTemplate] = field(default_factory=list)

    @staticmethod
    def from_json(path: str) -> "AppConfig":
        if not os.path.exists(path):
            return AppConfig()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def rect_from(d: dict) -> Rect:
            return Rect(**{k: int(d.get(k, 0)) for k in ["x", "y", "w", "h"]})

        cfg = AppConfig()
        cfg.monitor_index = int(raw.get("monitor_index", 1))
        cfg.roi_trigger = rect_from(raw.get("roi_trigger", {}))
        cfg.roi_blue_name = rect_from(raw.get("roi_blue_name", {}))
        cfg.roi_red_name = rect_from(raw.get("roi_red_name", {}))
        cfg.roi_blue_color = rect_from(raw.get("roi_blue_color", {}))
        cfg.roi_red_color = rect_from(raw.get("roi_red_color", {}))

        tr = raw.get("trigger", {})
        cfg.trigger = TriggerConfig(
            enabled=bool(tr.get("enabled", True)),
            target_bgr=tuple(tr.get("target_bgr", [0, 255, 0])),
            tolerance=int(tr.get("tolerance", 35)),
            consecutive_needed=int(tr.get("consecutive_needed", 6)),
            window_frames=int(tr.get("window_frames", 8)),
            cooldown_sec=float(tr.get("cooldown_sec", 2.0)),
        )

        oc = raw.get("ocr", {})
        cfg.ocr = OCRConfig(
            samples=int(oc.get("samples", 2)),
            allow_chars=str(oc.get("allow_chars", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")),
            unknown_label=str(oc.get("unknown_label", "UNKNOWN")),
            sim_threshold=int(oc.get("sim_threshold", 90)),
            require_gap=int(oc.get("require_gap", 5)),
        )

        cc = raw.get("color", {})
        cfg.color = ColorConfig(
            delay_after_event_sec=float(cc.get("delay_after_event_sec", 2.0)),
            sample_frames=int(cc.get("sample_frames", 5)),
            vote_enabled=bool(cc.get("vote_enabled", True)),
        )

        cfg.players = {str(k).upper(): str(v) for k, v in raw.get("players", {}).items()}

        cfg.sound_enabled = bool(raw.get("sound_enabled", False))
        cfg.sound_device = raw.get("sound_device", None)
        cfg.sound_templates = []
        for t in raw.get("sound_templates", []):
            cfg.sound_templates.append(SoundTemplate(
                name=str(t.get("name", "")),
                wav_path=str(t.get("wav_path", "")),
                thr=float(t.get("thr", 0.85)),
                cooldown=float(t.get("cooldown", 3.0)),
                delay=float(t.get("delay", 0.0)),
                action=str(t.get("action", "SCHEDULE_COLOR_READ")),
            ))
        return cfg

    def to_json(self, path: str) -> None:
        raw = {
            "monitor_index": self.monitor_index,
            "roi_trigger": asdict(self.roi_trigger),
            "roi_blue_name": asdict(self.roi_blue_name),
            "roi_red_name": asdict(self.roi_red_name),
            "roi_blue_color": asdict(self.roi_blue_color),
            "roi_red_color": asdict(self.roi_red_color),
            "trigger": asdict(self.trigger),
            "ocr": asdict(self.ocr),
            "color": asdict(self.color),
            "players": self.players,
            "sound_enabled": self.sound_enabled,
            "sound_device": self.sound_device,
            "sound_templates": [asdict(t) for t in self.sound_templates],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)


# -----------------------------
# Utilities
# -----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def normalize_game_id(s: str, allow: str) -> str:
    s = (s or "").upper().strip()
    s = "".join(ch for ch in s if ch in allow)
    return s


def bgr_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32)))


def capture_monitor_np(monitor_index: int) -> np.ndarray:
    """
    Returns BGR numpy image of selected monitor.
    """
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index < 1 or monitor_index >= len(monitors):
            monitor_index = 1
        mon = monitors[monitor_index]
        img = np.array(sct.grab(mon))  # BGRA
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr


def crop(bgr: np.ndarray, r: Rect) -> np.ndarray:
    h, w = bgr.shape[:2]
    x = clamp(r.x, 0, w - 1)
    y = clamp(r.y, 0, h - 1)
    ww = clamp(r.w, 0, w - x)
    hh = clamp(r.h, 0, h - y)
    if ww <= 0 or hh <= 0:
        return bgr[0:0, 0:0]
    return bgr[y:y + hh, x:x + ww]


def dominant_color_name_from_roi(bgr_roi: np.ndarray) -> str:
    """
    아주 단순하지만 튼튼한 방식:
    - HSV로 바꿈
    - 중앙값으로 대표색 잡음
    - 빨/주/노/초/파/보/흰/검/회 분류
    """
    if bgr_roi.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.float32)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)

    # 너무 어두움/너무 무채색이면: black/white/gray
    s_med = float(np.median(S))
    v_med = float(np.median(V))
    if v_med < 50:
        return "BLACK"
    if s_med < 30:
        if v_med > 200:
            return "WHITE"
        return "GRAY"

    h_med = float(np.median(H))  # 0..179 (OpenCV)
    # hue bins (rough)
    if h_med < 10 or h_med >= 170:
        return "RED"
    if 10 <= h_med < 20:
        return "ORANGE"
    if 20 <= h_med < 35:
        return "YELLOW"
    if 35 <= h_med < 85:
        return "GREEN"
    if 85 <= h_med < 130:
        return "BLUE"
    if 130 <= h_med < 170:
        return "PURPLE"
    return "UNKNOWN"


def ocr_one_line_english(bgr_roi: np.ndarray) -> str:
    """
    OCR: 영어 한 줄(닉네임) 용도.
    """
    if not HAS_TESSERACT:
        return ""
    if bgr_roi.size == 0:
        return ""

    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    # 대비 강화 + 이진화
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil = Image.fromarray(th)
    config = "--psm 7 --oem 3"
    text = pytesseract.image_to_string(pil, config=config)
    return (text or "").strip()


def best_match_id(ocr_text: str, id_map: Dict[str, str], cfg: OCRConfig) -> Tuple[str, int]:
    """
    등록된 아이디들 중 가장 비슷한 GAME_ID 찾기.
    반환: (matched_id, score)
    """
    cleaned = normalize_game_id(ocr_text, cfg.allow_chars)
    if not cleaned:
        return cfg.unknown_label, 0

    ids = list(id_map.keys())
    if not ids:
        # 등록이 없으면 그냥 OCR값을 그대로 ID로 쓸 수도 있음(원하면)
        return cleaned, 100

    # 완전일치
    if cleaned in id_map:
        return cleaned, 100

    if HAS_RAPIDFUZZ:
        # top2 뽑아서 gap 확인
        results = rf_process.extract(cleaned, ids, scorer=rf_fuzz.ratio, limit=2)
        if not results:
            return cfg.unknown_label, 0
        top1_id, top1_score, _ = results[0]
        top2_score = results[1][1] if len(results) > 1 else 0

        if int(top1_score) < cfg.sim_threshold:
            return cfg.unknown_label, int(top1_score)
        if int(top1_score) - int(top2_score) < cfg.require_gap:
            return cfg.unknown_label, int(top1_score)
        return str(top1_id), int(top1_score)

    # fallback: 간단한 유사도(비추천이지만 최소 동작)
    def simple_ratio(a: str, b: str) -> int:
        common = sum(1 for ch in a if ch in b)
        return int(100 * common / max(1, max(len(a), len(b))))

    best_id, best_sc = cfg.unknown_label, 0
    for gid in ids:
        sc = simple_ratio(cleaned, gid)
        if sc > best_sc:
            best_id, best_sc = gid, sc
    if best_sc >= cfg.sim_threshold:
        return best_id, best_sc
    return cfg.unknown_label, best_sc


# -----------------------------
# ROI Selection Widget
# -----------------------------
class RoiPickerDialog(QDialog):
    """
    스샷 위에서 드래그로 ROI 박스 지정.
    """
    def __init__(self, parent: QWidget, bgr_frame: np.ndarray, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.frame = bgr_frame
        self.start = None
        self.end = None
        self.rect = Rect()

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(800, 450)

        btn_ok = QPushButton("확인")
        btn_cancel = QPushButton("취소")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        lay_btn = QHBoxLayout()
        lay_btn.addStretch(1)
        lay_btn.addWidget(btn_ok)
        lay_btn.addWidget(btn_cancel)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("마우스로 드래그해서 박스를 그려줘"))
        layout.addWidget(self.label)
        layout.addLayout(lay_btn)
        self.setLayout(layout)

        self._update_pixmap()

        self.label.mousePressEvent = self._mouse_press
        self.label.mouseMoveEvent = self._mouse_move
        self.label.mouseReleaseEvent = self._mouse_release

    def _update_pixmap(self):
        draw = self.frame.copy()
        if self.start and self.end:
            x1, y1 = self.start
            x2, y2 = self.end
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 255), 2)

        rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.label.setPixmap(pix.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _map_to_image_coords(self, pos) -> Tuple[int, int]:
        """
        QLabel에 fit된 pixmap 좌표를 원본 이미지 좌표로 근사 변환.
        """
        pix = self.label.pixmap()
        if pix is None:
            return 0, 0

        lbl_w = self.label.width()
        lbl_h = self.label.height()
        img_h, img_w = self.frame.shape[:2]

        # 비율 맞춰서 중앙정렬 된 pixmap 크기 추정
        scale = min(lbl_w / img_w, lbl_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        offset_x = (lbl_w - disp_w) // 2
        offset_y = (lbl_h - disp_h) // 2

        x = int((pos.x() - offset_x) / max(scale, 1e-6))
        y = int((pos.y() - offset_y) / max(scale, 1e-6))
        x = clamp(x, 0, img_w - 1)
        y = clamp(y, 0, img_h - 1)
        return x, y

    def _mouse_press(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.start = self._map_to_image_coords(ev.pos())
            self.end = self.start
            self._update_pixmap()

    def _mouse_move(self, ev):
        if self.start is not None:
            self.end = self._map_to_image_coords(ev.pos())
            self._update_pixmap()

    def _mouse_release(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self.start and self.end:
            x1, y1 = self.start
            x2, y2 = self.end
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            self.rect = Rect(x=x, y=y, w=w, h=h)
            self._update_pixmap()


# -----------------------------
# Background Watchers (threads)
# -----------------------------
class ScreenWatcher(QObject):
    """
    화면 트리거(버튼색) 감시 -> 이벤트 발생
    """
    trigger_fired = pyqtSignal()

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._cooldown_until = 0.0
        self._window: List[bool] = []

    def start(self):
        self._stop = False
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            time.sleep(0.05)  # ~20fps
            if not self.cfg.trigger.enabled:
                continue
            if not self.cfg.roi_trigger.valid():
                continue
            now = time.time()
            if now < self._cooldown_until:
                continue

            frame = capture_monitor_np(self.cfg.monitor_index)
            roi = crop(frame, self.cfg.roi_trigger)
            if roi.size == 0:
                continue
            # 대표색(중앙값)
            b = int(np.median(roi[..., 0]))
            g = int(np.median(roi[..., 1]))
            r = int(np.median(roi[..., 2]))
            dist = bgr_distance((b, g, r), self.cfg.trigger.target_bgr)
            is_hit = dist <= float(self.cfg.trigger.tolerance)

            self._window.append(is_hit)
            if len(self._window) > self.cfg.trigger.window_frames:
                self._window.pop(0)

            if sum(self._window) >= self.cfg.trigger.consecutive_needed and len(self._window) >= self.cfg.trigger.window_frames:
                # fire
                self._cooldown_until = time.time() + self.cfg.trigger.cooldown_sec
                self._window.clear()
                self.trigger_fired.emit()


class Controller(QObject):
    """
    이벤트 받아서:
    - 닉네임 OCR
    - 색상 인식(딜레이 후 1회)
    - 타이머 UI 갱신
    """
    ui_update = pyqtSignal(dict)  # {"blue_name":..., "red_name":..., "blue_color":..., "red_color":...}
    status_update = pyqtSignal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self._lock_names = False
        self._lock_colors = False

    def unlock_all(self):
        self._lock_names = False
        self._lock_colors = False

    def on_screen_trigger_for_names(self):
        # 닉네임 인식
        if self._lock_names:
            return
        self.status_update.emit("닉네임 인식중...")
        result = self._read_names()
        self.ui_update.emit(result)
        self._lock_names = True
        self.status_update.emit("닉네임 확정 완료")

    def schedule_color_read(self):
        # 딜레이 후 색상 인식(1회)
        if self._lock_colors:
            return
        delay = self.cfg.color.delay_after_event_sec
        self.status_update.emit(f"색상 인식 대기({delay:.1f}s)...")
        t = threading.Thread(target=self._delayed_color, daemon=True)
        t.start()

    def _delayed_color(self):
        time.sleep(max(0.0, self.cfg.color.delay_after_event_sec))
        if self._lock_colors:
            return
        self.status_update.emit("색상 인식중...")
        result = self._read_colors()
        self.ui_update.emit(result)
        self._lock_colors = True
        self.status_update.emit("색상 확정 완료")

    def _read_names(self) -> dict:
        frame = capture_monitor_np(self.cfg.monitor_index)
        blue = ""
        red = ""
        # 샘플링
        for _ in range(max(1, self.cfg.ocr.samples)):
            if self.cfg.roi_blue_name.valid():
                blue_txt = ocr_one_line_english(crop(frame, self.cfg.roi_blue_name))
                blue = blue_txt or blue
            if self.cfg.roi_red_name.valid():
                red_txt = ocr_one_line_english(crop(frame, self.cfg.roi_red_name))
                red = red_txt or red
            time.sleep(0.05)

        blue_id, _ = best_match_id(blue, self.cfg.players, self.cfg.ocr)
        red_id, _ = best_match_id(red, self.cfg.players, self.cfg.ocr)
        blue_disp = self.cfg.players.get(blue_id, blue_id)
        red_disp = self.cfg.players.get(red_id, red_id)
        return {"blue_name": blue_disp, "red_name": red_disp}

    def _read_colors(self) -> dict:
        votes_blue = []
        votes_red = []
        for _ in range(max(1, self.cfg.color.sample_frames)):
            frame = capture_monitor_np(self.cfg.monitor_index)
            if self.cfg.roi_blue_color.valid():
                votes_blue.append(dominant_color_name_from_roi(crop(frame, self.cfg.roi_blue_color)))
            if self.cfg.roi_red_color.valid():
                votes_red.append(dominant_color_name_from_roi(crop(frame, self.cfg.roi_red_color)))
            time.sleep(0.05)

        def vote(vs: List[str]) -> str:
            if not vs:
                return "UNKNOWN"
            if not self.cfg.color.vote_enabled:
                return vs[-1]
            from collections import Counter
            return Counter(vs).most_common(1)[0][0]

        return {"blue_color": vote(votes_blue), "red_color": vote(votes_red)}


# -----------------------------
# GUI: Timer (Output) Window
# -----------------------------
COLOR_NAME_TO_RGB = {
    "RED": (220, 60, 60),
    "ORANGE": (240, 140, 40),
    "YELLOW": (230, 210, 60),
    "GREEN": (60, 190, 90),
    "BLUE": (60, 120, 230),
    "PURPLE": (160, 90, 210),
    "WHITE": (240, 240, 240),
    "BLACK": (20, 20, 20),
    "GRAY": (140, 140, 140),
    "UNKNOWN": (80, 80, 80),
}


class ColorBox(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(26, 26)
        self.setStyleSheet("border: 1px solid #333; background: #444;")

    def set_color_name(self, name: str):
        rgb = COLOR_NAME_TO_RGB.get(name, COLOR_NAME_TO_RGB["UNKNOWN"])
        self.setStyleSheet(
            f"border: 1px solid #333; background: rgb({rgb[0]},{rgb[1]},{rgb[2]});"
        )


class TimerWindow(QMainWindow):
    open_settings = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Box Timer (Output)")
        self.setFixedSize(520, 180)

        # Menu (업데이트는 나중에 하기로 해서 일단 비워둠)
        menubar = self.menuBar()
        menu_file = menubar.addMenu("파일")
        act_quit = QAction("종료", self)
        act_quit.triggered.connect(self.close)
        menu_file.addAction(act_quit)

        # layout
        root = QWidget()
        self.setCentralWidget(root)

        self.lbl_status = QLabel("대기중")
        self.lbl_status.setStyleSheet("color:#aaa;")

        self.lbl_round = QLabel("RD 1 / 3")
        self.lbl_round.setStyleSheet("font-size:18px; font-weight:700; color:#ddd;")

        self.lbl_time = QLabel("3:00")
        self.lbl_time.setStyleSheet("font-size:44px; font-weight:900; color:#fff;")

        self.blue_name = QLabel("BLUE")
        self.blue_name.setStyleSheet("font-size:18px; font-weight:800; color:#fff; background:#1e55ff; padding:6px;")
        self.red_name = QLabel("RED")
        self.red_name.setStyleSheet("font-size:18px; font-weight:800; color:#fff; background:#d93b3b; padding:6px;")

        self.blue_color = ColorBox()
        self.red_color = ColorBox()

        self.btn_start = QPushButton("▶ 시작/일시정지")
        self.btn_reset = QPushButton("⟲ 리셋")
        self.btn_gear = QPushButton("⚙️ 설정")

        # Timer state
        self.total_rounds = 3
        self.current_round = 1
        self.seconds_left = 180
        self.running = False
        self._qtimer = QTimer(self)
        self._qtimer.timeout.connect(self._tick)

        self.btn_start.clicked.connect(self.toggle_timer)
        self.btn_reset.clicked.connect(self.reset_timer)
        self.btn_gear.clicked.connect(lambda: self.open_settings.emit())

        # arrange
        grid = QGridLayout()
        grid.addWidget(self.lbl_round, 0, 0, 1, 2)
        grid.addWidget(self.lbl_status, 0, 2, 1, 2, alignment=Qt.AlignmentFlag.AlignRight)

        grid.addWidget(self.lbl_time, 1, 0, 2, 2)

        # right panel: names + colors
        right = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.blue_color)
        row1.addWidget(self.blue_name, 1)
        row2 = QHBoxLayout()
        row2.addWidget(self.red_color)
        row2.addWidget(self.red_name, 1)
        right.addLayout(row1)
        right.addLayout(row2)

        grid.addLayout(right, 1, 2, 2, 2)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_reset)
        btns.addWidget(self.btn_gear)

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addStretch(1)
        layout.addLayout(btns)

        root.setLayout(layout)
        self.setStyleSheet("background:#111;")

        self._refresh_time()

    def set_status(self, s: str):
        self.lbl_status.setText(s)

    def set_names(self, blue: str, red: str):
        self.blue_name.setText(blue or "BLUE")
        self.red_name.setText(red or "RED")

    def set_colors(self, blue_color: str, red_color: str):
        self.blue_color.set_color_name(blue_color)
        self.red_color.set_color_name(red_color)

    def _refresh_time(self):
        m = self.seconds_left // 60
        s = self.seconds_left % 60
        self.lbl_time.setText(f"{m}:{s:02d}")
        self.lbl_round.setText(f"RD {self.current_round} / {self.total_rounds}")

    def toggle_timer(self):
        self.running = not self.running
        if self.running:
            self._qtimer.start(1000)
        else:
            self._qtimer.stop()

    def reset_timer(self):
        self.running = False
        self._qtimer.stop()
        self.current_round = 1
        self.seconds_left = 180
        self._refresh_time()

    def _tick(self):
        if self.seconds_left > 0:
            self.seconds_left -= 1
        else:
            # round end
            self.running = False
            self._qtimer.stop()
        self._refresh_time()


# -----------------------------
# Settings Dialog
# -----------------------------
class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget, cfg: AppConfig):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.resize(920, 620)
        self.cfg = cfg

        self.tabs = QTabWidget()

        # Tabs
        self.tab_quick = QWidget()
        self.tab_players = QWidget()
        self.tab_trigger = QWidget()
        self.tab_color = QWidget()

        self.tabs.addTab(self.tab_quick, "빠른 시작")
        self.tabs.addTab(self.tab_players, "아이디/닉네임")
        self.tabs.addTab(self.tab_trigger, "트리거/OCR")
        self.tabs.addTab(self.tab_color, "색상")

        # bottom buttons
        self.btn_apply = QPushButton("적용")
        self.btn_save = QPushButton("저장")
        self.btn_close = QPushButton("닫기")
        self.btn_apply.clicked.connect(self.apply_only)
        self.btn_save.clicked.connect(self.save_profile)
        self.btn_close.clicked.connect(self.close)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(self.btn_apply)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_close)

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self._build_quick()
        self._build_players()
        self._build_trigger()
        self._build_color()

    # ---- Quick tab ----
    def _build_quick(self):
        lay = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("모니터 선택:"))
        self.cmb_monitor = QComboBox()
        self._refresh_monitors()
        row.addWidget(self.cmb_monitor, 1)

        self.btn_preview = QPushButton("미리보기(캡처)")
        self.btn_preview.clicked.connect(self.preview_capture)
        row.addWidget(self.btn_preview)

        lay.addLayout(row)

        # ROI buttons
        roi_grid = QGridLayout()
        self.btn_set_trigger = QPushButton("트리거 ROI 지정")
        self.btn_set_blue_name = QPushButton("블루 닉네임 ROI 지정")
        self.btn_set_red_name = QPushButton("레드 닉네임 ROI 지정")
        self.btn_set_blue_color = QPushButton("블루 색상칸 ROI 지정")
        self.btn_set_red_color = QPushButton("레드 색상칸 ROI 지정")

        self.lbl_trigger = QLabel(self._roi_text(self.cfg.roi_trigger))
        self.lbl_bname = QLabel(self._roi_text(self.cfg.roi_blue_name))
        self.lbl_rname = QLabel(self._roi_text(self.cfg.roi_red_name))
        self.lbl_bcol = QLabel(self._roi_text(self.cfg.roi_blue_color))
        self.lbl_rcol = QLabel(self._roi_text(self.cfg.roi_red_color))

        self.btn_set_trigger.clicked.connect(lambda: self.pick_roi("트리거 ROI", "roi_trigger"))
        self.btn_set_blue_name.clicked.connect(lambda: self.pick_roi("블루 닉네임 ROI", "roi_blue_name"))
        self.btn_set_red_name.clicked.connect(lambda: self.pick_roi("레드 닉네임 ROI", "roi_red_name"))
        self.btn_set_blue_color.clicked.connect(lambda: self.pick_roi("블루 색상칸 ROI", "roi_blue_color"))
        self.btn_set_red_color.clicked.connect(lambda: self.pick_roi("레드 색상칸 ROI", "roi_red_color"))

        roi_grid.addWidget(self.btn_set_trigger, 0, 0)
        roi_grid.addWidget(self.lbl_trigger, 0, 1)
        roi_grid.addWidget(self.btn_set_blue_name, 1, 0)
        roi_grid.addWidget(self.lbl_bname, 1, 1)
        roi_grid.addWidget(self.btn_set_red_name, 2, 0)
        roi_grid.addWidget(self.lbl_rname, 2, 1)
        roi_grid.addWidget(self.btn_set_blue_color, 3, 0)
        roi_grid.addWidget(self.lbl_bcol, 3, 1)
        roi_grid.addWidget(self.btn_set_red_color, 4, 0)
        roi_grid.addWidget(self.lbl_rcol, 4, 1)

        lay.addLayout(roi_grid)

        # Test buttons
        test_row = QHBoxLayout()
        self.btn_test_ocr = QPushButton("닉네임 테스트(OCR)")
        self.btn_test_col = QPushButton("색상 테스트")
        self.btn_test_ocr.clicked.connect(self.test_ocr)
        self.btn_test_col.clicked.connect(self.test_color)
        test_row.addWidget(self.btn_test_ocr)
        test_row.addWidget(self.btn_test_col)
        test_row.addStretch(1)
        lay.addLayout(test_row)

        # tips
        tip = QLabel("TIP: 트리거/닉네임/색상칸 ROI를 먼저 잡고 테스트해봐!")
        tip.setStyleSheet("color:#888;")
        lay.addWidget(tip)

        self.tab_quick.setLayout(lay)

    def _refresh_monitors(self):
        self.cmb_monitor.clear()
        with mss.mss() as sct:
            mons = sct.monitors  # 0 is all
            for i in range(1, len(mons)):
                m = mons[i]
                self.cmb_monitor.addItem(f"모니터 {i} ({m['width']}x{m['height']})", i)
        # set current
        idx = max(0, self.cmb_monitor.findData(self.cfg.monitor_index))
        self.cmb_monitor.setCurrentIndex(idx)

    def _roi_text(self, r: Rect) -> str:
        if not r.valid():
            return "미설정"
        return f"x={r.x}, y={r.y}, w={r.w}, h={r.h}"

    def preview_capture(self):
        self.cfg.monitor_index = int(self.cmb_monitor.currentData())
        frame = capture_monitor_np(self.cfg.monitor_index)
        dlg = RoiPickerDialog(self, frame, "미리보기(닫기=취소)")
        dlg.exec()

    def pick_roi(self, title: str, attr_name: str):
        self.cfg.monitor_index = int(self.cmb_monitor.currentData())
        frame = capture_monitor_np(self.cfg.monitor_index)
        dlg = RoiPickerDialog(self, frame, title)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            rect = dlg.rect
            setattr(self.cfg, attr_name, rect)
            # refresh labels
            self.lbl_trigger.setText(self._roi_text(self.cfg.roi_trigger))
            self.lbl_bname.setText(self._roi_text(self.cfg.roi_blue_name))
            self.lbl_rname.setText(self._roi_text(self.cfg.roi_red_name))
            self.lbl_bcol.setText(self._roi_text(self.cfg.roi_blue_color))
            self.lbl_rcol.setText(self._roi_text(self.cfg.roi_red_color))

    def test_ocr(self):
        if not HAS_TESSERACT:
            QMessageBox.warning(self, "OCR 불가", "pytesseract 또는 Tesseract 설치가 필요해.")
            return
        self.cfg.monitor_index = int(self.cmb_monitor.currentData())
        frame = capture_monitor_np(self.cfg.monitor_index)
        b = ocr_one_line_english(crop(frame, self.cfg.roi_blue_name)) if self.cfg.roi_blue_name.valid() else ""
        r = ocr_one_line_english(crop(frame, self.cfg.roi_red_name)) if self.cfg.roi_red_name.valid() else ""
        bid, bsc = best_match_id(b, self.cfg.players, self.cfg.ocr)
        rid, rsc = best_match_id(r, self.cfg.players, self.cfg.ocr)
        QMessageBox.information(
            self,
            "OCR 테스트",
            f"BLUE OCR: {b}\n→ 매칭: {bid} ({bsc})\n\n"
            f"RED OCR: {r}\n→ 매칭: {rid} ({rsc})",
        )

    def test_color(self):
        self.cfg.monitor_index = int(self.cmb_monitor.currentData())
        frame = capture_monitor_np(self.cfg.monitor_index)
        bc = dominant_color_name_from_roi(crop(frame, self.cfg.roi_blue_color)) if self.cfg.roi_blue_color.valid() else "UNKNOWN"
        rc = dominant_color_name_from_roi(crop(frame, self.cfg.roi_red_color)) if self.cfg.roi_red_color.valid() else "UNKNOWN"
        QMessageBox.information(self, "색상 테스트", f"BLUE 색: {bc}\nRED 색: {rc}")

    # ---- Players tab ----
    def _build_players(self):
        lay = QVBoxLayout()

        row = QHBoxLayout()
        self.txt_new_id = QLineEdit()
        self.txt_new_id.setPlaceholderText("GAME_ID (대문자)")
        self.txt_new_name = QLineEdit()
        self.txt_new_name.setPlaceholderText("표시 닉네임")
        btn_add = QPushButton("추가")
        btn_del = QPushButton("삭제(선택행)")
        btn_add.clicked.connect(self.add_player)
        btn_del.clicked.connect(self.del_player)

        row.addWidget(self.txt_new_id)
        row.addWidget(self.txt_new_name)
        row.addWidget(btn_add)
        row.addWidget(btn_del)
        lay.addLayout(row)

        self.tbl_players = QTableWidget(0, 2)
        self.tbl_players.setHorizontalHeaderLabels(["GAME_ID", "표시 닉네임"])
        self.tbl_players.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_players.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.tbl_players)

        self._reload_players_table()
        self.tab_players.setLayout(lay)

    def _reload_players_table(self):
        self.tbl_players.setRowCount(0)
        for gid, name in sorted(self.cfg.players.items()):
            r = self.tbl_players.rowCount()
            self.tbl_players.insertRow(r)
            self.tbl_players.setItem(r, 0, QTableWidgetItem(gid))
            self.tbl_players.setItem(r, 1, QTableWidgetItem(name))

    def add_player(self):
        gid = (self.txt_new_id.text() or "").upper().strip()
        name = (self.txt_new_name.text() or "").strip()
        if not gid or not name:
            return
        self.cfg.players[gid] = name
        self.txt_new_id.clear()
        self.txt_new_name.clear()
        self._reload_players_table()

    def del_player(self):
        row = self.tbl_players.currentRow()
        if row < 0:
            return
        gid = self.tbl_players.item(row, 0).text()
        if gid in self.cfg.players:
            del self.cfg.players[gid]
        self._reload_players_table()

    # ---- Trigger/OCR tab ----
    def _build_trigger(self):
        lay = QVBoxLayout()

        chk = QCheckBox("화면 트리거(버튼 색) 사용")
        chk.setChecked(self.cfg.trigger.enabled)
        chk.stateChanged.connect(lambda s: setattr(self.cfg.trigger, "enabled", bool(s)))
        lay.addWidget(chk)

        row = QHBoxLayout()
        row.addWidget(QLabel("목표색(BGR):"))
        self.txt_b = QSpinBox()
        self.txt_b.setRange(0, 255)
        self.txt_b.setValue(int(self.cfg.trigger.target_bgr[0]))
        self.txt_g = QSpinBox()
        self.txt_g.setRange(0, 255)
        self.txt_g.setValue(int(self.cfg.trigger.target_bgr[1]))
        self.txt_r = QSpinBox()
        self.txt_r.setRange(0, 255)
        self.txt_r.setValue(int(self.cfg.trigger.target_bgr[2]))
        row.addWidget(QLabel("B"))
        row.addWidget(self.txt_b)
        row.addWidget(QLabel("G"))
        row.addWidget(self.txt_g)
        row.addWidget(QLabel("R"))
        row.addWidget(self.txt_r)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("허용오차:"))
        self.sp_tol = QSpinBox()
        self.sp_tol.setRange(0, 200)
        self.sp_tol.setValue(self.cfg.trigger.tolerance)
        row2.addWidget(self.sp_tol)
        row2.addWidget(QLabel("연속 판정(window/need):"))
        self.sp_win = QSpinBox()
        self.sp_win.setRange(1, 30)
        self.sp_win.setValue(self.cfg.trigger.window_frames)
        self.sp_need = QSpinBox()
        self.sp_need.setRange(1, 30)
        self.sp_need.setValue(self.cfg.trigger.consecutive_needed)
        row2.addWidget(self.sp_win)
        row2.addWidget(QLabel("/"))
        row2.addWidget(self.sp_need)
        row2.addWidget(QLabel("쿨다운(초):"))
        self.sp_cd = QSpinBox()
        self.sp_cd.setRange(0, 30)
        self.sp_cd.setValue(int(self.cfg.trigger.cooldown_sec))
        row2.addWidget(self.sp_cd)
        row2.addStretch(1)
        lay.addLayout(row2)

        # OCR settings
        lay.addWidget(QLabel("OCR 보정 설정"))
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("샘플링 횟수:"))
        self.sp_samples = QSpinBox()
        self.sp_samples.setRange(1, 5)
        self.sp_samples.setValue(self.cfg.ocr.samples)
        row3.addWidget(self.sp_samples)
        row3.addWidget(QLabel("유사도 임계값:"))
        self.sp_sim = QSpinBox()
        self.sp_sim.setRange(0, 100)
        self.sp_sim.setValue(self.cfg.ocr.sim_threshold)
        row3.addWidget(self.sp_sim)
        row3.addWidget(QLabel("1등-2등 차이:"))
        self.sp_gap = QSpinBox()
        self.sp_gap.setRange(0, 50)
        self.sp_gap.setValue(self.cfg.ocr.require_gap)
        row3.addWidget(self.sp_gap)
        row3.addStretch(1)
        lay.addLayout(row3)

        self.tab_trigger.setLayout(lay)

    # ---- Color tab ----
    def _build_color(self):
        lay = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("색상 인식 딜레이(초):"))
        self.sp_delay = QSpinBox()
        self.sp_delay.setRange(0, 20)
        self.sp_delay.setValue(int(self.cfg.color.delay_after_event_sec))
        row.addWidget(self.sp_delay)
        row.addWidget(QLabel("샘플 프레임 수:"))
        self.sp_frames = QSpinBox()
        self.sp_frames.setRange(1, 20)
        self.sp_frames.setValue(int(self.cfg.color.sample_frames))
        row.addWidget(self.sp_frames)
        self.chk_vote = QCheckBox("다수결 사용")
        self.chk_vote.setChecked(self.cfg.color.vote_enabled)
        row.addWidget(self.chk_vote)
        row.addStretch(1)
        lay.addLayout(row)

        tip = QLabel("TIP: 색상은 '딱 그 순간'에 3~5장 찍어서 다수결로 결정하면 안정적이야.")
        tip.setStyleSheet("color:#888;")
        lay.addWidget(tip)
        self.tab_color.setLayout(lay)

    # ---- Apply / Save ----
    def apply_only(self):
        self.cfg.monitor_index = int(self.cmb_monitor.currentData())
        self.cfg.trigger.target_bgr = (int(self.txt_b.value()), int(self.txt_g.value()), int(self.txt_r.value()))
        self.cfg.trigger.tolerance = int(self.sp_tol.value())
        self.cfg.trigger.window_frames = int(self.sp_win.value())
        self.cfg.trigger.consecutive_needed = int(self.sp_need.value())
        self.cfg.trigger.cooldown_sec = float(self.sp_cd.value())

        self.cfg.ocr.samples = int(self.sp_samples.value())
        self.cfg.ocr.sim_threshold = int(self.sp_sim.value())
        self.cfg.ocr.require_gap = int(self.sp_gap.value())

        self.cfg.color.delay_after_event_sec = float(self.sp_delay.value())
        self.cfg.color.sample_frames = int(self.sp_frames.value())
        self.cfg.color.vote_enabled = bool(self.chk_vote.isChecked())

        QMessageBox.information(self, "적용", "설정 적용 완료!")

    def save_profile(self):
        self.apply_only()
        path, _ = QFileDialog.getSaveFileName(self, "설정 저장", "profile.json", "JSON (*.json)")
        if not path:
            return
        self.cfg.to_json(path)
        QMessageBox.information(self, "저장", f"저장 완료!\n{path}")


# -----------------------------
# App Wiring
# -----------------------------
class MainApp(QObject):
    def __init__(self, cfg_path: str):
        super().__init__()
        self.cfg_path = cfg_path
        self.cfg = AppConfig.from_json(cfg_path)

        self.timer_win = TimerWindow()
        self.settings_dlg = None

        self.controller = Controller(self.cfg)
        self.watcher = ScreenWatcher(self.cfg)

        # connect
        self.timer_win.open_settings.connect(self.open_settings)
        self.watcher.trigger_fired.connect(self.controller.on_screen_trigger_for_names)

        self.controller.ui_update.connect(self.apply_ui_update)
        self.controller.status_update.connect(self.timer_win.set_status)

        # start watcher
        self.watcher.start()

        # 첫 상태
        self.timer_win.set_status("대기중 (트리거 감시중)")
        if not HAS_TESSERACT:
            self.timer_win.set_status("주의: OCR 사용하려면 Tesseract 설치 필요")

    def open_settings(self):
        if self.settings_dlg and self.settings_dlg.isVisible():
            self.settings_dlg.raise_()
            return
        self.settings_dlg = SettingsDialog(self.timer_win, self.cfg)
        self.settings_dlg.finished.connect(self.on_settings_closed)
        self.settings_dlg.show()

    def on_settings_closed(self, _):
        # 닫힐 때 자동 저장(원하면 끄면 됨)
        try:
            self.cfg.to_json(self.cfg_path)
        except Exception:
            pass

    def apply_ui_update(self, d: dict):
        # d may have partial fields
        if "blue_name" in d or "red_name" in d:
            self.timer_win.set_names(
                d.get("blue_name", self.timer_win.blue_name.text()),
                d.get("red_name", self.timer_win.red_name.text()),
            )
        if "blue_color" in d or "red_color" in d:
            self.timer_win.set_colors(
                d.get("blue_color", "UNKNOWN"),
                d.get("red_color", "UNKNOWN"),
            )

    def show(self):
        self.timer_win.show()


# -----------------------------
# Entry
# -----------------------------
def main():
    app = QApplication(sys.argv)

    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    main_app = MainApp(cfg_path)
    main_app.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
