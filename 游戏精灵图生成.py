# video_sprites_gui_enhanced_v7_6.py
# Enhanced version v7.6 - 极速专业版 (完整修复版 + 算法增强)
# 
# 【v7.6 算法增强与修复】
# - 优化背景移除后的边缘清理 (使用形态学开运算)
# - 优化孤立色块去除 (使用连通域分析，智能保留主体)
# - 修复试用退出机制
# - 包含完整的激活验证流程
#
# Dependencies:
# pip install opencv-python pillow PyQt5 numpy rembg imageio imageio-ffmpeg onnxruntime

import sys
import os
import math
import traceback
import shutil
import hashlib
import subprocess
import threading
import queue
import time
import json
import gc
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, date
from io import BytesIO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 依赖库检查 ====================
try:
    import winsound
    HAS_WINSOUND = True
except:
    HAS_WINSOUND = False

try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("错误: Pillow 未安装")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("错误: numpy 未安装")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("错误: opencv-python 未安装")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("错误: imageio 未安装")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QListWidget, QSpinBox, QCheckBox,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog, QProgressBar, QMessageBox,
    QTextEdit, QComboBox, QRadioButton, QButtonGroup, QGroupBox, QDoubleSpinBox,
    QTabWidget, QLineEdit, QDialog, QFrame, QToolTip, QSplitter, QPlainTextEdit,
    QListWidgetItem, QColorDialog, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QSizePolicy, QSlider, QSpacerItem, QStackedWidget, QFormLayout,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QSize, QSignalBlocker
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor, QIcon, QPixmap, QImage, QPainter, QPen, QBrush

# ==================== 设置模型路径环境变量 ====================
BIEMO_DIR = Path.cwd() / "biemo"
BIEMO_DIR.mkdir(parents=True, exist_ok=True)
os.environ["U2NET_HOME"] = str(BIEMO_DIR / "models")
os.environ["REMBG_HOME"] = str(BIEMO_DIR / "models")
(BIEMO_DIR / "models").mkdir(parents=True, exist_ok=True)

# ==================== 配置管理器 ====================
class ConfigManager:
    BIEMO_BASE = Path.cwd() / "biemo"
    CONFIG_FILE = BIEMO_BASE / "config.json"
    MODELS_CONFIG_FILE = BIEMO_BASE / "models_config.json"
    LICENSE_FILE = BIEMO_BASE / "tools" / "license.key"
    
    DEFAULT_CONFIG = {
        "model_dir": str(BIEMO_BASE / "models"),
        "output_paths": {
            "sprite": str(BIEMO_BASE / "output_sprites"),
            "extract": str(BIEMO_BASE / "output_images"),
            "video": str(BIEMO_BASE / "output_videos"),
            "gif": str(BIEMO_BASE / "output_gifs"),
            "single": str(BIEMO_BASE / "output_single"),
            "beiou": str(BIEMO_BASE / "output_beiou"),
        },
        "default_model": "isnet-general-use",
        "default_threads": 4,
        "enable_sound": True,
        "auto_open_folder": True,
        "model_mirrors": {"global": "", "cn": ""}
    }
    _config = None
    
    @classmethod
    def init_directories(cls):
        cls.BIEMO_BASE.mkdir(parents=True, exist_ok=True)
        (cls.BIEMO_BASE / "models").mkdir(parents=True, exist_ok=True)
        (cls.BIEMO_BASE / "tools").mkdir(parents=True, exist_ok=True)
        for key in cls.DEFAULT_CONFIG["output_paths"]:
            path = Path(cls.DEFAULT_CONFIG["output_paths"][key])
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_biemo_dir(cls): return cls.BIEMO_BASE
    @classmethod
    def get_license_file(cls): return str(cls.LICENSE_FILE)
    
    @classmethod
    def load(cls):
        if cls._config is not None: return cls._config
        cls.init_directories()
        cls._config = cls.DEFAULT_CONFIG.copy()
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    for key, value in saved.items():
                        if key == "output_paths": cls._config["output_paths"].update(value)
                        else: cls._config[key] = value
        except Exception: pass
        return cls._config
    
    @classmethod
    def save(cls):
        try:
            cls.BIEMO_BASE.mkdir(parents=True, exist_ok=True)
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cls._config, f, indent=2, ensure_ascii=False)
        except Exception: pass
    
    @classmethod
    def get(cls, key, default=None): return cls.load().get(key, default)
    @classmethod
    def set(cls, key, value):
        config = cls.load(); config[key] = value; cls.save()
    
    @classmethod
    def get_model_dir(cls): return str(cls.BIEMO_BASE / "models")
    @classmethod
    def get_output_path(cls, key):
        paths = cls.get("output_paths", cls.DEFAULT_CONFIG["output_paths"])
        base_path = paths.get(key, str(cls.BIEMO_BASE / f"output_{key}"))
        Path(base_path).mkdir(parents=True, exist_ok=True)
        return base_path
    @classmethod
    def get_tools_dir(cls):
        p = cls.BIEMO_BASE / "tools"; p.mkdir(parents=True, exist_ok=True); return str(p)
    @classmethod
    def get_tool_path(cls, name: str):
        p = Path(cls.get_tools_dir()) / (name + (".exe" if os.name == "nt" else ""))
        return str(p) if p.exists() else name

ConfigManager.load()
os.environ["U2NET_HOME"] = ConfigManager.get_model_dir()
os.environ["REMBG_HOME"] = ConfigManager.get_model_dir()

# ==================== 全局日志系统 ====================
class LogManager(QObject):
    log_signal = pyqtSignal(str, str)
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    def __init__(self): super().__init__(); self.logs = []
    def log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.logs.append((formatted, level))
        self.log_signal.emit(formatted, level)
        print(f"[{level.upper()}] {message}")
    def info(self, msg): self.log(msg, "info")
    def warning(self, msg): self.log(msg, "warning")
    def error(self, msg): self.log(msg, "error")
    def success(self, msg): self.log(msg, "success")
    def pipe(self, message: str, level: str = "info"): self.log(message, level)

logger = LogManager.instance()

# ==================== 硬件检测 ====================
class HardwareInfo:
    gpu_available = False; gpu_name = "N/A"; gpu_memory_mb = 0; cpu_threads = os.cpu_count() or 4
    onnx_providers = []; available_memory_mb = 4096
    @classmethod
    def detect(cls):
        try:
            import onnxruntime as ort
            cls.onnx_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in cls.onnx_providers:
                cls.gpu_available = True; cls.gpu_name = "CUDA GPU"
                try: import torch; cls.gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                except: cls.gpu_memory_mb = 4096
                logger.success("✓ GPU 加速已开启 (CUDA)")
            elif 'DmlExecutionProvider' in cls.onnx_providers:
                cls.gpu_available = True; cls.gpu_name = "DirectML GPU"; cls.gpu_memory_mb = 4096
                logger.success("✓ GPU 加速已开启 (DirectML)")
            else: logger.warning("○ 正在使用 CPU 模式")
        except: logger.error("✗ onnxruntime 未安装")
        if HAS_PSUTIL:
            try: cls.available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
            except: pass
        return cls
    @classmethod
    def has_sufficient_resources(cls, model_size_mb: int = 900) -> bool:
        if cls.gpu_available and cls.gpu_memory_mb >= model_size_mb * 2: return True
        if cls.available_memory_mb >= model_size_mb * 3: return True
        return False

# ==================== 模型管理器 ====================
class ModelManager:
    MODELS = {
        "birefnet-general": {"name": "BiRefNet 通用 (SOTA)", "file": "BiRefNet-general-epoch_244.onnx", "size_mb": 900, "quality": 5, "large": True},
        "birefnet-general-lite": {"name": "BiRefNet Lite", "file": "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", "size_mb": 200, "quality": 4, "large": False},
        "isnet-general-use": {"name": "ISNet 通用 ★推荐", "file": "isnet-general-use.onnx", "size_mb": 170, "quality": 4, "large": False},
        "isnet-anime": {"name": "ISNet 动漫", "file": "isnet-anime.onnx", "size_mb": 170, "quality": 4, "large": False},
        "u2net": {"name": "U²-Net 标准", "file": "u2net.onnx", "size_mb": 170, "quality": 3, "large": False},
        "u2netp": {"name": "U²-Net 轻量 ★低配", "file": "u2netp.onnx", "size_mb": 4, "quality": 2, "large": False},
    }
    _sessions = {}; _lock = threading.Lock(); _models_status = {}
    
    @classmethod
    def check_model_exists(cls, model_id: str) -> bool:
        model_dir = Path(ConfigManager.get_model_dir())
        if model_id in cls.MODELS: return (model_dir / cls.MODELS[model_id]["file"]).exists()
        return False
    
    @classmethod
    def scan_models(cls):
        model_dir = Path(ConfigManager.get_model_dir())
        cls._models_status = {}
        for model_id, info in cls.MODELS.items():
            model_file = model_dir / info["file"]
            cls._models_status[model_id] = {"exists": model_file.exists(), "file": info["file"], "size_mb": info["size_mb"]}
        return cls._models_status

    @classmethod
    def should_scale_down(cls, model_id: str) -> bool:
        info = cls.MODELS.get(model_id, {})
        if not info.get("large", False): return False
        return not HardwareInfo.has_sufficient_resources(info.get("size_mb", 200))
    
    @classmethod
    def load_model(cls, model_id: str):
        global USE_REMBG, rembg_new_session
        if not USE_REMBG: return None
        with cls._lock:
            if model_id in cls._sessions: return cls._sessions[model_id]
        
        try:
            logger.info(f"加载模型: {model_id}...")
            gc.collect()
            session = rembg_new_session(model_id)
            with cls._lock: cls._sessions[model_id] = session
            return session
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            if model_id != "u2netp":
                logger.warning("尝试回退到 u2netp...")
                return cls.load_model("u2netp")
            return None
    
    @classmethod
    def clear_cache(cls):
        with cls._lock: cls._sessions.clear()
        gc.collect()
        logger.info("模型缓存已清除")

# ==================== rembg 导入 ====================
USE_REMBG = False
rembg_remove = None
rembg_new_session = None
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    USE_REMBG = True
    logger.success("✓ rembg 模块加载成功")
except Exception as e:
    logger.error(f"✗ rembg 加载失败: {e}")

HardwareInfo.detect()
ModelManager.scan_models()

# ==================== 激活验证模块 ====================
MAGIC_VALUE = "788990"
class LicenseManager:
    @staticmethod
    def get_license_file(): return ConfigManager.get_license_file()
    @staticmethod
    def get_machine_code():
        try:
            import uuid
            return str(uuid.getnode())
        except: return "ERROR-ID"
    @staticmethod
    def verify_key(machine_code, input_key):
        try:
            clean_mac = machine_code.replace("-", "").replace(" ", "")
            today_str = date.today().strftime("%Y%m%d")
            input_str = f"{clean_mac}{today_str}{MAGIC_VALUE}"
            sha = hashlib.sha256(input_str.encode()).hexdigest().upper()
            correct = "-".join([sha[i:i+5] for i in range(0, 25, 5)])
            return input_key.strip().upper() == correct
        except: return False
    @staticmethod
    def check_license_file():
        license_file = LicenseManager.get_license_file()
        if not os.path.exists(license_file): return False
        try:
            with open(license_file, "r") as f: saved = f.read().strip()
            curr = hashlib.md5(LicenseManager.get_machine_code().encode()).hexdigest()
            return saved == curr
        except: return False
    @staticmethod
    def save_license():
        license_file = LicenseManager.get_license_file()
        os.makedirs(os.path.dirname(license_file), exist_ok=True)
        with open(license_file, "w") as f:
            f.write(hashlib.md5(LicenseManager.get_machine_code().encode()).hexdigest())

# ==================== 图像处理算法 (增强版) ====================
def play_completion_sound():
    if HAS_WINSOUND and ConfigManager.get("enable_sound", True):
        try: winsound.MessageBeep(winsound.MB_OK)
        except: pass

def smart_resize_for_model(pil_img: Image.Image, model_id: str) -> tuple:
    original_size = pil_img.size
    if not ModelManager.should_scale_down(model_id): return pil_img, original_size, False
    w, h = original_size
    scale = 1024 / max(w, h)
    if scale >= 1: return pil_img, original_size, False
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS), original_size, True

def remove_bg_with_session_smart(pil_img: Image.Image, session, model_id: str) -> Image.Image:
    if not USE_REMBG or not rembg_remove or not session: return pil_img.convert("RGBA")
    try:
        img, orig_size, resized = smart_resize_for_model(pil_img, model_id)
        res = rembg_remove(img, session=session)
        if resized: res = res.resize(orig_size, Image.Resampling.LANCZOS)
        return res
    except Exception as e:
        logger.error(f"BG remove error: {e}")
        return pil_img.convert("RGBA")

def cleanup_edge_pixels(pil_img, feather: int = 1, blur: int = 1, gamma: float = 1.2):
    """边缘清理 - 增强版 v2 (形态学开运算)"""
    if not HAS_CV2 or not HAS_NUMPY: return pil_img
    if pil_img.mode != 'RGBA': pil_img = pil_img.convert('RGBA')
    
    img_array = np.array(pil_img)
    b, g, r, a = cv2.split(img_array)
    
    # 1. 预阈值：清除低透明度幽灵像素
    _, hard_alpha = cv2.threshold(a, 20, 255, cv2.THRESH_TOZERO)
    
    # 2. 形态学开运算：平滑边缘，断开细小粘连
    if feather > 0:
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        hard_alpha = cv2.morphologyEx(hard_alpha, cv2.MORPH_OPEN, kernel, iterations=1)
        if feather > 1:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            hard_alpha = cv2.erode(hard_alpha, erode_kernel, iterations=feather - 1)
    
    # 3. 高斯模糊与Gamma校正
    if blur > 0:
        alpha_f = hard_alpha.astype(np.float32) / 255.0
        k = blur * 2 + 1
        alpha_f = cv2.GaussianBlur(alpha_f, (k, k), 0)
        if gamma != 1.0: alpha_f = np.power(alpha_f, gamma)
        hard_alpha = np.clip(alpha_f * 255, 0, 255).astype(np.uint8)

    img_array = cv2.merge((b, g, r, hard_alpha))
    return Image.fromarray(img_array, mode='RGBA')

def remove_isolated_colors(pil_img, min_area: int, remove_internal: bool = True, internal_max_area: int = 100):
    """移除孤立色块 - 增强版 v2 (连通域分析)"""
    if not HAS_CV2 or not HAS_NUMPY: return pil_img
    if min_area <= 0 and not remove_internal: return pil_img
    if pil_img.mode != 'RGBA': pil_img = pil_img.convert('RGBA')
        
    img_array = np.array(pil_img)
    alpha = img_array[:, :, 3].copy()
    _, binary = cv2.threshold(alpha, 20, 255, cv2.THRESH_BINARY)
    has_change = False
    
    # 1. 连通域分析去除孤立块
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA] # 跳过背景0
            if len(areas) > 0:
                max_idx = np.argmax(areas) + 1
                new_alpha = np.zeros_like(alpha)
                for i in range(1, num_labels):
                    # 保留主体(最大块) 或 大于阈值的块
                    if i == max_idx or stats[i, cv2.CC_STAT_AREA] >= min_area:
                        mask = (labels == i).astype(np.uint8) * 255
                        new_alpha = cv2.bitwise_or(new_alpha, cv2.bitwise_and(alpha, alpha, mask=mask))
                    else: has_change = True
                alpha = new_alpha
                _, binary = cv2.threshold(alpha, 20, 255, cv2.THRESH_BINARY)
    
    # 2. 闭运算填充内部孔洞
    if remove_internal and internal_max_area > 0:
        ks = max(3, int(math.sqrt(internal_max_area)))
        if ks % 2 == 0: ks += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        holes = cv2.bitwise_and(closed, cv2.bitwise_not(binary))
        if cv2.countNonZero(holes) > 0:
            alpha = cv2.add(alpha, holes)
            has_change = True
            
    if not has_change: return pil_img
    img_array[:, :, 3] = alpha
    return Image.fromarray(img_array, mode='RGBA')

def fill_alpha_with_bg(pil_img, bg_type: str, bg_color: str = "#FFFFFF", bg_image_path: str = None):
    if pil_img.mode != 'RGBA': pil_img = pil_img.convert('RGBA')
    if bg_type == "none": return pil_img
    
    if bg_type == "color":
        c = bg_color.strip().lstrip('#')
        rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) if len(c) == 6 else (255,255,255)
        base = Image.new('RGB', pil_img.size, rgb)
    elif bg_type == "image" and bg_image_path and Path(bg_image_path).exists():
        try: base = Image.open(bg_image_path).convert('RGB').resize(pil_img.size, Image.Resampling.LANCZOS)
        except: base = Image.new('RGB', pil_img.size, (255, 255, 255))
    else: base = Image.new('RGB', pil_img.size, (255, 255, 255))
    base.paste(pil_img, mask=pil_img.split()[-1])
    return base

def process_single_frame(frame_data: tuple, session, params: dict, model_id: str = "u2net") -> tuple:
    idx, frame_rgb = frame_data
    try:
        pil = Image.fromarray(frame_rgb)
        if params.get("remove_bg"):
            pil = remove_bg_with_session_smart(pil, session, model_id)
            if params.get("cleanup_edge"):
                pil = cleanup_edge_pixels(pil, params.get("edge_feather", 1), params.get("edge_blur", 1), params.get("edge_gamma", 1.2))
            if params.get("remove_isolated"):
                pil = remove_isolated_colors(pil, params.get("isolated_area", 50), params.get("remove_internal", True), params.get("internal_max_area", 100))
            if params.get("bg_type", "none") != "none":
                pil = fill_alpha_with_bg(pil, params.get("bg_type"), params.get("bg_color"), params.get("bg_image"))
        else: pil = pil.convert("RGBA")
        return (idx, pil, None)
    except Exception as e: return (idx, None, str(e))

# ==================== UI组件 ====================
class ColorPickerWidget(QWidget):
    color_changed = pyqtSignal(str)
    def __init__(self, default_color: str = "#FFFFFF", parent=None):
        super().__init__(parent); self.current_color = default_color; self._setup_ui()
    def _setup_ui(self):
        layout = QHBoxLayout(); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(4)
        self.preview = QLabel(); self.preview.setFixedSize(24, 24)
        self.edit = QLineEdit(self.current_color); self.edit.setFixedWidth(80)
        self.edit.textChanged.connect(self._on_text)
        btn = QPushButton("选色"); btn.setFixedWidth(45); btn.clicked.connect(self._pick)
        layout.addWidget(self.preview); layout.addWidget(self.edit); layout.addWidget(btn)
        self.setLayout(layout); self._update_preview()
    def _on_text(self, t):
        if t.startswith('#') and len(t) in (4,7):
            self.current_color = t; self._update_preview(); self.color_changed.emit(t)
    def _pick(self):
        c = QColorDialog.getColor(QColor(self.current_color), self, "选择颜色")
        if c.isValid():
            self.current_color = c.name().upper(); self.edit.setText(self.current_color); self._update_preview()
            self.color_changed.emit(self.current_color)
    def _update_preview(self): self.preview.setStyleSheet(f"background:{self.current_color};border:1px solid #666;")
    def get_color(self): return self.current_color

class FileDropLineEdit(QLineEdit):
    def __init__(self, parent=None, placeholder=""):
        super().__init__(parent); self.setAcceptDrops(True); self.setPlaceholderText(placeholder)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()
        else: e.ignore()
    def dropEvent(self, e):
        try: self.setText(e.mimeData().urls()[0].toLocalFile())
        except: pass

class LogWidget(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent); self.setReadOnly(True); logger.log_signal.connect(self.add_log)
        self.setStyleSheet("background:#1e1e1e;color:#d4d4d4;font-family:Consolas;")
    def add_log(self, msg, level):
        c = {"info":"#d4d4d4","warning":"#dcdcaa","error":"#f14c4c","success":"#4ec9b0"}.get(level,"#d4d4d4")
        self.appendHtml(f'<span style="color:{c};">{msg}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class ModelSelector(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent); self.refresh_models()
    def refresh_models(self):
        self.clear(); ModelManager.scan_models()
        for mid, info in ModelManager.MODELS.items():
            loaded = mid in ModelManager._sessions
            exists = ModelManager.check_model_exists(mid)
            icon = "★" if loaded else ("✓" if exists else "○")
            self.addItem(f"{icon} {info['name']}", mid)
        for mid, stat in ModelManager._models_status.items():
            if mid not in ModelManager.MODELS: self.addItem(f"✓ [自定义] {stat['file']}", mid)
        self.setCurrentIndex(0)
    def get_current_model(self): return self.currentData()

# ==================== 弹窗类 ====================
class ActivationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("软件激活"); self.setFixedSize(500, 400)
        self.activated = False; self.trial_mode = False; self.mac = LicenseManager.get_machine_code()
        
        layout = QVBoxLayout(); layout.setSpacing(15); layout.setContentsMargins(30,30,30,30)
        layout.addWidget(QLabel("别快视频精灵图 v7.6"), alignment=Qt.AlignCenter)
        
        self.mac_edit = QLineEdit(self.mac); self.mac_edit.setReadOnly(True)
        copy_btn = QPushButton("复制机器码"); copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.mac))
        h = QHBoxLayout(); h.addWidget(self.mac_edit); h.addWidget(copy_btn)
        layout.addLayout(h)
        
        self.key_edit = QLineEdit(); self.key_edit.setPlaceholderText("在此输入激活密钥")
        layout.addWidget(self.key_edit)
        
        btn_layout = QHBoxLayout()
        trial_btn = QPushButton("试用 (15分钟)"); trial_btn.clicked.connect(self.on_trial)
        act_btn = QPushButton("激活"); act_btn.clicked.connect(self.on_activate)
        btn_layout.addWidget(trial_btn); btn_layout.addWidget(act_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def on_trial(self):
        if QMessageBox.question(self, "试用", "试用模式限时15分钟，确定？", QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self.trial_mode = True; self.accept()
    def on_activate(self):
        if LicenseManager.verify_key(self.mac, self.key_edit.text()):
            LicenseManager.save_license(); self.activated = True
            QMessageBox.information(self, "成功", "激活成功！"); self.accept()
        else: QMessageBox.critical(self, "错误", "激活码无效")

# ==================== Worker 类 ====================
class BaseWorker(QThread):
    progress = pyqtSignal(int, str); finished = pyqtSignal(dict); error = pyqtSignal(str)
    def __init__(self): super().__init__(); self._stop = False
    def stop(self): self._stop = True

class SpriteWorker(BaseWorker):
    def __init__(self, src, out, params): super().__init__(); self.src=Path(src); self.out=Path(out); self.params=params
    def run(self):
        try:
            model = self.params.get("model_name")
            sess = ModelManager.load_model(model) if self.params.get("remove_bg") else None
            frames = []
            
            if self.params["source_type"] == "video":
                cap = cv2.VideoCapture(str(self.src))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = self.params["frame_step"]
                for i in range(0, total, step):
                    if self._stop: break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, f = cap.read()
                    if ret: frames.append((len(frames), cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    self.progress.emit(int(i/total*30), "读取视频...")
                cap.release()
            
            # 多线程处理
            if frames:
                processed = []
                with ThreadPoolExecutor(max_workers=self.params["num_threads"]) as exc:
                    futures = {exc.submit(process_single_frame, f, sess, self.params, model): f[0] for f in frames}
                    cnt = 0
                    for fut in as_completed(futures):
                        if self._stop: break
                        idx, pil, err = fut.result()
                        if pil: processed.append((idx, pil))
                        cnt += 1; self.progress.emit(30 + int(cnt/len(frames)*50), "处理帧...")
                
                processed.sort(key=lambda x: x[0])
                final_frames = [p[1] for p in processed]
                
                # 合成精灵图
                if final_frames:
                    tw, th = self.params["thumb_w"], self.params["thumb_h"]
                    if self.params["scale_mode"] == "percent":
                        sc = self.params["scale_percent"]/100; tw, th = int(final_frames[0].width*sc), int(final_frames[0].height*sc)
                    
                    cols = self.params["columns"]; rows = math.ceil(len(final_frames)/cols)
                    sheet = Image.new("RGBA", (cols*tw, rows*th))
                    for i, img in enumerate(final_frames):
                        sheet.paste(img.resize((tw,th)), ((i%cols)*tw, (i//cols)*th))
                    
                    out_path = self.out / f"{self.src.stem}_sprite.png"
                    sheet.save(out_path)
                    self.finished.emit({"file": str(out_path)})
                    return

            self.error.emit("无有效帧")
        except Exception as e: self.error.emit(str(e))

# ==================== 主窗口 ====================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.activated = False; self.trial_seconds = 900
        
        # 激活检查
        if LicenseManager.check_license_file(): self.activated = True
        else:
            dlg = ActivationDialog()
            if dlg.exec_() == QDialog.Accepted:
                if dlg.activated: self.activated = True
                elif dlg.trial_mode: 
                    self.trial_timer = QTimer(self); self.trial_timer.timeout.connect(self._trial_tick)
                    self.trial_timer.start(1000)
            else: sys.exit()
            
        self.setWindowTitle(f"别快视频精灵图 v7.6 - {'已激活' if self.activated else '试用模式'}")
        self.setFixedWidth(650); self.setup_ui()
    
    def _trial_tick(self):
        self.trial_seconds -= 1
        self.setWindowTitle(f"别快视频精灵图 v7.6 - 试用剩余 {self.trial_seconds//60}:{self.trial_seconds%60:02d}")
        if self.trial_seconds <= 0:
            QMessageBox.warning(self, "结束", "试用时间已到"); sys.exit()

    def setup_ui(self):
        layout = QVBoxLayout(); self.setLayout(layout)
        
        # 顶部输入
        grp_src = QGroupBox("输入源")
        l_src = QHBoxLayout(); self.path_edit = FileDropLineEdit(); btn_src = QPushButton("选择")
        btn_src.clicked.connect(lambda: self.path_edit.setText(QFileDialog.getOpenFileName(self, "选视频", "", "Video (*.mp4 *.avi)")[0]))
        l_src.addWidget(self.path_edit); l_src.addWidget(btn_src); grp_src.setLayout(l_src)
        layout.addWidget(grp_src)
        
        # 模型与参数
        grp_set = QGroupBox("设置")
        grid = QGridLayout()
        self.model_sel = ModelSelector()
        grid.addWidget(QLabel("模型:"), 0, 0); grid.addWidget(self.model_sel, 0, 1)
        
        self.chk_rembg = QCheckBox("移除背景"); self.chk_rembg.setChecked(True)
        self.chk_clean = QCheckBox("边缘清理(OpenCV)"); self.chk_clean.setChecked(True)
        self.chk_iso = QCheckBox("去孤立块(连通域)"); self.chk_iso.setChecked(True)
        grid.addWidget(self.chk_rembg, 1, 0); grid.addWidget(self.chk_clean, 1, 1); grid.addWidget(self.chk_iso, 1, 2)
        
        grid.addWidget(QLabel("列数:"), 2, 0); self.spin_col = QSpinBox(); self.spin_col.setValue(10); grid.addWidget(self.spin_col, 2, 1)
        grid.addWidget(QLabel("缩放%:"), 2, 2); self.spin_scale = QSpinBox(); self.spin_scale.setValue(50); grid.addWidget(self.spin_scale, 2, 3)
        
        grp_set.setLayout(grid); layout.addWidget(grp_set)
        
        # 操作
        btn_run = QPushButton("开始生成"); btn_run.clicked.connect(self.start_task)
        btn_run.setStyleSheet("background:#27ae60;color:white;font-weight:bold;padding:8px;")
        layout.addWidget(btn_run)
        
        self.prog = QProgressBar(); layout.addWidget(self.prog)
        self.log_view = LogWidget(); layout.addWidget(self.log_view)

    def start_task(self):
        src = self.path_edit.text()
        if not os.path.exists(src): return QMessageBox.warning(self, "错", "文件不存在")
        
        params = {
            "source_type": "video",
            "model_name": self.model_sel.get_current_model(),
            "num_threads": 4,
            "frame_step": 1,
            "columns": self.spin_col.value(),
            "scale_mode": "percent",
            "scale_percent": self.spin_scale.value(),
            "thumb_w": 256, "thumb_h": 256,
            "remove_bg": self.chk_rembg.isChecked(),
            "cleanup_edge": self.chk_clean.isChecked(), "edge_feather": 1, "edge_blur": 1, "edge_gamma": 1.2,
            "remove_isolated": self.chk_iso.isChecked(), "isolated_area": 100, "remove_internal": True, "internal_max_area": 100,
            "bg_type": "none"
        }
        
        self.worker = SpriteWorker(src, ConfigManager.get_output_path("sprite"), params)
        self.worker.progress.connect(lambda v, m: [self.prog.setValue(v), logger.info(m)])
        self.worker.error.connect(lambda e: [logger.error(e), QMessageBox.critical(self, "错", e)])
        self.worker.finished.connect(lambda d: [logger.success("完成"), play_completion_sound(), os.startfile(Path(d["file"]).parent)])
        self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 确保主窗口实例被保留
    w = MainWindow()
    w.show()
    
    sys.exit(app.exec_())
