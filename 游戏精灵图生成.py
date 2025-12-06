# video_sprites_gui_enhanced_v7_6.py
# Enhanced version v7.6 - 极速专业版 (完整修复版)
# 
# 【v7.6 修复与改进】
# - 修复试用退出机制：使用总秒数倒计时，确保准确退出
# - 添加颜色选择器：所有背景色输入都支持可视化选择
# - 优化用户体验：颜色预览、预设颜色、友好的交互
# - 安全退出流程：停止任务 -> 提示用户 -> 延迟退出
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
from pathlib import Path
from datetime import datetime, date
from io import BytesIO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 设置模型路径环境变量 ====================
BIEMO_DIR = Path.cwd() / "biemo"
BIEMO_DIR.mkdir(parents=True, exist_ok=True)

# 设置 rembg/u2net 模型下载路径
os.environ["U2NET_HOME"] = str(BIEMO_DIR / "models")
os.environ["REMBG_HOME"] = str(BIEMO_DIR / "models")

# 确保模型目录存在
(BIEMO_DIR / "models").mkdir(parents=True, exist_ok=True)

# ==================== 依赖检测系统 ====================
class DependencyChecker:
    """检测所有必要的依赖库"""
    
    REQUIRED_PACKAGES = [
        ("cv2", "opencv-python", "图像处理核心库"),
        ("PIL", "Pillow", "图像格式支持"),
        ("numpy", "numpy", "数值计算"),
        ("imageio", "imageio", "视频/GIF读写"),
    ]
    
    OPTIONAL_PACKAGES = [
        ("rembg", "rembg[gpu]", "AI背景移除 (核心功能)"),
        ("onnxruntime", "onnxruntime", "AI推理引擎 (CPU)"),
    ]
    
    results = {}
    missing_required = []
    missing_optional = []
    install_commands = []
    
    @classmethod
    def check_all(cls):
        cls.results = {}
        cls.missing_required = []
        cls.missing_optional = []
        cls.install_commands = []
        
        for module_name, pip_name, desc in cls.REQUIRED_PACKAGES:
            try:
                __import__(module_name)
                cls.results[module_name] = ("ok", desc)
            except ImportError:
                cls.results[module_name] = ("missing", desc)
                cls.missing_required.append((pip_name, desc))
                cls.install_commands.append(f"pip install {pip_name}")
        
        for module_name, pip_name, desc in cls.OPTIONAL_PACKAGES:
            try:
                __import__(module_name)
                cls.results[module_name] = ("ok", desc)
            except ImportError:
                cls.results[module_name] = ("missing", desc)
                cls.missing_optional.append((pip_name, desc))
        
        return cls
    
    @classmethod
    def get_install_command(cls):
        if cls.missing_required:
            pkgs = [p[0] for p in cls.missing_required]
            return f"pip install {' '.join(pkgs)}"
        return None
    
    @classmethod
    def get_full_install_command(cls):
        return "pip install opencv-python Pillow numpy imageio imageio-ffmpeg rembg[gpu] onnxruntime-gpu"
    
    @classmethod
    def has_critical_missing(cls):
        return len(cls.missing_required) > 0

DependencyChecker.check_all()

# 尝试导入可能缺失的库
try:
    import winsound
    HAS_WINSOUND = True
except:
    HAS_WINSOUND = False

try:
    from PIL import Image
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
    QListWidgetItem, QColorDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor, QIcon, QPixmap

# ==================== 配置管理器 ====================
class ConfigManager:
    """统一配置管理 - 所有路径都在 biemo 文件夹下"""
    
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
    }
    
    _config = None
    
    @classmethod
    def init_directories(cls):
        """初始化所有目录"""
        cls.BIEMO_BASE.mkdir(parents=True, exist_ok=True)
        (cls.BIEMO_BASE / "models").mkdir(parents=True, exist_ok=True)
        (cls.BIEMO_BASE / "tools").mkdir(parents=True, exist_ok=True)
        
        for key in cls.DEFAULT_CONFIG["output_paths"]:
            path = Path(cls.DEFAULT_CONFIG["output_paths"][key])
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_biemo_dir(cls):
        return cls.BIEMO_BASE
    
    @classmethod
    def get_license_file(cls):
        return str(cls.LICENSE_FILE)
    
    @classmethod
    def load(cls):
        if cls._config is not None:
            return cls._config
        
        cls.init_directories()
        cls._config = cls.DEFAULT_CONFIG.copy()
        
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    for key, value in saved.items():
                        if key == "output_paths":
                            cls._config["output_paths"].update(value)
                        else:
                            cls._config[key] = value
        except Exception as e:
            print(f"配置加载失败: {e}")
        
        return cls._config
    
    @classmethod
    def save(cls):
        try:
            cls.BIEMO_BASE.mkdir(parents=True, exist_ok=True)
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cls._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"配置保存失败: {e}")
    
    @classmethod
    def get(cls, key, default=None):
        config = cls.load()
        return config.get(key, default)
    
    @classmethod
    def set(cls, key, value):
        config = cls.load()
        config[key] = value
        cls.save()
    
    @classmethod
    def get_model_dir(cls):
        return str(cls.BIEMO_BASE / "models")
    
    @classmethod
    def get_output_path(cls, key):
        paths = cls.get("output_paths", cls.DEFAULT_CONFIG["output_paths"])
        base_path = paths.get(key, str(cls.BIEMO_BASE / f"output_{key}"))
        Path(base_path).mkdir(parents=True, exist_ok=True)
        return base_path

ConfigManager.load()
os.environ["U2NET_HOME"] = ConfigManager.get_model_dir()
os.environ["REMBG_HOME"] = ConfigManager.get_model_dir()

# ==================== 全局日志系统 ====================
class LogManager(QObject):
    log_signal = pyqtSignal(str, str)
    
    _instance = None
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.logs = []
    
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

logger = LogManager.instance()

# ==================== 硬件检测 ====================
class HardwareInfo:
    gpu_available = False
    gpu_name = "N/A"
    gpu_memory_mb = 0
    cpu_threads = os.cpu_count() or 4
    onnx_providers = []
    available_memory_mb = 4096
    
    @classmethod
    def detect(cls):
        try:
            import onnxruntime as ort
            cls.onnx_providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in cls.onnx_providers:
                cls.gpu_available = True
                cls.gpu_name = "CUDA GPU"
                try:
                    import torch
                    if torch.cuda.is_available():
                        cls.gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                except:
                    cls.gpu_memory_mb = 4096
                logger.success("✓ GPU 加速已开启 (CUDA)")
            elif 'DmlExecutionProvider' in cls.onnx_providers:
                cls.gpu_available = True
                cls.gpu_name = "DirectML GPU"
                cls.gpu_memory_mb = 4096
                logger.success("✓ GPU 加速已开启 (DirectML)")
            else:
                logger.warning("○ 正在使用 CPU 模式")
                
        except ImportError:
            logger.error("✗ onnxruntime 未安装")
        except Exception as e:
            logger.error(f"硬件检测失败: {e}")
        
        if HAS_PSUTIL:
            try:
                cls.available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
            except:
                pass
        
        logger.info(f"CPU 线程数: {cls.cpu_threads}")
        logger.info(f"可用内存: {cls.available_memory_mb} MB")
        if cls.gpu_available:
            logger.info(f"GPU 显存: {cls.gpu_memory_mb} MB")
        return cls
    
    @classmethod
    def has_sufficient_resources(cls, model_size_mb: int = 900) -> bool:
        """检查是否有足够的资源处理大模型"""
        if cls.gpu_available and cls.gpu_memory_mb >= model_size_mb * 2:
            return True
        if cls.available_memory_mb >= model_size_mb * 3:
            return True
        return False

# ==================== 模型管理器 ====================
class ModelManager:
    """模型管理：统一文件名，支持用户导入模型"""
    
    MODELS = {
        "birefnet-general": {
            "name": "BiRefNet 通用 (SOTA)",
            "desc": "最高质量，需要较多资源",
            "file": "BiRefNet-general-epoch_244.onnx",
            "size_mb": 900,
            "quality": 5,
            "large": True
        },
        "birefnet-general-lite": {
            "name": "BiRefNet Lite",
            "desc": "快速高质量",
            "file": "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
            "size_mb": 200,
            "quality": 4,
            "large": False
        },
        "birefnet-portrait": {
            "name": "BiRefNet 人像",
            "desc": "人像优化，需要较多资源",
            "file": "BiRefNet-portrait-epoch_150.onnx",
            "size_mb": 900,
            "quality": 5,
            "large": True
        },
        "isnet-general-use": {
            "name": "ISNet 通用 ★推荐",
            "desc": "推荐，平衡质量和速度",
            "file": "isnet-general-use.onnx",
            "size_mb": 170,
            "quality": 4,
            "large": False
        },
        "isnet-anime": {
            "name": "ISNet 动漫",
            "desc": "二次元/插画优化",
            "file": "isnet-anime.onnx",
            "size_mb": 170,
            "quality": 4,
            "large": False
        },
        "u2net": {
            "name": "U²-Net 标准",
            "desc": "经典稳定，兼容性好",
            "file": "u2net.onnx",
            "size_mb": 170,
            "quality": 3,
            "large": False
        },
        "u2netp": {
            "name": "U²-Net 轻量 ★低配",
            "desc": "最快速度，低配首选",
            "file": "u2netp.onnx",
            "size_mb": 4,
            "quality": 2,
            "large": False
        },
        "u2net_human_seg": {
            "name": "U²-Net 人像",
            "desc": "人体分割优化",
            "file": "u2net_human_seg.onnx",
            "size_mb": 170,
            "quality": 3,
            "large": False
        },
        "u2net_cloth_seg": {
            "name": "U²-Net 服装",
            "desc": "衣物分割",
            "file": "u2net_cloth_seg.onnx",
            "size_mb": 170,
            "quality": 3,
            "large": False
        },
        "silueta": {
            "name": "Silueta",
            "desc": "轮廓优化",
            "file": "silueta.onnx",
            "size_mb": 40,
            "quality": 3,
            "large": False
        },
    }
    
    _sessions = {}
    _lock = threading.Lock()
    _models_status = {}
    
    @classmethod
    def get_model_dir(cls) -> Path:
        model_dir = Path(ConfigManager.get_model_dir())
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    @classmethod
    def get_models_config_file(cls) -> Path:
        return ConfigManager.MODELS_CONFIG_FILE
    
    @classmethod
    def load_models_config(cls):
        """从配置文件加载模型状态"""
        config_file = cls.get_models_config_file()
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    cls._models_status = json.load(f)
            except:
                cls._models_status = {}
        return cls._models_status
    
    @classmethod
    def save_models_config(cls):
        """保存模型状态到配置文件"""
        config_file = cls.get_models_config_file()
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(cls._models_status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存模型配置失败: {e}")
    
    @classmethod
    def scan_models(cls) -> dict:
        """扫描模型目录，更新模型状态"""
        model_dir = cls.get_model_dir()
        cls._models_status = {}
        found_count = 0
        
        for model_id, info in cls.MODELS.items():
            model_file = model_dir / info["file"]
            exists = model_file.exists()
            
            if exists:
                found_count += 1
                file_size = model_file.stat().st_size // (1024 * 1024)
                cls._models_status[model_id] = {
                    "exists": True,
                    "file": info["file"],
                    "path": str(model_file),
                    "size_mb": file_size,
                    "scan_time": datetime.now().isoformat()
                }
            else:
                cls._models_status[model_id] = {
                    "exists": False,
                    "file": info["file"],
                    "path": str(model_file),
                    "size_mb": 0,
                    "scan_time": datetime.now().isoformat()
                }
        
        # 扫描用户自定义模型
        for onnx_file in model_dir.glob("*.onnx"):
            is_known = False
            for model_id, info in cls.MODELS.items():
                if onnx_file.name == info["file"]:
                    is_known = True
                    break
            
            if not is_known:
                custom_id = onnx_file.stem
                file_size = onnx_file.stat().st_size // (1024 * 1024)
                cls._models_status[f"custom_{custom_id}"] = {
                    "exists": True,
                    "file": onnx_file.name,
                    "path": str(onnx_file),
                    "size_mb": file_size,
                    "custom": True,
                    "scan_time": datetime.now().isoformat()
                }
                found_count += 1
        
        cls.save_models_config()
        logger.info(f"扫描完成: {found_count} 个模型")
        return cls._models_status
    
    @classmethod
    def check_model_exists(cls, model_id: str) -> bool:
        """检查模型文件是否存在"""
        model_dir = cls.get_model_dir()
        
        if model_id in cls.MODELS:
            model_file = model_dir / cls.MODELS[model_id]["file"]
            return model_file.exists()
        
        if model_id.startswith("custom_"):
            status = cls._models_status.get(model_id, {})
            if status.get("path"):
                return Path(status["path"]).exists()
        
        return False
    
    @classmethod
    def get_model_status(cls, model_id: str) -> dict:
        """获取模型状态"""
        if model_id in cls._models_status:
            cached = cls._models_status[model_id]
            cached["exists"] = cls.check_model_exists(model_id)
            return cached
        
        if not cls._models_status:
            cls.scan_models()
        
        return cls._models_status.get(model_id, {
            "exists": False,
            "file": cls.MODELS.get(model_id, {}).get("file", f"{model_id}.onnx")
        })
    
    @classmethod
    def is_large_model(cls, model_id: str) -> bool:
        """判断是否是大模型"""
        info = cls.MODELS.get(model_id, {})
        return info.get("large", False)
    
    @classmethod
    def should_scale_down(cls, model_id: str) -> bool:
        """判断是否需要缩小处理"""
        if not cls.is_large_model(model_id):
            return False
        
        info = cls.MODELS.get(model_id, {})
        model_size = info.get("size_mb", 200)
        
        if HardwareInfo.has_sufficient_resources(model_size):
            return False
        
        return True
    
    @classmethod
    def load_model(cls, model_id: str):
        """加载模型"""
        global USE_REMBG, rembg_new_session
        
        if not USE_REMBG:
            logger.error("rembg 未安装，无法加载模型")
            return None
        
        with cls._lock:
            if model_id in cls._sessions:
                logger.info(f"模型 {model_id} 已在缓存中")
                return cls._sessions[model_id]
        
        exists = cls.check_model_exists(model_id)
        if not exists:
            logger.warning(f"模型文件不存在，将在首次使用时自动下载")
        
        try:
            logger.info(f"加载模型: {model_id}...")
            start = time.time()
            
            gc.collect()
            
            session = rembg_new_session(model_id)
            
            elapsed = time.time() - start
            logger.success(f"模型加载成功 ({elapsed:.1f}s)")
            
            with cls._lock:
                cls._sessions[model_id] = session
            
            cls._models_status[model_id] = cls._models_status.get(model_id, {})
            cls._models_status[model_id]["exists"] = True
            cls._models_status[model_id]["loaded"] = True
            cls.save_models_config()
            
            return session
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            traceback.print_exc()
            
            if model_id != "u2netp":
                logger.warning("尝试回退到 u2netp 轻量模型...")
                return cls.load_model("u2netp")
            
            return None
    
    @classmethod
    def get_session(cls, model_id: str):
        with cls._lock:
            return cls._sessions.get(model_id)
    
    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._sessions.clear()
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
except ImportError:
    logger.error("✗ rembg 模块未安装")
except Exception as e:
    logger.error(f"rembg 加载失败: {e}")

# 执行硬件检测和模型扫描
HardwareInfo.detect()
ModelManager.scan_models()

# ==================== 激活验证模块 ====================
MAGIC_VALUE = "788990"

class LicenseManager:
    @staticmethod
    def get_license_file():
        return ConfigManager.get_license_file()
    
    @staticmethod
    def get_machine_code():
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            def get_cmd(c):
                try: 
                    return subprocess.check_output(c, startupinfo=si).decode().split('\n')[1].strip()
                except: 
                    return ""
            raw = f"{get_cmd('wmic cpu get processorid')}{get_cmd('wmic baseboard get serialnumber')}{get_cmd('wmic diskdrive where index=0 get serialnumber')}".replace(" ", "")
            if len(raw) < 5: 
                import uuid
                raw = str(uuid.getnode())
            hashed = hashlib.md5(raw.encode()).hexdigest().upper()
            return f"{hashed[0:4]}-{hashed[4:8]}-{hashed[8:12]}-{hashed[12:16]}"
        except: 
            return "ERROR-ID"

    @staticmethod
    def verify_key(machine_code, input_key):
        try:
            clean_mac = machine_code.replace("-", "").replace(" ", "")
            today_str = date.today().strftime("%Y%m%d")
            input_str = f"{clean_mac}{today_str}{MAGIC_VALUE}"
            sha = hashlib.sha256(input_str.encode()).hexdigest().upper()
            correct = "-".join([sha[i:i+5] for i in range(0, 25, 5)])
            return input_key.strip().upper() == correct
        except: 
            return False

    @staticmethod
    def check_license_file():
        license_file = LicenseManager.get_license_file()
        if not os.path.exists(license_file): 
            return False
        try:
            with open(license_file, "r") as f: 
                saved = f.read().strip()
            curr = hashlib.md5(LicenseManager.get_machine_code().encode()).hexdigest()
            return saved == curr
        except: 
            return False

    @staticmethod
    def save_license():
        license_file = LicenseManager.get_license_file()
        os.makedirs(os.path.dirname(license_file), exist_ok=True)
        with open(license_file, "w") as f:
            f.write(hashlib.md5(LicenseManager.get_machine_code().encode()).hexdigest())
# ==================== 第二部分：UI组件、颜色选择器、图像处理、Workers ====================

# ==================== 颜色选择器组件 ====================
class ColorPickerWidget(QWidget):
    """带颜色选择器的输入组件"""
    
    # 预设常用颜色
    PRESET_COLORS = [
        ("#FFFFFF", "白色"),
        ("#000000", "黑色"),
        ("#00FF00", "绿幕"),
        ("#0000FF", "蓝幕"),
        ("#FF0000", "红色"),
        ("#FFFF00", "黄色"),
        ("#00FFFF", "青色"),
        ("#FF00FF", "品红"),
        ("#808080", "灰色"),
        ("#F5F5DC", "米色"),
    ]
    
    color_changed = pyqtSignal(str)
    
    def __init__(self, default_color: str = "#FFFFFF", parent=None):
        super().__init__(parent)
        self.current_color = default_color
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # 颜色预览框
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(24, 24)
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {self.current_color};
                border: 1px solid #666;
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.color_preview)
        
        # 颜色代码输入框
        self.color_edit = QLineEdit(self.current_color)
        self.color_edit.setFixedWidth(80)
        self.color_edit.setPlaceholderText("#RRGGBB")
        self.color_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.color_edit)
        
        # 选择颜色按钮
        self.pick_btn = QPushButton("选色")
        self.pick_btn.setFixedWidth(45)
        self.pick_btn.setToolTip("打开颜色选择器")
        self.pick_btn.clicked.connect(self._open_color_dialog)
        layout.addWidget(self.pick_btn)
        
        # 预设颜色下拉框
        self.preset_combo = QComboBox()
        self.preset_combo.setFixedWidth(70)
        self.preset_combo.addItem("预设...")
        for color, name in self.PRESET_COLORS:
            self.preset_combo.addItem(name, color)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        layout.addWidget(self.preset_combo)
        
        self.setLayout(layout)
    
    def _on_text_changed(self, text: str):
        """输入框文本变化时更新预览"""
        text = text.strip()
        if self._is_valid_color(text):
            self.current_color = text
            self._update_preview()
            self.color_changed.emit(text)
    
    def _is_valid_color(self, color: str) -> bool:
        """验证颜色代码格式"""
        if not color.startswith('#'):
            return False
        color = color[1:]
        if len(color) not in (3, 6):
            return False
        try:
            int(color, 16)
            return True
        except ValueError:
            return False
    
    def _update_preview(self):
        """更新颜色预览"""
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {self.current_color};
                border: 1px solid #666;
                border-radius: 3px;
            }}
        """)
    
    def _open_color_dialog(self):
        """打开系统颜色选择器"""
        initial_color = QColor(self.current_color)
        color = QColorDialog.getColor(initial_color, self, "选择背景颜色")
        
        if color.isValid():
            hex_color = color.name().upper()
            self.current_color = hex_color
            self.color_edit.setText(hex_color)
            self._update_preview()
            self.color_changed.emit(hex_color)
    
    def _on_preset_selected(self, index: int):
        """选择预设颜色"""
        if index <= 0:
            return
        
        color = self.preset_combo.itemData(index)
        if color:
            self.current_color = color
            self.color_edit.setText(color)
            self._update_preview()
            self.color_changed.emit(color)
        
        # 重置下拉框
        self.preset_combo.setCurrentIndex(0)
    
    def get_color(self) -> str:
        """获取当前颜色"""
        return self.current_color
    
    def set_color(self, color: str):
        """设置颜色"""
        if self._is_valid_color(color):
            self.current_color = color
            self.color_edit.setText(color)
            self._update_preview()

# ==================== 自定义UI组件 ====================
class FileDropLineEdit(QLineEdit):
    def __init__(self, parent=None, placeholder="可以直接拖入文件..."):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText(placeholder)
    
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        try:
            path = e.mimeData().urls()[0].toLocalFile()
            self.setText(path)
            self.editingFinished.emit()
        except:
            pass

class LogWidget(QPlainTextEdit):
    """系统日志显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(1000)
        self.setFont(QFont("Consolas", 9))
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
        """)
        logger.log_signal.connect(self.append_log)
    
    def append_log(self, message: str, level: str):
        colors = {
            "info": "#d4d4d4",
            "warning": "#dcdcaa",
            "error": "#f14c4c",
            "success": "#4ec9b0"
        }
        color = colors.get(level, "#d4d4d4")
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
        self.appendHtml(f'<span style="color: {color};">{message}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class ModelSelector(QComboBox):
    """模型选择器：带状态指示"""
    
    model_changed = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.refresh_models()
        self.currentIndexChanged.connect(self._on_selection_changed)
    
    def refresh_models(self):
        """刷新模型列表"""
        self.clear()
        
        ModelManager.scan_models()
        
        for model_id, info in ModelManager.MODELS.items():
            exists = ModelManager.check_model_exists(model_id)
            loaded = model_id in ModelManager._sessions
            
            status_icon = "★" if loaded else ("✓" if exists else "○")
            large_mark = "🔴" if info.get("large") else ""
            quality_stars = "★" * info.get("quality", 3)
            display_text = f"{status_icon} {large_mark}{info['name']} [{quality_stars}]"
            
            self.addItem(display_text, model_id)
        
        for model_id, status in ModelManager._models_status.items():
            if status.get("custom"):
                display_text = f"✓ [自定义] {status['file']}"
                self.addItem(display_text, model_id)
        
        default_model = ConfigManager.get("default_model", "isnet-general-use")
        for i in range(self.count()):
            if self.itemData(i) == default_model:
                self.setCurrentIndex(i)
                break
    
    def _on_selection_changed(self, index):
        model_id = self.currentData()
        if model_id:
            exists = ModelManager.check_model_exists(model_id)
            info = ModelManager.MODELS.get(model_id, {})
            
            status = {
                "exists": exists,
                "info": info,
                "large": info.get("large", False)
            }
            self.model_changed.emit(model_id, status)
            
            if exists:
                logger.info(f"已选择模型: {info.get('name', model_id)}")
            else:
                logger.warning(f"模型未下载，首次使用将自动下载")
            
            if info.get("large"):
                if HardwareInfo.has_sufficient_resources(info.get("size_mb", 900)):
                    logger.info("资源充足，将使用原始分辨率处理")
                else:
                    logger.info("大模型将使用缩放处理以节省内存")
    
    def get_current_model(self) -> str:
        return self.currentData() or "isnet-general-use"

# ==================== 激活对话框 ====================
class ActivationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("软件激活验证")
        self.setFixedSize(600, 530)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        self.activated = False
        self.trial_mode = False
        self.machine_code = LicenseManager.get_machine_code()
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("别快视频精灵图 v7.6 极速专业版")
        title.setFont(QFont("Microsoft YaHei UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50;")
        main_layout.addWidget(title)

        subtitle = QLabel("请完成激活以使用完整功能")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 11pt;")
        main_layout.addWidget(subtitle)
        
        info_group = QGroupBox("第一步：获取机器码")
        info_layout = QVBoxLayout()
        code_layout = QHBoxLayout()
        self.mac_edit = QLineEdit()
        self.mac_edit.setText(self.machine_code)
        self.mac_edit.setReadOnly(True)
        self.mac_edit.setAlignment(Qt.AlignCenter)
        self.mac_edit.setFont(QFont("Consolas", 12, QFont.Bold))
        self.mac_edit.setFixedHeight(40)
        
        copy_btn = QPushButton("复制")
        copy_btn.setFixedSize(80, 40)
        copy_btn.clicked.connect(self.copy_machine_code)
        
        code_layout.addWidget(self.mac_edit)
        code_layout.addWidget(copy_btn)
        info_layout.addLayout(code_layout)
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)
        
        input_group = QGroupBox("第二步：输入激活密钥")
        input_layout = QVBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setAlignment(Qt.AlignCenter)
        self.key_edit.setFont(QFont("Consolas", 12))
        self.key_edit.setPlaceholderText("在此处粘贴激活密钥")
        self.key_edit.setFixedHeight(45)
        input_layout.addWidget(self.key_edit)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        btn_layout = QHBoxLayout()
        activate_btn = QPushButton("立即激活")
        activate_btn.setFixedHeight(50)
        activate_btn.clicked.connect(self.activate)
        
        trial_btn = QPushButton("试用 (15分钟)")
        trial_btn.setFixedHeight(50)
        trial_btn.clicked.connect(self.start_trial)
        
        btn_layout.addWidget(trial_btn, 1)
        btn_layout.addWidget(activate_btn, 2)
        main_layout.addLayout(btn_layout)
        
        contact = QLabel("联系开发者获取密钥: u788990@163.com")
        contact.setAlignment(Qt.AlignCenter)
        contact.setStyleSheet("color: #95a5a6; font-size: 9pt;")
        main_layout.addWidget(contact)
        
        self.setLayout(main_layout)

    def copy_machine_code(self):
        QApplication.clipboard().setText(self.machine_code)
        QMessageBox.information(self, "复制成功", "机器码已复制到剪贴板！")
    
    def activate(self):
        key = self.key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "请输入激活密钥！")
            return
        if LicenseManager.verify_key(self.machine_code, key):
            LicenseManager.save_license()
            self.activated = True
            QMessageBox.information(self, "激活成功", "软件已永久激活！")
            self.accept()
        else:
            QMessageBox.critical(self, "激活失败", "激活密钥无效！")
    
    def start_trial(self):
        reply = QMessageBox.question(self, "确认试用", "每次启动仅限使用 15 分钟，确定要继续吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.trial_mode = True
            self.accept()

# ==================== 依赖检测对话框 ====================
class DependencyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("依赖检测")
        self.setFixedSize(600, 500)
        
        layout = QVBoxLayout()
        
        title = QLabel("依赖库检测结果")
        title.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        layout.addWidget(title)
        
        list_widget = QListWidget()
        list_widget.setFont(QFont("Consolas", 10))
        
        for module, (status, desc) in DependencyChecker.results.items():
            icon = "✓" if status == "ok" else "✗"
            color = "green" if status == "ok" else "red"
            item = QListWidgetItem(f"{icon} {module}: {desc}")
            item.setForeground(QColor(color))
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        if DependencyChecker.missing_required or DependencyChecker.missing_optional:
            cmd_group = QGroupBox("安装命令")
            cmd_layout = QVBoxLayout()
            
            full_cmd_edit = QLineEdit(DependencyChecker.get_full_install_command())
            full_cmd_edit.setReadOnly(True)
            cmd_layout.addWidget(full_cmd_edit)
            
            copy_btn = QPushButton("复制安装命令")
            copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(DependencyChecker.get_full_install_command()))
            cmd_layout.addWidget(copy_btn)
            
            cmd_group.setLayout(cmd_layout)
            layout.addWidget(cmd_group)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

# ==================== 核心图像处理函数 ====================
def play_completion_sound():
    if HAS_WINSOUND:
        try: 
            winsound.MessageBeep(winsound.MB_OK)
        except: 
            pass

def smart_resize_for_model(pil_img: Image.Image, model_id: str) -> tuple:
    """智能调整图片大小"""
    original_size = pil_img.size
    
    if not ModelManager.should_scale_down(model_id):
        return pil_img, original_size, False
    
    info = ModelManager.MODELS.get(model_id, {})
    
    available_mb = HardwareInfo.available_memory_mb
    if available_mb < 2048:
        max_res = 512
    elif available_mb < 4096:
        max_res = 768
    else:
        max_res = 1024
    
    w, h = original_size
    if max(w, h) <= max_res:
        return pil_img, original_size, False
    
    scale = max_res / max(w, h)
    new_w = int(w * scale) // 2 * 2
    new_h = int(h * scale) // 2 * 2
    
    logger.info(f"内存优化缩放: {original_size} -> ({new_w}, {new_h})")
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, original_size, True

def remove_bg_with_session_smart(pil_img: Image.Image, session, model_id: str) -> Image.Image:
    """智能背景移除"""
    if not USE_REMBG or not rembg_remove or not session:
        return pil_img.convert("RGBA")
    
    original_size = pil_img.size
    
    try:
        resized_img, orig_size, was_resized = smart_resize_for_model(pil_img, model_id)
        result = rembg_remove(resized_img, session=session)
        
        if was_resized and result.mode == 'RGBA':
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
            original_rgba = pil_img.convert('RGBA')
            r, g, b, _ = original_rgba.split()
            _, _, _, a = result.split()
            result = Image.merge('RGBA', (r, g, b, a))
        
        return result
        
    except MemoryError:
        logger.error("内存不足，尝试强制缩放...")
        gc.collect()
        
        w, h = original_size
        scale = 512 / max(w, h)
        small_size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
        small_img = pil_img.resize(small_size, Image.Resampling.LANCZOS)
        
        try:
            result = rembg_remove(small_img, session=session)
            result = result.resize(original_size, Image.Resampling.LANCZOS)
            original_rgba = pil_img.convert('RGBA')
            r, g, b, _ = original_rgba.split()
            _, _, _, a = result.split()
            return Image.merge('RGBA', (r, g, b, a))
        except Exception as e:
            logger.error(f"处理失败: {e}")
            return pil_img.convert("RGBA")
    
    except Exception as e:
        logger.error(f"背景移除失败: {e}")
        return pil_img.convert("RGBA")

def remove_bg_with_session(pil_img, session):
    """兼容旧接口"""
    if USE_REMBG and rembg_remove and session:
        try:
            return rembg_remove(pil_img, session=session)
        except Exception as e:
            logger.error(f"背景移除失败: {e}")
            return pil_img.convert("RGBA")
    return pil_img.convert("RGBA")

def cleanup_edge_pixels(pil_img, feather: int = 1, blur: int = 1, gamma: float = 1.2):
    """边缘清理"""
    if not HAS_CV2 or not HAS_NUMPY:
        return pil_img
        
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    
    img_array = np.array(pil_img)
    alpha = img_array[:, :, 3].astype(np.float32) / 255.0
    
    if feather > 0:
        kernel_size = feather * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        alpha = cv2.erode(alpha, kernel, iterations=1)
    
    if blur > 0:
        k_blur = blur * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k_blur, k_blur), 0)
        if gamma != 1.0:
            alpha = np.power(alpha, gamma)
    
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    img_array[:, :, 3] = alpha
    
    return Image.fromarray(img_array, mode='RGBA')

def remove_isolated_colors(pil_img, min_area: int, remove_internal: bool = True, internal_max_area: int = 100):
    """移除孤立色块"""
    if not HAS_CV2 or not HAS_NUMPY:
        return pil_img
        
    if min_area <= 0 and not remove_internal:
        return pil_img
        
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
        
    img_array = np.array(pil_img)
    alpha = img_array[:, :, 3].copy()
    
    _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    
    has_change = False
    
    if min_area > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask_keep = np.zeros_like(alpha)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(mask_keep, [contour], -1, 255, thickness=-1)
                else:
                    has_change = True
            
            alpha = cv2.bitwise_and(alpha, alpha, mask=mask_keep)
            _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    
    if remove_internal and internal_max_area > 0:
        kernel_size = max(3, int(math.sqrt(internal_max_area)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_alpha = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        holes_mask = cv2.bitwise_and(cv2.bitwise_not(binary), closed_alpha)
        
        if cv2.countNonZero(holes_mask) > 0:
            alpha = cv2.add(alpha, holes_mask)
            has_change = True
    
    if not has_change:
        return pil_img
    
    img_array[:, :, 3] = alpha
    return Image.fromarray(img_array, mode='RGBA')

def fill_alpha_with_bg(pil_img, bg_type: str, bg_color: str = "#FFFFFF", bg_image_path: str = None):
    """填充背景"""
    if pil_img.mode != 'RGBA': 
        pil_img = pil_img.convert('RGBA')
    if bg_type == "none": 
        return pil_img
    
    if bg_type == "color":
        c = bg_color.strip().lstrip('#')
        rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) if len(c) == 6 else (255,255,255)
        base = Image.new('RGB', pil_img.size, rgb)
    elif bg_type == "image" and bg_image_path and Path(bg_image_path).exists():
        try:
            base = Image.open(bg_image_path).convert('RGB').resize(pil_img.size, Image.Resampling.LANCZOS)
        except: 
            base = Image.new('RGB', pil_img.size, (255, 255, 255))
    else:
        base = Image.new('RGB', pil_img.size, (255, 255, 255))
    
    base.paste(pil_img, mask=pil_img.split()[-1])
    return base

def process_single_frame(frame_data: tuple, session, params: dict, model_id: str = "u2net") -> tuple:
    """处理单帧"""
    idx, frame_rgb = frame_data
    
    try:
        pil = Image.fromarray(frame_rgb)
        
        if params.get("remove_bg"):
            pil = remove_bg_with_session_smart(pil, session, model_id)
            
            if params.get("cleanup_edge"):
                pil = cleanup_edge_pixels(
                    pil, 
                    params.get("edge_feather", 1), 
                    params.get("edge_blur", 1),
                    params.get("edge_gamma", 1.2)
                )
            if params.get("remove_isolated"):
                pil = remove_isolated_colors(
                    pil, 
                    params.get("isolated_area", 50),
                    params.get("remove_internal", True),
                    params.get("internal_max_area", 100)
                )
            if params.get("bg_type", "none") != "none":
                pil = fill_alpha_with_bg(pil, params.get("bg_type"), params.get("bg_color"), params.get("bg_image"))
        else:
            pil = pil.convert("RGBA")
        
        return (idx, pil, None)
    except Exception as e:
        return (idx, None, str(e))

# ==================== Workers ====================
class BaseWorker(QThread):
    """基础 Worker 类"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._stop = False
    
    def stop(self):
        self._stop = True
        logger.info("正在停止任务...")

class VideoToImagesWorker(BaseWorker):
    def __init__(self, video_path: str, output_dir: str, params: dict):
        super().__init__()
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.params = params

    def run(self):
        if not HAS_CV2:
            self.error.emit("opencv-python 未安装")
            return
            
        try:
            output_folder = self.output_dir / self.video_path.stem
            output_folder.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                self.error.emit(f"无法打开视频：{self.video_path}")
                return
            
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(self.params.get("frame_step", 1)))
            need_remove_bg = self.params.get("remove_bg") and USE_REMBG
            
            if self.params.get("extract_mode") == "first_last":
                frames_idx = [0, max(0, total-1)]
                names = ["_AA", "_BB"]
            else:
                frames_idx = list(range(0, total, step))
                names = [f"_{i+1:06d}" for i in range(len(frames_idx))]
            
            total_frames = len(frames_idx)
            logger.info(f"准备提取 {total_frames} 帧 (共 {total} 帧, 间隔 {step})")
            
            if not need_remove_bg:
                logger.info("快速模式：直接提取帧（无背景处理）")
                saved = 0
                for i, idx in enumerate(frames_idx):
                    if self._stop:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        suffix = names[i]
                        out_path = output_folder / f"{self.video_path.stem}{suffix}.png"
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        Image.fromarray(frame_rgb).save(str(out_path))
                        saved += 1
                    
                    self.progress.emit(int((i + 1) / total_frames * 100), f"提取帧 {i+1}/{total_frames}")
                
                cap.release()
                logger.success(f"快速提取完成: 保存 {saved} 张图片")
                self.finished.emit({"count": saved, "folder": str(output_folder)})
                return
            
            model_name = self.params.get("model_name", "isnet-general-use")
            self.progress.emit(0, f"加载模型 {model_name}...")
            session = ModelManager.load_model(model_name)
            if not session:
                self.error.emit(f"模型加载失败")
                cap.release()
                return
            
            frames_data = []
            for i, idx in enumerate(frames_idx):
                if self._stop: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_data.append((i, frame_rgb))
                self.progress.emit(int((i + 1) / total_frames * 20), f"读取帧 {i+1}/{total_frames}")
            cap.release()
            
            if not frames_data:
                self.error.emit("无有效帧")
                return
            
            num_workers = min(self.params.get("num_threads", 4), len(frames_data))
            processed = 0
            results = {}
            
            logger.info(f"开始处理 {len(frames_data)} 帧 (使用 {num_workers} 线程)")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_single_frame, fd, session, self.params, model_name): fd[0]
                    for fd in frames_data
                }
                
                for future in as_completed(futures):
                    if self._stop:
                        break
                    
                    idx, pil, err = future.result()
                    if err:
                        logger.warning(f"帧 {idx} 处理失败: {err}")
                    else:
                        results[idx] = pil
                    
                    processed += 1
                    self.progress.emit(20 + int(processed / len(frames_data) * 70), f"处理帧 {processed}/{len(frames_data)}")
                    
                    if processed % 10 == 0:
                        gc.collect()
            
            self.progress.emit(90, "保存图片...")
            saved = 0
            for i in range(len(frames_data)):
                if i in results:
                    suffix = names[i] if self.params.get("extract_mode") == "first_last" else f"_{saved+1:06d}"
                    results[i].save(str(output_folder / f"{self.video_path.stem}{suffix}.png"))
                    saved += 1
            
            gc.collect()
            logger.success(f"完成: 保存 {saved} 张图片")
            self.finished.emit({"count": saved, "folder": str(output_folder)})
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            traceback.print_exc()
            self.error.emit(str(e))

class VideoRemoveBgWorker(BaseWorker):
    """视频扣像 Worker"""

    def __init__(self, video_path: str, output_path: str, params: dict):
        super().__init__()
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.params = params

    def run(self):
        if not HAS_CV2:
            self.error.emit("opencv-python 未安装")
            return
            
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                self.error.emit(f"无法打开视频：{self.video_path}")
                return
            
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            model_name = self.params.get("model_name", "isnet-general-use")
            self.progress.emit(0, f"加载模型 {model_name}...")
            session = ModelManager.load_model(model_name)
            if not session:
                self.error.emit(f"模型加载失败")
                cap.release()
                return
            
            output_format = self.params.get("output_format", "mp4")
            
            if output_format == "webm":
                fourcc = cv2.VideoWriter_fourcc(*'VP90')
                out_file = str(self.output_path)
            elif output_format == "mov":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_file = str(self.output_path)
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_file = str(self.output_path)
            
            writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                self.error.emit("无法创建输出视频")
                cap.release()
                return
            
            frame_idx = 0
            processed = 0
            
            logger.info(f"开始处理视频: {total} 帧, {width}x{height}, {fps:.1f}fps")
            
            while True:
                if self._stop:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame_rgb)
                
                pil = remove_bg_with_session_smart(pil, session, model_name)
                
                if self.params.get("cleanup_edge"):
                    pil = cleanup_edge_pixels(
                        pil,
                        self.params.get("edge_feather", 1),
                        self.params.get("edge_blur", 1),
                        self.params.get("edge_gamma", 1.2)
                    )
                
                if self.params.get("remove_isolated"):
                    pil = remove_isolated_colors(
                        pil,
                        self.params.get("isolated_area", 50),
                        self.params.get("remove_internal", True),
                        self.params.get("internal_max_area", 100)
                    )
                
                bg_color = self.params.get("bg_color", "#00FF00")
                pil = fill_alpha_with_bg(pil, "color", bg_color)
                
                frame_out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                writer.write(frame_out)
                
                frame_idx += 1
                processed += 1
                
                if frame_idx % 10 == 0:
                    self.progress.emit(int(frame_idx / total * 100), f"处理帧 {frame_idx}/{total}")
                    gc.collect()
            
            cap.release()
            writer.release()
            gc.collect()
            
            logger.success(f"视频扣像完成: {self.output_path}")
            self.finished.emit({
                "video": str(self.output_path),
                "frames": processed,
                "folder": str(self.output_path.parent)
            })
            
        except Exception as e:
            logger.error(f"视频扣像失败: {e}")
            traceback.print_exc()
            self.error.emit(str(e))

class SpriteWorker(BaseWorker):
    def __init__(self, source_path: str, output_dir: str, params: dict):
        super().__init__()
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.params = params

    def run(self):
        try:
            need_remove_bg = self.params.get("remove_bg") and USE_REMBG
            model_name = self.params.get("model_name", "isnet-general-use")
            session = None
            
            if need_remove_bg:
                self.progress.emit(0, f"加载模型 {model_name}...")
                session = ModelManager.load_model(model_name)
            
            frames_data = []
            frames_pil = []
            
            if self.params.get("source_type") == "video":
                if not HAS_CV2:
                    self.error.emit("opencv-python 未安装")
                    return
                    
                cap = cv2.VideoCapture(str(self.source_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, self.params.get("frame_step", 1))
                
                frame_count = 0
                for i in range(0, total, step):
                    if self._stop: break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, f = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        if need_remove_bg:
                            frames_data.append((frame_count, frame_rgb))
                        else:
                            frames_pil.append(Image.fromarray(frame_rgb).convert("RGBA"))
                        frame_count += 1
                        self.progress.emit(int(i/total*30), f"采样 {frame_count}")
                cap.release()
            else:
                files = sorted([f for f in (self.source_path.glob('*') if self.source_path.is_dir() else [self.source_path]) 
                              if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
                for i, f in enumerate(files):
                    if self._stop: break
                    img = Image.open(f)
                    if need_remove_bg:
                        frames_data.append((i, np.array(img.convert("RGB"))))
                    else:
                        frames_pil.append(img.convert("RGBA"))
                    self.progress.emit(int((i+1)/len(files)*30), f"加载 {i+1}/{len(files)}")

            if need_remove_bg:
                if not frames_data:
                    self.error.emit("无有效帧")
                    return
                
                num_workers = min(self.params.get("num_threads", 4), len(frames_data))
                processed = 0
                results = {}
                
                logger.info(f"处理 {len(frames_data)} 帧 (使用 {num_workers} 线程)")
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(process_single_frame, fd, session, self.params, model_name): fd[0]
                        for fd in frames_data
                    }
                    
                    for future in as_completed(futures):
                        if self._stop:
                            break
                        
                        idx, pil, err = future.result()
                        if not err:
                            results[idx] = pil
                        
                        processed += 1
                        self.progress.emit(30 + int(processed / len(frames_data) * 40), f"处理帧 {processed}/{len(frames_data)}")
                        
                        if processed % 10 == 0:
                            gc.collect()
                
                frames = [results[i] for i in range(len(frames_data)) if i in results]
            else:
                frames = frames_pil
                logger.info(f"快速模式：直接使用 {len(frames)} 帧")
            
            if not frames:
                self.error.emit("无有效帧")
                return

            fw, fh = frames[0].size
            if self.params.get("scale_mode") == "percent":
                sc = self.params.get("scale_percent", 100) / 100
                tw, th = int(fw*sc), int(fh*sc)
            else:
                tw, th = int(self.params.get("thumb_w", 256)), int(self.params.get("thumb_h", 256))
            
            cols = self.params.get("columns", 10)
            rows = math.ceil(len(frames)/cols)
            sheet = Image.new("RGBA", (cols*tw, rows*th))
            
            for idx, fr in enumerate(frames):
                if self._stop: break
                thumb = fr.resize((tw, th), Image.Resampling.LANCZOS)
                c, r = idx % cols, idx // cols
                sheet.paste(thumb, (c*tw, r*th), thumb)
                self.progress.emit(70 + int((idx+1)/len(frames)*30), "合成中...")
            
            out_name = f"{self.source_path.stem}_sprite_{len(frames)}.png"
            out_path = self.output_dir / out_name
            sheet.save(out_path)
            
            gc.collect()
            logger.success(f"精灵图生成完成: {out_path}")
            self.finished.emit({"sheet": str(out_path), "count": len(frames), "folder": str(self.output_dir)})
            
        except Exception as e:
            logger.error(f"精灵图生成失败: {e}")
            traceback.print_exc()
            self.error.emit(str(e))

class VideoToGifWorker(BaseWorker):
    def __init__(self, video_path: str, output_path: str, params: dict):
        super().__init__()
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.params = params

    def run(self):
        if not HAS_CV2:
            self.error.emit("opencv-python 未安装")
            return
            
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, self.params.get("frame_step", 1))
            
            preserve_transparency = self.params.get("preserve_transparency", False)
            need_remove_bg = self.params.get("remove_bg") and USE_REMBG
            model_name = self.params.get("model_name", "isnet-general-use")
            session = None
            
            if need_remove_bg:
                self.progress.emit(0, f"加载模型 {model_name}...")
                session = ModelManager.load_model(model_name)

            total_to_extract = len(range(0, total, step))
            logger.info(f"准备提取 {total_to_extract} 帧 (共 {total} 帧, 间隔 {step})")
            
            if not need_remove_bg:
                logger.info("快速模式：直接提取帧生成 GIF")
                frames = []
                frame_count = 0
                for i in range(0, total, step):
                    if self._stop: break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, f = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(frame_rgb)
                        frames.append(pil)
                        frame_count += 1
                    self.progress.emit(int(frame_count / total_to_extract * 80), f"提取帧 {frame_count}/{total_to_extract}")
                cap.release()
            else:
                frames_data = []
                frame_count = 0
                for i in range(0, total, step):
                    if self._stop: break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, f = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        frames_data.append((frame_count, frame_rgb))
                        frame_count += 1
                    self.progress.emit(int(frame_count / total_to_extract * 20), f"读取帧 {frame_count}/{total_to_extract}")
                cap.release()
                
                if not frames_data:
                    self.error.emit("无帧")
                    return

                num_workers = min(self.params.get("num_threads", 4), len(frames_data))
                processed = 0
                results = {}
                
                logger.info(f"处理 {len(frames_data)} 帧 (使用 {num_workers} 线程)")
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(process_single_frame, fd, session, self.params, model_name): fd[0]
                        for fd in frames_data
                    }
                    
                    for future in as_completed(futures):
                        if self._stop:
                            break
                        
                        idx, pil, err = future.result()
                        if not err:
                            results[idx] = pil
                        
                        processed += 1
                        self.progress.emit(20 + int(processed / len(frames_data) * 60), f"处理帧 {processed}/{len(frames_data)}")
                        
                        if processed % 10 == 0:
                            gc.collect()
                
                frames = [results[i] for i in range(len(frames_data)) if i in results]
            
            if not frames:
                self.error.emit("无帧")
                return

            duration = int(1000 / max(1, self.params.get("fps", 10)))
            
            self.progress.emit(90, "生成 GIF...")
            
            save_kwargs = {
                "save_all": True,
                "append_images": frames[1:],
                "duration": duration,
                "loop": 0
            }

            if preserve_transparency and need_remove_bg:
                converted_frames = []
                for frame in frames:
                    if frame.mode != 'RGBA':
                        frame = frame.convert('RGBA')
                    alpha = frame.split()[-1]
                    frame_p = frame.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
                    mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                    frame_p.paste(255, mask)
                    converted_frames.append(frame_p)
                
                save_kwargs["append_images"] = converted_frames[1:]
                save_kwargs["transparency"] = 255
                save_kwargs["disposal"] = 2
                converted_frames[0].save(str(self.output_path), **save_kwargs)
            else:
                if frames[0].mode == 'RGBA':
                    frames = [fill_alpha_with_bg(f, "color", "#FFFFFF") for f in frames]
                frames[0].save(str(self.output_path), **save_kwargs)

            gc.collect()
            logger.success(f"GIF 生成完成: {self.output_path}")
            self.finished.emit({"gif": str(self.output_path), "count": len(frames), "folder": str(self.output_path.parent)})
            
        except Exception as e:
            logger.error(f"GIF 生成失败: {e}")
            traceback.print_exc()
            self.error.emit(str(e))

class ImagesToGifWorker(BaseWorker):
    def __init__(self, source_path: str, output_path: str, params: dict):
        super().__init__()
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.params = params

    def run(self):
        try:
            if self.source_path.is_dir():
                files = sorted([f for f in self.source_path.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            else: 
                files = [self.source_path]
            
            frames = []
            preserve_transparency = self.params.get("preserve_transparency", False)
            
            for i, f in enumerate(files):
                img = Image.open(f)
                frames.append(img)
                self.progress.emit(int((i+1)/len(files)*100), "加载中")
            
            duration = int(1000 / max(1, self.params.get("fps", 10)))
            save_kwargs = {"save_all": True, "append_images": frames[1:], "duration": duration, "loop": 0}

            if preserve_transparency:
                converted_frames = []
                for frame in frames:
                    if frame.mode != 'RGBA': 
                        frame = frame.convert('RGBA')
                    alpha = frame.split()[-1]
                    frame_p = frame.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
                    mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                    frame_p.paste(255, mask)
                    converted_frames.append(frame_p)
                save_kwargs["append_images"] = converted_frames[1:]
                save_kwargs["transparency"] = 255
                save_kwargs["disposal"] = 2
                converted_frames[0].save(str(self.output_path), **save_kwargs)
            else:
                frames[0].save(str(self.output_path), **save_kwargs)
                
            logger.success(f"GIF 生成完成")
            self.finished.emit({"gif": str(self.output_path), "count": len(frames), "folder": str(self.output_path.parent)})
        except Exception as e:
            logger.error(f"GIF 生成失败: {e}")
            self.error.emit(str(e))

class ImagesToVideoWorker(BaseWorker):
    def __init__(self, source, output, params):
        super().__init__()
        self.source, self.output, self.params = Path(source), Path(output), params
        
    def run(self):
        if not HAS_CV2:
            self.error.emit("opencv-python 未安装")
            return
            
        try:
            files = sorted([f for f in self.source.glob('*') if f.suffix.lower() in ['.png','.jpg']]) if self.source.is_dir() else [self.source]
            if not files: 
                raise Exception("无图片")
            fps = max(1, self.params.get("fps", 24))
            
            first_img = Image.open(files[0])
            w, h = first_img.size
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.output), fourcc, fps, (w, h))
            
            for i, f in enumerate(files):
                if self._stop: break
                img = Image.open(f).convert("RGBA")
                if img.size != (w, h): 
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                bg = fill_alpha_with_bg(img, self.params.get("bg_type", "color"), self.params.get("bg_color", "#FFFFFF"), self.params.get("bg_image"))
                frame = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
                writer.write(frame)
                self.progress.emit(int((i+1)/len(files)*100), f"写入 {i+1}/{len(files)}")
            
            writer.release()
            logger.success(f"视频生成完成")
            self.finished.emit({"video": str(self.output), "folder": str(self.output.parent)})
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            self.error.emit(str(e))

class SingleImageWorker(BaseWorker):
    def __init__(self, input_path, output_path, params):
        super().__init__()
        self.input, self.output, self.params = input_path, output_path, params
        
    def run(self):
        try:
            model_name = self.params.get("model_name", "isnet-general-use")
            
            self.progress.emit(5, f"加载模型 {model_name}...")
            session = ModelManager.load_model(model_name)
            
            if not session:
                self.error.emit(f"模型加载失败")
                return
            
            self.progress.emit(20, "加载图片...")
            pil = Image.open(self.input).convert("RGBA")
            
            self.progress.emit(40, "移除背景...")
            pil = remove_bg_with_session_smart(pil, session, model_name)
            
            if self.params.get("cleanup_edge"): 
                self.progress.emit(60, "清理边缘...")
                pil = cleanup_edge_pixels(
                    pil, 
                    self.params.get("edge_feather", 1), 
                    self.params.get("edge_blur", 1),
                    self.params.get("edge_gamma", 1.2)
                )
            if self.params.get("remove_isolated"): 
                self.progress.emit(75, "移除杂色...")
                pil = remove_isolated_colors(
                    pil, 
                    self.params.get("isolated_area", 50),
                    self.params.get("remove_internal", True),
                    self.params.get("internal_max_area", 100)
                )
            if self.params.get("bg_type", "none") != "none": 
                self.progress.emit(90, "填充背景...")
                pil = fill_alpha_with_bg(pil, self.params.get("bg_type"), self.params.get("bg_color"), self.params.get("bg_image"))
            
            pil.save(self.output)
            self.progress.emit(100, "完成")
            gc.collect()
            logger.success(f"图片处理完成")
            self.finished.emit({"output": str(self.output), "folder": str(Path(self.output).parent)})
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            self.error.emit(str(e))
# ==================== 第三部分：主窗口和程序入口 ====================

# ==================== 主窗口 ====================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.activated = False
        self.trial_mode = False
        # 【修复】使用总秒数进行倒计时，更简单可靠
        self.trial_total_seconds = 15 * 60  # 15分钟 = 900秒
        self.current_worker = None
        self.trial_expired = False  # 标记试用是否已过期
        
        if LicenseManager.check_license_file():
            self.activated = True
        else:
            dialog = ActivationDialog(None)
            if dialog.exec_() == QDialog.Accepted:
                if dialog.activated: 
                    self.activated = True
                elif dialog.trial_mode: 
                    self.trial_mode = True
            else:
                sys.exit()
        
        gpu_status = f"GPU: {HardwareInfo.gpu_name}" if HardwareInfo.gpu_available else "CPU模式"
        base_title = f"别快视频精灵图 v7.6 [{gpu_status}]"
        self.setWindowTitle(f"{base_title} - {'已激活' if self.activated else f'试用 (15:00)'}")
        self.resize(1200, 1000)
        self.setAcceptDrops(True)
        
        self.enable_sound = ConfigManager.get("enable_sound", True)
        self._setup_style()
        self._build_ui()
        
        # 【修复】试用模式定时器
        if self.trial_mode and not self.activated:
            self.trial_timer = QTimer(self)
            self.trial_timer.timeout.connect(self._update_trial_countdown)
            self.trial_timer.start(1000)  # 每秒更新
        
        logger.info("软件启动完成")
        logger.info(f"模型目录: {ConfigManager.get_model_dir()}")
        logger.info(f"biemo 目录: {ConfigManager.get_biemo_dir()}")

    def _update_trial_countdown(self):
        """【修复】更新试用倒计时 - 使用总秒数，逻辑清晰"""
        if self.activated or self.trial_expired:
            return
        
        # 每秒减1
        self.trial_total_seconds -= 1
        
        # 计算分钟和秒
        mins = self.trial_total_seconds // 60
        secs = self.trial_total_seconds % 60
        
        # 更新窗口标题
        gpu_status = f"GPU: {HardwareInfo.gpu_name}" if HardwareInfo.gpu_available else "CPU模式"
        self.setWindowTitle(f"别快视频精灵图 v7.6 [{gpu_status}] - 试用 ({mins:02d}:{secs:02d})")
        
        # 最后1分钟警告
        if self.trial_total_seconds == 60:
            logger.warning("⚠ 试用时间仅剩 1 分钟！")
        
        # 最后30秒警告
        if self.trial_total_seconds == 30:
            logger.warning("⚠ 试用时间仅剩 30 秒！请保存工作。")
        
        # 时间到
        if self.trial_total_seconds <= 0:
            self._handle_trial_expired()
    
    def _handle_trial_expired(self):
        """【修复】处理试用到期 - 安全退出流程"""
        self.trial_expired = True
        
        # 1. 停止定时器
        if hasattr(self, 'trial_timer'):
            self.trial_timer.stop()
        
        # 2. 停止当前任务
        self._stop_current_task()
        
        logger.error("试用时间已到！")
        
        # 3. 更新标题
        gpu_status = f"GPU: {HardwareInfo.gpu_name}" if HardwareInfo.gpu_available else "CPU模式"
        self.setWindowTitle(f"别快视频精灵图 v7.6 [{gpu_status}] - 试用已到期")
        
        # 4. 显示提示对话框
        msg = QMessageBox(self)
        msg.setWindowTitle("试用结束")
        msg.setIcon(QMessageBox.Warning)
        msg.setText("试用时间已到！")
        msg.setInformativeText("程序将在 60 秒后自动退出。\n\n请保存您的工作，或点击'立即退出'。\n\n如需继续使用，请购买激活码。")
        
        exit_now_btn = msg.addButton("立即退出", QMessageBox.DestructiveRole)
        activate_btn = msg.addButton("输入激活码", QMessageBox.ActionRole)
        wait_btn = msg.addButton("等待60秒", QMessageBox.RejectRole)
        
        msg.exec_()
        
        clicked = msg.clickedButton()
        
        if clicked == exit_now_btn:
            # 立即退出
            logger.info("用户选择立即退出")
            QApplication.quit()
            sys.exit(0)
        elif clicked == activate_btn:
            # 尝试激活
            self._show_activation_dialog()
            if not self.activated:
                # 激活失败，启动60秒倒计时
                self._start_exit_countdown()
        else:
            # 等待60秒
            self._start_exit_countdown()
    
    def _start_exit_countdown(self):
        """启动60秒退出倒计时"""
        self.exit_countdown = 60
        
        self.exit_timer = QTimer(self)
        self.exit_timer.timeout.connect(self._exit_countdown_tick)
        self.exit_timer.start(1000)
        
        logger.warning(f"程序将在 {self.exit_countdown} 秒后退出...")
    
    def _exit_countdown_tick(self):
        """退出倒计时"""
        self.exit_countdown -= 1
        
        if self.exit_countdown <= 0:
            self.exit_timer.stop()
            logger.info("退出程序")
            QApplication.quit()
            sys.exit(0)
        
        if self.exit_countdown % 10 == 0:
            logger.warning(f"程序将在 {self.exit_countdown} 秒后退出...")
    
    def _show_activation_dialog(self):
        """显示激活对话框"""
        dialog = ActivationDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.activated:
            self.activated = True
            self.trial_expired = False
            
            # 停止退出定时器
            if hasattr(self, 'exit_timer'):
                self.exit_timer.stop()
            
            # 更新标题
            gpu_status = f"GPU: {HardwareInfo.gpu_name}" if HardwareInfo.gpu_available else "CPU模式"
            self.setWindowTitle(f"别快视频精灵图 v7.6 [{gpu_status}] - 已激活")
            
            logger.success("软件已激活！")
            QMessageBox.information(self, "激活成功", "软件已永久激活！")
    
    def _stop_current_task(self):
        """停止当前正在运行的任务"""
        if self.current_worker and self.current_worker.isRunning():
            logger.warning("正在停止当前任务...")
            self.current_worker.stop()
            self.current_worker.wait(5000)
            if self.current_worker.isRunning():
                self.current_worker.terminate()
            logger.info("任务已停止")

    def _setup_style(self):
        self.setStyleSheet("""
            QWidget { font-family: 'Microsoft YaHei UI'; font-size: 9pt; }
            QGroupBox { font-weight: bold; border: 1px solid #3498db; border-radius: 4px; margin-top: 8px; padding-top: 8px; background: #f8f9fa; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #2c3e50; }
            QPushButton { background: #3498db; color: white; border-radius: 3px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background: #2980b9; }
            QPushButton#actionButton { background: #27ae60; font-size: 10pt; padding: 8px; }
            QPushButton#stopButton { background: #e74c3c; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { padding: 4px; border: 1px solid #bdc3c7; border-radius: 3px; }
            QProgressBar { border: 1px solid #bdc3c7; text-align: center; height: 20px; }
            QProgressBar::chunk { background: #3498db; }
        """)

    def _build_ui(self):
        main = QVBoxLayout()
        
        # 状态栏
        status_layout = QHBoxLayout()
        
        gpu_label = QLabel(f"{'✓ ' + HardwareInfo.gpu_name if HardwareInfo.gpu_available else '○ CPU模式'}")
        gpu_label.setStyleSheet(f"color: {'#27ae60' if HardwareInfo.gpu_available else '#e74c3c'}; font-weight: bold;")
        status_layout.addWidget(gpu_label)
        
        mem_label = QLabel(f"内存: {HardwareInfo.available_memory_mb}MB")
        status_layout.addWidget(mem_label)
        
        if HardwareInfo.gpu_available:
            gpu_mem_label = QLabel(f"显存: {HardwareInfo.gpu_memory_mb}MB")
            status_layout.addWidget(gpu_mem_label)
        
        rembg_label = QLabel(f"{'✓ rembg' if USE_REMBG else '✗ rembg'}")
        rembg_label.setStyleSheet(f"color: {'#27ae60' if USE_REMBG else '#e74c3c'};")
        status_layout.addWidget(rembg_label)
        
        dep_btn = QPushButton("检测依赖")
        dep_btn.setFixedWidth(80)
        dep_btn.clicked.connect(lambda: DependencyDialog(self).exec_())
        status_layout.addWidget(dep_btn)
        
        status_layout.addStretch()
        
        license_label = QLabel(f"{'✓ 已激活' if self.activated else f'试用模式'}")
        license_label.setStyleSheet(f"color: {'#27ae60' if self.activated else '#e74c3c'}; font-weight: bold;")
        status_layout.addWidget(license_label)
        
        main.addLayout(status_layout)

        # 主内容
        splitter = QSplitter(Qt.Vertical)
        
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_sprite_tab(), "精灵图")
        self.tabs.addTab(self._build_video_extract_tab(), "视频转图")
        self.tabs.addTab(self._build_video_rembg_tab(), "视频扣像")
        self.tabs.addTab(self._build_images_to_video_tab(), "图片转视频")
        self.tabs.addTab(self._build_gif_tab(), "视频转GIF")
        self.tabs.addTab(self._build_single_image_tab(), "图片扣图")
        self.tabs.addTab(self._build_settings_tab(), "设置")
        tab_layout.addWidget(self.tabs)
        
        splitter.addWidget(tab_widget)
        
        # 日志
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_widget = LogWidget()
        self.log_widget.setMinimumHeight(150)
        log_layout.addWidget(self.log_widget)
        
        log_btn_layout = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(lambda: self.log_widget.clear())
        log_btn_layout.addWidget(clear_log_btn)
        
        clear_cache_btn = QPushButton("清除模型缓存")
        clear_cache_btn.clicked.connect(ModelManager.clear_cache)
        log_btn_layout.addWidget(clear_cache_btn)
        
        refresh_models_btn = QPushButton("刷新模型状态")
        refresh_models_btn.clicked.connect(self._refresh_all_model_selectors)
        log_btn_layout.addWidget(refresh_models_btn)
        
        stop_btn = QPushButton("停止当前任务")
        stop_btn.setObjectName("stopButton")
        stop_btn.clicked.connect(self._stop_current_task)
        log_btn_layout.addWidget(stop_btn)
        
        log_btn_layout.addStretch()
        log_layout.addLayout(log_btn_layout)
        
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)
        
        splitter.setSizes([700, 200])
        main.addWidget(splitter)
        
        self.setLayout(main)
    
    def _refresh_all_model_selectors(self):
        """刷新所有模型选择器"""
        ModelManager.scan_models()
        for selector in [self.sprite_model, self.extract_model, self.gif_model, self.single_model, self.beiou_model]:
            selector.refresh_models()
        logger.info("模型状态已刷新")

    def create_file_input(self, btn_callback, placeholder="拖入文件或点击选择..."):
        layout = QHBoxLayout()
        line_edit = FileDropLineEdit(placeholder=placeholder)
        btn = QPushButton("选择")
        btn.clicked.connect(btn_callback)
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        return layout, line_edit

    def create_hint_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("color: #7f8c8d; font-size: 8pt; font-style: italic;")
        label.setWordWrap(True)
        return label

    def create_thread_selector(self):
        spin = QSpinBox()
        spin.setRange(1, HardwareInfo.cpu_threads * 2)
        spin.setValue(min(4, HardwareInfo.cpu_threads))
        return spin

    def _on_model_changed(self, model_id: str, status: dict):
        """模型选择变化回调"""
        pass

    def _build_sprite_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        
        src_grp = QGroupBox("源文件")
        self.sprite_source_type = QButtonGroup(w)
        r1 = QRadioButton("视频"); r1.setChecked(True)
        r2 = QRadioButton("图片文件夹")
        self.sprite_source_type.addButton(r1, 0)
        self.sprite_source_type.addButton(r2, 1)
        
        hl = QHBoxLayout()
        hl.addWidget(r1)
        hl.addWidget(r2)
        hl.addStretch()
        src_grp.setLayout(QVBoxLayout())
        src_grp.layout().addLayout(hl)
        
        inp_layout, self.sprite_path_edit = self.create_file_input(self.sprite_select_source)
        src_grp.layout().addLayout(inp_layout)
        layout.addWidget(src_grp)

        model_grp = QGroupBox("AI 模型")
        ml = QGridLayout()
        ml.addWidget(QLabel("选择模型:"), 0, 0)
        self.sprite_model = ModelSelector()
        self.sprite_model.model_changed.connect(self._on_model_changed)
        ml.addWidget(self.sprite_model, 0, 1, 1, 2)
        ml.addWidget(QLabel("并行线程:"), 0, 3)
        self.sprite_threads = self.create_thread_selector()
        ml.addWidget(self.sprite_threads, 0, 4)
        
        hint = self.create_hint_label("★ = 已加载 | ✓ = 已下载 | ○ = 需下载 | 🔴 = 大模型")
        ml.addWidget(hint, 1, 0, 1, 5)
        model_grp.setLayout(ml)
        layout.addWidget(model_grp)

        set_grp = QGroupBox("设置")
        sl = QGridLayout()
        sl.addWidget(QLabel("帧间隔:"), 0, 0)
        self.sprite_step = QSpinBox()
        self.sprite_step.setRange(1, 1000)
        self.sprite_step.setValue(1)
        sl.addWidget(self.sprite_step, 0, 1)
        sl.addWidget(QLabel("列数:"), 0, 2)
        self.sprite_cols = QSpinBox()
        self.sprite_cols.setRange(1, 100)
        self.sprite_cols.setValue(10)
        sl.addWidget(self.sprite_cols, 0, 3)
        
        self.sprite_percent = QRadioButton("百分比")
        self.sprite_percent.setChecked(True)
        self.sprite_fixed = QRadioButton("固定尺寸")
        sl.addWidget(self.sprite_percent, 1, 0)
        sl.addWidget(self.sprite_fixed, 1, 1)
        
        self.sprite_scale_val = QDoubleSpinBox()
        self.sprite_scale_val.setValue(100)
        self.sprite_scale_val.setRange(1, 1000)
        sl.addWidget(self.sprite_scale_val, 1, 2)
        
        self.sprite_w = QSpinBox()
        self.sprite_w.setValue(256)
        self.sprite_w.setRange(1, 4096)
        self.sprite_w.setEnabled(False)
        self.sprite_h = QSpinBox()
        self.sprite_h.setValue(256)
        self.sprite_h.setRange(1, 4096)
        self.sprite_h.setEnabled(False)
        
        wh_layout = QHBoxLayout()
        wh_layout.addWidget(self.sprite_w)
        wh_layout.addWidget(QLabel("x"))
        wh_layout.addWidget(self.sprite_h)
        sl.addLayout(wh_layout, 1, 3)
        
        self.sprite_percent.toggled.connect(lambda c: [self.sprite_w.setEnabled(not c), self.sprite_h.setEnabled(not c), self.sprite_scale_val.setEnabled(c)])
        
        set_grp.setLayout(sl)
        layout.addWidget(set_grp)
        
        bg_grp = QGroupBox("背景移除与清理")
        bl = QGridLayout()
        
        self.sprite_rembg = QCheckBox("启用背景移除")
        bl.addWidget(self.sprite_rembg, 0, 0, 1, 2)
        
        self.sprite_clean = QCheckBox("边缘清理")
        self.sprite_clean.setEnabled(False)
        bl.addWidget(self.sprite_clean, 1, 0)
        bl.addWidget(QLabel("腐蚀:"), 1, 1)
        self.sprite_feather = QSpinBox()
        self.sprite_feather.setValue(1)
        self.sprite_feather.setRange(0, 10)
        bl.addWidget(self.sprite_feather, 1, 2)
        bl.addWidget(QLabel("模糊:"), 1, 3)
        self.sprite_blur = QSpinBox()
        self.sprite_blur.setValue(1)
        self.sprite_blur.setRange(0, 10)
        bl.addWidget(self.sprite_blur, 1, 4)
        bl.addWidget(QLabel("Gamma:"), 1, 5)
        self.sprite_gamma = QDoubleSpinBox()
        self.sprite_gamma.setValue(1.2)
        self.sprite_gamma.setRange(0.5, 2.0)
        self.sprite_gamma.setSingleStep(0.1)
        bl.addWidget(self.sprite_gamma, 1, 6)
        
        self.sprite_iso = QCheckBox("移除孤立色块")
        self.sprite_iso.setEnabled(False)
        bl.addWidget(self.sprite_iso, 2, 0, 1, 2)
        bl.addWidget(QLabel("最小保留:"), 2, 2)
        self.sprite_iso_area = QSpinBox()
        self.sprite_iso_area.setValue(50)
        self.sprite_iso_area.setRange(1, 50000)
        bl.addWidget(self.sprite_iso_area, 2, 3)
        
        self.sprite_internal = QCheckBox("清理内部孔洞")
        self.sprite_internal.setEnabled(False)
        self.sprite_internal.setChecked(True)
        bl.addWidget(self.sprite_internal, 2, 4, 1, 2)
        bl.addWidget(QLabel("孔洞最大:"), 2, 6)
        self.sprite_internal_area = QSpinBox()
        self.sprite_internal_area.setValue(100)
        self.sprite_internal_area.setRange(1, 10000)
        bl.addWidget(self.sprite_internal_area, 3, 0)
        
        self.sprite_rembg.stateChanged.connect(lambda s: [
            self.sprite_clean.setEnabled(s), 
            self.sprite_iso.setEnabled(s),
            self.sprite_internal.setEnabled(s)
        ])
        
        bg_grp.setLayout(bl)
        layout.addWidget(bg_grp)
        
        btn = QPushButton("生成精灵图")
        btn.setObjectName("actionButton")
        btn.clicked.connect(self.sprite_run)
        layout.addWidget(btn)
        
        self.sprite_prog = QProgressBar()
        layout.addWidget(self.sprite_prog)
        w.setLayout(layout)
        return w

    def _build_video_extract_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        
        grp = QGroupBox("视频源")
        l, self.extract_path_edit = self.create_file_input(self.extract_select)
        grp.setLayout(l)
        layout.addWidget(grp)
        
        model_grp = QGroupBox("AI 模型")
        ml = QGridLayout()
        ml.addWidget(QLabel("选择模型:"), 0, 0)
        self.extract_model = ModelSelector()
        self.extract_model.model_changed.connect(self._on_model_changed)
        ml.addWidget(self.extract_model, 0, 1, 1, 2)
        ml.addWidget(QLabel("并行线程:"), 0, 3)
        self.extract_threads = self.create_thread_selector()
        ml.addWidget(self.extract_threads, 0, 4)
        model_grp.setLayout(ml)
        layout.addWidget(model_grp)
        
        opt = QGroupBox("提取选项")
        ol = QGridLayout()
        self.extract_mode = QButtonGroup(w)
        r1 = QRadioButton("首尾帧")
        r1.setChecked(True)
        self.extract_mode.addButton(r1, 0)
        r2 = QRadioButton("全部帧")
        self.extract_mode.addButton(r2, 1)
        ol.addWidget(r1, 0, 0)
        ol.addWidget(r2, 0, 1)
        ol.addWidget(QLabel("间隔:"), 0, 2)
        self.extract_step = QSpinBox()
        self.extract_step.setRange(1, 1000)
        self.extract_step.setValue(1)
        ol.addWidget(self.extract_step, 0, 3)
        
        self.extract_rembg = QCheckBox("移除背景")
        ol.addWidget(self.extract_rembg, 1, 0)
        self.extract_bg_type = QComboBox()
        self.extract_bg_type.addItems(["none", "color", "image"])
        ol.addWidget(self.extract_bg_type, 1, 1)
        
        # 【修复】使用颜色选择器
        ol.addWidget(QLabel("背景色:"), 1, 2)
        self.extract_bg_color = ColorPickerWidget("#FFFFFF")
        ol.addWidget(self.extract_bg_color, 1, 3)
        
        self.extract_bg_img = QLineEdit("背景图路径...")
        ol.addWidget(self.extract_bg_img, 2, 0, 1, 4)
        
        opt.setLayout(ol)
        layout.addWidget(opt)
        
        clean_grp = QGroupBox("清理选项")
        cl = QGridLayout()
        
        self.extract_clean = QCheckBox("边缘清理")
        cl.addWidget(self.extract_clean, 0, 0)
        cl.addWidget(QLabel("腐蚀:"), 0, 1)
        self.extract_feather = QSpinBox()
        self.extract_feather.setValue(1)
        self.extract_feather.setRange(0, 10)
        cl.addWidget(self.extract_feather, 0, 2)
        cl.addWidget(QLabel("模糊:"), 0, 3)
        self.extract_blur = QSpinBox()
        self.extract_blur.setValue(1)
        self.extract_blur.setRange(0, 10)
        cl.addWidget(self.extract_blur, 0, 4)
        cl.addWidget(QLabel("Gamma:"), 0, 5)
        self.extract_gamma = QDoubleSpinBox()
        self.extract_gamma.setValue(1.2)
        self.extract_gamma.setRange(0.5, 2.0)
        self.extract_gamma.setSingleStep(0.1)
        cl.addWidget(self.extract_gamma, 0, 6)
        
        self.extract_iso = QCheckBox("移除孤立色块")
        cl.addWidget(self.extract_iso, 1, 0)
        cl.addWidget(QLabel("最小保留:"), 1, 1)
        self.extract_iso_area = QSpinBox()
        self.extract_iso_area.setValue(50)
        self.extract_iso_area.setRange(1, 50000)
        cl.addWidget(self.extract_iso_area, 1, 2)
        
        self.extract_internal = QCheckBox("清理内部孔洞")
        self.extract_internal.setChecked(True)
        cl.addWidget(self.extract_internal, 1, 3)
        cl.addWidget(QLabel("孔洞最大:"), 1, 4)
        self.extract_internal_area = QSpinBox()
        self.extract_internal_area.setValue(100)
        self.extract_internal_area.setRange(1, 10000)
        cl.addWidget(self.extract_internal_area, 1, 5)
        
        clean_grp.setLayout(cl)
        layout.addWidget(clean_grp)
        
        btn = QPushButton("开始提取")
        btn.setObjectName("actionButton")
        btn.clicked.connect(self.extract_run)
        layout.addWidget(btn)
        self.extract_prog = QProgressBar()
        layout.addWidget(self.extract_prog)
        w.setLayout(layout)
        return w

    def _build_video_rembg_tab(self):
        """视频扣像 Tab"""
        w = QWidget()
        layout = QVBoxLayout()
        
        src_grp = QGroupBox("视频源")
        l, self.beiou_path_edit = self.create_file_input(self.beiou_select)
        src_grp.setLayout(l)
        layout.addWidget(src_grp)
        
        model_grp = QGroupBox("AI 模型")
        ml = QGridLayout()
        ml.addWidget(QLabel("选择模型:"), 0, 0)
        self.beiou_model = ModelSelector()
        self.beiou_model.model_changed.connect(self._on_model_changed)
        ml.addWidget(self.beiou_model, 0, 1, 1, 2)
        
        hint = self.create_hint_label("建议使用 ISNet 或 U²-Net 系列模型")
        ml.addWidget(hint, 1, 0, 1, 3)
        model_grp.setLayout(ml)
        layout.addWidget(model_grp)
        
        out_grp = QGroupBox("输出设置")
        ol = QGridLayout()
        
        ol.addWidget(QLabel("输出格式:"), 0, 0)
        self.beiou_format = QComboBox()
        self.beiou_format.addItems(["mp4 (绿幕/自定义背景)", "avi", "webm"])
        ol.addWidget(self.beiou_format, 0, 1)
        
        # 【修复】使用颜色选择器
        ol.addWidget(QLabel("背景色:"), 0, 2)
        self.beiou_bg_color = ColorPickerWidget("#00FF00")  # 默认绿幕
        ol.addWidget(self.beiou_bg_color, 0, 3)
        
        format_hint = self.create_hint_label("视频输出需要填充背景色（默认绿幕），后期可用视频软件抠除")
        ol.addWidget(format_hint, 1, 0, 1, 4)
        
        out_grp.setLayout(ol)
        layout.addWidget(out_grp)
        
        post_grp = QGroupBox("后处理选项")
        pl = QGridLayout()
        
        self.beiou_clean = QCheckBox("边缘清理")
        pl.addWidget(self.beiou_clean, 0, 0)
        pl.addWidget(QLabel("腐蚀:"), 0, 1)
        self.beiou_feather = QSpinBox()
        self.beiou_feather.setValue(1)
        self.beiou_feather.setRange(0, 10)
        pl.addWidget(self.beiou_feather, 0, 2)
        pl.addWidget(QLabel("模糊:"), 0, 3)
        self.beiou_blur = QSpinBox()
        self.beiou_blur.setValue(1)
        self.beiou_blur.setRange(0, 10)
        pl.addWidget(self.beiou_blur, 0, 4)
        pl.addWidget(QLabel("Gamma:"), 0, 5)
        self.beiou_gamma = QDoubleSpinBox()
        self.beiou_gamma.setValue(1.2)
        self.beiou_gamma.setRange(0.5, 2.0)
        self.beiou_gamma.setSingleStep(0.1)
        pl.addWidget(self.beiou_gamma, 0, 6)
        
        self.beiou_iso = QCheckBox("移除孤立色块")
        pl.addWidget(self.beiou_iso, 1, 0)
        pl.addWidget(QLabel("最小保留:"), 1, 1)
        self.beiou_iso_area = QSpinBox()
        self.beiou_iso_area.setValue(50)
        self.beiou_iso_area.setRange(1, 50000)
        pl.addWidget(self.beiou_iso_area, 1, 2)
        
        self.beiou_internal = QCheckBox("清理内部孔洞")
        self.beiou_internal.setChecked(True)
        pl.addWidget(self.beiou_internal, 1, 3)
        pl.addWidget(QLabel("孔洞最大:"), 1, 4)
        self.beiou_internal_area = QSpinBox()
        self.beiou_internal_area.setValue(100)
        self.beiou_internal_area.setRange(1, 10000)
        pl.addWidget(self.beiou_internal_area, 1, 5)
        
        post_grp.setLayout(pl)
        layout.addWidget(post_grp)
        
        btn = QPushButton("开始视频扣像")
        btn.setObjectName("actionButton")
        btn.clicked.connect(self.beiou_run)
        layout.addWidget(btn)
        
        self.beiou_prog = QProgressBar()
        layout.addWidget(self.beiou_prog)
        
        w.setLayout(layout)
        return w

    def _build_images_to_video_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        
        grp = QGroupBox("图片源")
        l, self.vid_src_edit = self.create_file_input(self.vid_select)
        grp.setLayout(l)
        layout.addWidget(grp)
        
        opt = QGroupBox("参数")
        ol = QGridLayout()
        ol.addWidget(QLabel("FPS:"), 0, 0)
        self.vid_fps = QSpinBox()
        self.vid_fps.setValue(24)
        self.vid_fps.setRange(1, 120)
        ol.addWidget(self.vid_fps, 0, 1)
        ol.addWidget(QLabel("背景填充:"), 0, 2)
        self.vid_bg_type = QComboBox()
        self.vid_bg_type.addItems(["none", "color", "image"])
        ol.addWidget(self.vid_bg_type, 0, 3)
        
        # 【修复】使用颜色选择器
        ol.addWidget(QLabel("背景色:"), 1, 0)
        self.vid_bg_color = ColorPickerWidget("#FFFFFF")
        ol.addWidget(self.vid_bg_color, 1, 1, 1, 3)
        
        opt.setLayout(ol)
        layout.addWidget(opt)
        
        btn = QPushButton("合成视频")
        btn.setObjectName("actionButton")
        btn.clicked.connect(self.vid_run)
        layout.addWidget(btn)
        self.vid_prog = QProgressBar()
        layout.addWidget(self.vid_prog)
        layout.addStretch()
        w.setLayout(layout)
        return w

    def _build_gif_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        
        src_grp = QGroupBox("源")
        self.gif_src_type = QButtonGroup(w)
        r1 = QRadioButton("视频")
        r1.setChecked(True)
        r2 = QRadioButton("图片文件夹")
        self.gif_src_type.addButton(r1)
        self.gif_src_type.addButton(r2)
        hl = QHBoxLayout()
        hl.addWidget(r1)
        hl.addWidget(r2)
        hl.addStretch()
        src_grp.setLayout(QVBoxLayout())
        src_grp.layout().addLayout(hl)
        
        l, self.gif_src_edit = self.create_file_input(self.gif_select)
        src_grp.layout().addLayout(l)
        layout.addWidget(src_grp)
        
        model_grp = QGroupBox("AI 模型")
        ml = QGridLayout()
        ml.addWidget(QLabel("选择模型:"), 0, 0)
        self.gif_model = ModelSelector()
        self.gif_model.model_changed.connect(self._on_model_changed)
        ml.addWidget(self.gif_model, 0, 1, 1, 2)
        ml.addWidget(QLabel("并行线程:"), 0, 3)
        self.gif_threads = self.create_thread_selector()
        ml.addWidget(self.gif_threads, 0, 4)
        model_grp.setLayout(ml)
        layout.addWidget(model_grp)
        
        opt = QGroupBox("GIF 参数")
        ol = QGridLayout()
        ol.addWidget(QLabel("FPS:"), 0, 0)
        self.gif_fps = QSpinBox()
        self.gif_fps.setValue(10)
        self.gif_fps.setRange(1, 60)
        ol.addWidget(self.gif_fps, 0, 1)
        ol.addWidget(QLabel("间隔:"), 0, 2)
        self.gif_step = QSpinBox()
        self.gif_step.setRange(1, 1000)
        self.gif_step.setValue(1)
        ol.addWidget(self.gif_step, 0, 3)
        
        self.gif_transparency = QCheckBox("保留透明通道")
        ol.addWidget(self.gif_transparency, 1, 0, 1, 2)
        self.gif_rembg = QCheckBox("移除背景")
        ol.addWidget(self.gif_rembg, 1, 2, 1, 2)
        
        opt.setLayout(ol)
        layout.addWidget(opt)
        
        clean_grp = QGroupBox("清理选项")
        cl = QGridLayout()
        
        self.gif_clean = QCheckBox("边缘清理")
        cl.addWidget(self.gif_clean, 0, 0)
        cl.addWidget(QLabel("腐蚀:"), 0, 1)
        self.gif_feather = QSpinBox()
        self.gif_feather.setValue(1)
        self.gif_feather.setRange(0, 10)
        cl.addWidget(self.gif_feather, 0, 2)
        cl.addWidget(QLabel("模糊:"), 0, 3)
        self.gif_blur = QSpinBox()
        self.gif_blur.setValue(1)
        self.gif_blur.setRange(0, 10)
        cl.addWidget(self.gif_blur, 0, 4)
        cl.addWidget(QLabel("Gamma:"), 0, 5)
        self.gif_gamma = QDoubleSpinBox()
        self.gif_gamma.setValue(1.2)
        self.gif_gamma.setRange(0.5, 2.0)
        self.gif_gamma.setSingleStep(0.1)
        cl.addWidget(self.gif_gamma, 0, 6)
        
        self.gif_iso = QCheckBox("移除孤立色块")
        cl.addWidget(self.gif_iso, 1, 0)
        cl.addWidget(QLabel("最小保留:"), 1, 1)
        self.gif_iso_area = QSpinBox()
        self.gif_iso_area.setValue(50)
        self.gif_iso_area.setRange(1, 50000)
        cl.addWidget(self.gif_iso_area, 1, 2)
        
        self.gif_internal = QCheckBox("清理内部孔洞")
        self.gif_internal.setChecked(True)
        cl.addWidget(self.gif_internal, 1, 3)
        cl.addWidget(QLabel("孔洞最大:"), 1, 4)
        self.gif_internal_area = QSpinBox()
        self.gif_internal_area.setValue(100)
        self.gif_internal_area.setRange(1, 10000)
        cl.addWidget(self.gif_internal_area, 1, 5)
        
        self.gif_bg_type = QComboBox()
        self.gif_bg_type.addItems(["none", "color"])
        cl.addWidget(QLabel("背景:"), 2, 0)
        cl.addWidget(self.gif_bg_type, 2, 1)
        
        # 【修复】使用颜色选择器
        self.gif_bg_color = ColorPickerWidget("#FFFFFF")
        cl.addWidget(self.gif_bg_color, 2, 2, 1, 2)
        
        self.gif_transparency.stateChanged.connect(lambda s: [self.gif_bg_type.setEnabled(not s), self.gif_bg_color.setEnabled(not s)])

        clean_grp.setLayout(cl)
        layout.addWidget(clean_grp)
        
        btn = QPushButton("生成 GIF")
        btn.setObjectName("actionButton")
        btn.clicked.connect(self.gif_run)
        layout.addWidget(btn)
        self.gif_prog = QProgressBar()
        layout.addWidget(self.gif_prog)
        w.setLayout(layout)
        return w

    def _build_single_image_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        
        grp = QGroupBox("单图")
        l, self.single_src_edit = self.create_file_input(self.single_select)
        grp.setLayout(l)
        layout.addWidget(grp)
        
        model_grp = QGroupBox("AI 模型")
        ml = QHBoxLayout()
        ml.addWidget(QLabel("选择模型:"))
        self.single_model = ModelSelector()
        self.single_model.model_changed.connect(self._on_model_changed)
        ml.addWidget(self.single_model)
        ml.addStretch()
        model_grp.setLayout(ml)
        layout.addWidget(model_grp)
        
        opt = QGroupBox("处理选项")
        ol = QGridLayout()
        
        self.single_clean = QCheckBox("边缘清理")
        ol.addWidget(self.single_clean, 0, 0)
        ol.addWidget(QLabel("腐蚀:"), 0, 1)
        self.single_feather = QSpinBox()
        self.single_feather.setValue(1)
        self.single_feather.setRange(0, 10)
        ol.addWidget(self.single_feather, 0, 2)
        ol.addWidget(QLabel("模糊:"), 0, 3)
        self.single_blur = QSpinBox()
        self.single_blur.setValue(1)
        self.single_blur.setRange(0, 10)
        ol.addWidget(self.single_blur, 0, 4)
        ol.addWidget(QLabel("Gamma:"), 0, 5)
        self.single_gamma = QDoubleSpinBox()
        self.single_gamma.setValue(1.2)
        self.single_gamma.setRange(0.5, 2.0)
        self.single_gamma.setSingleStep(0.1)
        ol.addWidget(self.single_gamma, 0, 6)
        
        self.single_iso = QCheckBox("去杂色")
        ol.addWidget(self.single_iso, 1, 0)
        ol.addWidget(QLabel("最小保留:"), 1, 1)
        self.single_iso_area = QSpinBox()
        self.single_iso_area.setValue(50)
        self.single_iso_area.setRange(1, 50000)
        ol.addWidget(self.single_iso_area, 1, 2)
        
        self.single_internal = QCheckBox("清理内部孔洞")
        self.single_internal.setChecked(True)
        ol.addWidget(self.single_internal, 1, 3)
        ol.addWidget(QLabel("孔洞最大:"), 1, 4)
        self.single_internal_area = QSpinBox()
        self.single_internal_area.setValue(100)
        self.single_internal_area.setRange(1, 10000)
        ol.addWidget(self.single_internal_area, 1, 5)
        
        self.single_bg_type = QComboBox()
        self.single_bg_type.addItems(["none", "color"])
        ol.addWidget(self.single_bg_type, 2, 0)
        
        # 【修复】使用颜色选择器
        self.single_bg_color = ColorPickerWidget("#FFFFFF")
        ol.addWidget(self.single_bg_color, 2, 1, 1, 3)
        
        opt.setLayout(ol)
        layout.addWidget(opt)
        
        btn = QPushButton("处理图片")
        btn.setObjectName("actionButton")
        btn.clicked.connect(self.single_run)
        layout.addWidget(btn)
        self.single_prog = QProgressBar()
        layout.addWidget(self.single_prog)
        layout.addStretch()
        w.setLayout(layout)
        return w

    def _build_settings_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        
        model_grp = QGroupBox("模型设置")
        ml = QGridLayout()
        
        ml.addWidget(QLabel("模型存储目录:"), 0, 0)
        self.model_dir_edit = QLineEdit(ConfigManager.get_model_dir())
        self.model_dir_edit.setReadOnly(True)
        ml.addWidget(self.model_dir_edit, 0, 1)
        
        open_model_dir_btn = QPushButton("打开目录")
        open_model_dir_btn.clicked.connect(lambda: os.startfile(ConfigManager.get_model_dir()) if os.path.exists(ConfigManager.get_model_dir()) else None)
        ml.addWidget(open_model_dir_btn, 0, 2)
        
        hint = self.create_hint_label('将 .onnx 模型文件放入此目录，然后点击"刷新模型状态"即可使用自定义模型')
        ml.addWidget(hint, 1, 0, 1, 3)
        
        model_grp.setLayout(ml)
        layout.addWidget(model_grp)
        
        hw_grp = QGroupBox("硬件信息")
        hl = QGridLayout()
        hl.addWidget(QLabel("GPU:"), 0, 0)
        hl.addWidget(QLabel(f"{'✓ ' + HardwareInfo.gpu_name if HardwareInfo.gpu_available else '○ 未检测到'}"), 0, 1)
        hl.addWidget(QLabel("GPU 显存:"), 1, 0)
        hl.addWidget(QLabel(f"{HardwareInfo.gpu_memory_mb} MB" if HardwareInfo.gpu_available else "N/A"), 1, 1)
        hl.addWidget(QLabel("ONNX 提供程序:"), 2, 0)
        hl.addWidget(QLabel(", ".join(HardwareInfo.onnx_providers) if HardwareInfo.onnx_providers else "N/A"), 2, 1)
        hl.addWidget(QLabel("CPU 线程:"), 3, 0)
        hl.addWidget(QLabel(str(HardwareInfo.cpu_threads)), 3, 1)
        hl.addWidget(QLabel("可用内存:"), 4, 0)
        hl.addWidget(QLabel(f"{HardwareInfo.available_memory_mb} MB"), 4, 1)
        hl.addWidget(QLabel("rembg:"), 5, 0)
        hl.addWidget(QLabel(f"{'✓ 已安装' if USE_REMBG else '✗ 未安装'}"), 5, 1)
        hw_grp.setLayout(hl)
        layout.addWidget(hw_grp)
        
        act = QGroupBox("激活信息")
        al = QVBoxLayout()
        al.addWidget(QLabel(f"机器码: {LicenseManager.get_machine_code()}"))
        al.addWidget(QLabel(f"激活文件: {LicenseManager.get_license_file()}"))
        if not self.activated:
            btn = QPushButton("输入激活码")
            btn.clicked.connect(self._show_activation_dialog)
            al.addWidget(btn)
        act.setLayout(al)
        layout.addWidget(act)
        
        pg = QGroupBox("输出路径 (biemo 目录)")
        pgl = QGridLayout()
        self.path_edits = {}
        output_paths = ConfigManager.get("output_paths", ConfigManager.DEFAULT_CONFIG["output_paths"])
        for i, (k, v) in enumerate(output_paths.items()):
            pgl.addWidget(QLabel(k), i, 0)
            le = QLineEdit(ConfigManager.get_output_path(k))
            le.setReadOnly(True)
            self.path_edits[k] = le
            pgl.addWidget(le, i, 1)
            btn = QPushButton("打开")
            btn.setFixedWidth(50)
            btn.clicked.connect(lambda _, path=ConfigManager.get_output_path(k): os.startfile(path) if os.path.exists(path) else None)
            pgl.addWidget(btn, i, 2)
        
        pg.setLayout(pgl)
        layout.addWidget(pg)
        
        s_box = QCheckBox("开启完成音效")
        s_box.setChecked(self.enable_sound)
        s_box.stateChanged.connect(lambda s: [setattr(self, 'enable_sound', s), ConfigManager.set("enable_sound", s)])
        layout.addWidget(s_box)
        
        layout.addStretch()
        w.setLayout(layout)
        return w

    def sprite_select_source(self): 
        self._select_file(self.sprite_path_edit, file_mode=self.sprite_source_type.checkedId()==0)
    def extract_select(self): 
        self._select_file(self.extract_path_edit, file_mode=True)
    def vid_select(self): 
        self._select_file(self.vid_src_edit, file_mode=False)
    def gif_select(self): 
        self._select_file(self.gif_src_edit, file_mode=self.gif_src_type.checkedButton().text()=="视频")
    def single_select(self): 
        self._select_file(self.single_src_edit, file_mode=True, filter="Img (*.png *.jpg *.bmp)")
    def beiou_select(self):
        self._select_file(self.beiou_path_edit, file_mode=True, filter="Video (*.mp4 *.avi *.mov *.mkv *.webm)")

    def _select_file(self, edit_widget, file_mode=True, filter="Video (*.mp4 *.avi *.mov *.mkv)"):
        if file_mode: 
            f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filter)
        else: 
            f = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if f: 
            edit_widget.setText(f)

    def show_result_dialog(self, folder_path):
        if self.enable_sound: 
            play_completion_sound()
        msg = QMessageBox(self)
        msg.setWindowTitle("任务完成")
        msg.setText("处理已完成！")
        msg.setIcon(QMessageBox.Information)
        
        open_btn = msg.addButton("打开文件夹", QMessageBox.ActionRole)
        msg.addButton("关闭", QMessageBox.RejectRole)
        msg.exec_()
        
        if msg.clickedButton() == open_btn:
            try:
                os.startfile(folder_path)
            except:
                try: 
                    subprocess.Popen(['xdg-open', folder_path])
                except: 
                    pass

    def sprite_run(self):
        path = self.sprite_path_edit.text()
        if not path: 
            logger.warning("请先选择源文件")
            return
        
        params = {
            "source_type": "video" if self.sprite_source_type.checkedId()==0 else "images",
            "model_name": self.sprite_model.get_current_model(),
            "num_threads": self.sprite_threads.value(),
            "frame_step": max(1, self.sprite_step.value()),
            "columns": max(1, self.sprite_cols.value()),
            "scale_mode": "percent" if self.sprite_percent.isChecked() else "fixed",
            "scale_percent": self.sprite_scale_val.value(),
            "thumb_w": self.sprite_w.value(), 
            "thumb_h": self.sprite_h.value(),
            "remove_bg": self.sprite_rembg.isChecked(),
            "cleanup_edge": self.sprite_clean.isChecked(),
            "edge_feather": self.sprite_feather.value(), 
            "edge_blur": self.sprite_blur.value(),
            "edge_gamma": self.sprite_gamma.value(),
            "remove_isolated": self.sprite_iso.isChecked(),
            "isolated_area": self.sprite_iso_area.value(),
            "remove_internal": self.sprite_internal.isChecked(),
            "internal_max_area": self.sprite_internal_area.value(),
        }
        
        logger.info(f"开始生成精灵图: {path}")
        
        out = Path(ConfigManager.get_output_path("sprite"))
        out.mkdir(parents=True, exist_ok=True)
        self.current_worker = SpriteWorker(path, str(out), params)
        self.current_worker.progress.connect(lambda v, m: [self.sprite_prog.setValue(v), self.sprite_prog.setFormat(m)])
        self.current_worker.error.connect(lambda e: logger.error(e))
        self.current_worker.finished.connect(lambda d: self.show_result_dialog(d['folder']))
        self.current_worker.start()

    def extract_run(self):
        path = self.extract_path_edit.text()
        if not path:
            logger.warning("请先选择视频文件")
            return
        
        params = {
            "extract_mode": "first_last" if self.extract_mode.checkedId()==0 else "all",
            "model_name": self.extract_model.get_current_model(),
            "num_threads": self.extract_threads.value(),
            "frame_step": max(1, self.extract_step.value()),
            "remove_bg": self.extract_rembg.isChecked(),
            "bg_type": self.extract_bg_type.currentText(),
            "bg_color": self.extract_bg_color.get_color(),  # 【修复】使用颜色选择器
            "bg_image": self.extract_bg_img.text(),
            "cleanup_edge": self.extract_clean.isChecked(),
            "edge_feather": self.extract_feather.value(), 
            "edge_blur": self.extract_blur.value(),
            "edge_gamma": self.extract_gamma.value(),
            "remove_isolated": self.extract_iso.isChecked(), 
            "isolated_area": self.extract_iso_area.value(),
            "remove_internal": self.extract_internal.isChecked(),
            "internal_max_area": self.extract_internal_area.value(),
        }
        
        logger.info(f"开始提取视频帧: {path}")
        
        out = Path(ConfigManager.get_output_path("extract"))
        self.current_worker = VideoToImagesWorker(path, str(out), params)
        self.current_worker.progress.connect(lambda v, m: [self.extract_prog.setValue(v), self.extract_prog.setFormat(m)])
        self.current_worker.error.connect(lambda e: logger.error(e))
        self.current_worker.finished.connect(lambda d: self.show_result_dialog(d['folder']))
        self.current_worker.start()

    def beiou_run(self):
        """视频扣像"""
        path = self.beiou_path_edit.text()
        if not path:
            logger.warning("请先选择视频文件")
            return
        
        format_text = self.beiou_format.currentText()
        if "mp4" in format_text:
            output_format = "mp4"
            ext = ".mp4"
        elif "avi" in format_text:
            output_format = "avi"
            ext = ".avi"
        else:
            output_format = "webm"
            ext = ".webm"
        
        params = {
            "model_name": self.beiou_model.get_current_model(),
            "output_format": output_format,
            "bg_color": self.beiou_bg_color.get_color(),  # 【修复】使用颜色选择器
            "cleanup_edge": self.beiou_clean.isChecked(),
            "edge_feather": self.beiou_feather.value(),
            "edge_blur": self.beiou_blur.value(),
            "edge_gamma": self.beiou_gamma.value(),
            "remove_isolated": self.beiou_iso.isChecked(),
            "isolated_area": self.beiou_iso_area.value(),
            "remove_internal": self.beiou_internal.isChecked(),
            "internal_max_area": self.beiou_internal_area.value(),
        }
        
        logger.info(f"开始视频扣像: {path}")
        
        out_dir = Path(ConfigManager.get_output_path("beiou"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(path).stem}_rembg_{datetime.now():%H%M%S}{ext}"
        
        self.current_worker = VideoRemoveBgWorker(path, str(out_path), params)
        self.current_worker.progress.connect(lambda v, m: [self.beiou_prog.setValue(v), self.beiou_prog.setFormat(m)])
        self.current_worker.error.connect(lambda e: logger.error(e))
        self.current_worker.finished.connect(lambda d: self.show_result_dialog(d['folder']))
        self.current_worker.start()

    def vid_run(self):
        path = self.vid_src_edit.text()
        if not path:
            logger.warning("请先选择图片文件夹")
            return
        
        params = {
            "fps": max(1, self.vid_fps.value()),
            "bg_type": self.vid_bg_type.currentText(),
            "bg_color": self.vid_bg_color.get_color()  # 【修复】使用颜色选择器
        }
        
        logger.info(f"开始合成视频: {path}")
        
        out = Path(ConfigManager.get_output_path("video")) / f"video_{datetime.now():%H%M%S}.mp4"
        out.parent.mkdir(exist_ok=True)
        self.current_worker = ImagesToVideoWorker(path, str(out), params)
        self.current_worker.progress.connect(lambda v, m: [self.vid_prog.setValue(v), self.vid_prog.setFormat(m)])
        self.current_worker.error.connect(lambda e: logger.error(e))
        self.current_worker.finished.connect(lambda d: self.show_result_dialog(d['folder']))
        self.current_worker.start()

    def gif_run(self):
        path = self.gif_src_edit.text()
        if not path:
            logger.warning("请先选择源文件")
            return
        
        params = {
            "model_name": self.gif_model.get_current_model(),
            "num_threads": self.gif_threads.value(),
            "fps": max(1, self.gif_fps.value()),
            "frame_step": max(1, self.gif_step.value()),
            "preserve_transparency": self.gif_transparency.isChecked(),
            "remove_bg": self.gif_rembg.isChecked(),
            "bg_type": self.gif_bg_type.currentText(),
            "bg_color": self.gif_bg_color.get_color(),  # 【修复】使用颜色选择器
            "cleanup_edge": self.gif_clean.isChecked(), 
            "edge_feather": self.gif_feather.value(), 
            "edge_blur": self.gif_blur.value(),
            "edge_gamma": self.gif_gamma.value(),
            "remove_isolated": self.gif_iso.isChecked(), 
            "isolated_area": self.gif_iso_area.value(),
            "remove_internal": self.gif_internal.isChecked(),
            "internal_max_area": self.gif_internal_area.value(),
        }
        
        logger.info(f"开始生成 GIF: {path}")
        
        out = Path(ConfigManager.get_output_path("gif")) / f"gif_{datetime.now():%H%M%S}.gif"
        out.parent.mkdir(exist_ok=True)
        
        if self.gif_src_type.checkedButton().text() == "视频":
            self.current_worker = VideoToGifWorker(path, str(out), params)
        else:
            self.current_worker = ImagesToGifWorker(path, str(out), params)
            
        self.current_worker.progress.connect(lambda v, m: [self.gif_prog.setValue(v), self.gif_prog.setFormat(m)])
        self.current_worker.error.connect(lambda e: logger.error(e))
        self.current_worker.finished.connect(lambda d: self.show_result_dialog(d['folder']))
        self.current_worker.start()

    def single_run(self):
        path = self.single_src_edit.text()
        if not path:
            logger.warning("请先选择图片文件")
            return
        
        params = {
            "model_name": self.single_model.get_current_model(),
            "cleanup_edge": self.single_clean.isChecked(),
            "edge_feather": self.single_feather.value(), 
            "edge_blur": self.single_blur.value(),
            "edge_gamma": self.single_gamma.value(),
            "remove_isolated": self.single_iso.isChecked(),
            "isolated_area": self.single_iso_area.value(),
            "remove_internal": self.single_internal.isChecked(),
            "internal_max_area": self.single_internal_area.value(),
            "bg_type": self.single_bg_type.currentText(),
            "bg_color": self.single_bg_color.get_color()  # 【修复】使用颜色选择器
        }
        
        logger.info(f"开始处理图片: {path}")
        
        out = Path(ConfigManager.get_output_path("single")) / f"proc_{Path(path).name}"
        out.parent.mkdir(exist_ok=True)
        self.current_worker = SingleImageWorker(path, str(out), params)
        self.current_worker.progress.connect(lambda v, m: [self.single_prog.setValue(v), self.single_prog.setFormat(m)])
        self.current_worker.error.connect(lambda e: logger.error(e))
        self.current_worker.finished.connect(lambda d: self.show_result_dialog(d['folder']))
        self.current_worker.start()

# ==================== 程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    if DependencyChecker.has_critical_missing():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("缺少必要依赖")
        msg.setText("程序缺少必要的依赖库，请先安装：")
        msg.setDetailedText(f"安装命令:\n{DependencyChecker.get_install_command()}\n\n完整安装:\n{DependencyChecker.get_full_install_command()}")
        msg.exec_()
        sys.exit(1)
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())