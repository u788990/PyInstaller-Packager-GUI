#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用云打包器 v6.1 - 完整版（含图标支持）

功能：
1. 自动检测任意项目的依赖（无需预设配置）
2. EXE 图标支持（PNG 自动转 ICO）
3. 窗口标题栏图标（Tkinter/PyQt/Pygame）
4. 任务栏图标（Windows AppUserModelID）
5. 自动收集资源文件
6. requirements.txt 自动安装

作者：基于 u788990@160.com 的项目改进
"""

import os
import sys
import ast
import re
import subprocess
import time
import glob
import json
import importlib
import pkgutil
import shutil
import tempfile
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional


class UniversalCloudPackager:
    """
    通用云打包器 v6.1 - 完整版
    
    特性：
    - 动态依赖检测（支持任意库）
    - 完整图标支持（EXE/窗口/任务栏）
    - 自动资源收集
    """
    
    # ==================== 标准库列表 ====================
    STDLIB = {
        'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 
        'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex',
        'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 
        'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections', 'colorsys',
        'compileall', 'concurrent', 'configparser', 'contextlib', 'contextvars',
        'copy', 'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes', 'curses',
        'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 
        'distutils', 'doctest', 'email', 'encodings', 'enum', 'errno',
        'faulthandler', 'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'fractions',
        'ftplib', 'functools', 'gc', 'getopt', 'getpass', 'gettext', 'glob',
        'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
        'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io',
        'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
        'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
        'mimetypes', 'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
        'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
        'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 
        'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
        'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc',
        'queue', 'quopri', 'random', 're', 'readline', 'reprlib', 'resource',
        'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
        'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib',
        'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat',
        'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'sunau',
        'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile',
        'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading',
        'time', 'timeit', 'tkinter', 'token', 'tokenize', 'tomllib', 'trace',
        'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types',
        'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv',
        'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
        'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport',
        'zlib', '_thread', '__future__', '__main__',
    }
    
    # import名 -> pip包名 映射
    IMPORT_TO_PIP = {
        'PIL': 'Pillow', 'cv2': 'opencv-python', 'sklearn': 'scikit-learn',
        'skimage': 'scikit-image', 'yaml': 'PyYAML', 'bs4': 'beautifulsoup4',
        'dateutil': 'python-dateutil', 'dotenv': 'python-dotenv',
        'jwt': 'PyJWT', 'serial': 'pyserial', 'wx': 'wxPython',
        'gi': 'PyGObject', 'cairo': 'pycairo', 'OpenGL': 'PyOpenGL',
        'usb': 'pyusb', 'Crypto': 'pycryptodome', 'win32api': 'pywin32',
        'win32com': 'pywin32', 'win32gui': 'pywin32', 'pywintypes': 'pywin32',
    }
    
    # 排除的模块
    ALWAYS_EXCLUDE = [
        'numpy.array_api', 'numpy.distutils', 'numpy.f2py', 'numpy.testing',
        'scipy.testing', 'matplotlib.testing', 'matplotlib.tests',
        'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx',
        'setuptools', 'pip', 'wheel', 'twine', 'black', 'flake8', 'pylint',
    ]
    
    # 始终包含的隐藏导入
    ALWAYS_INCLUDE = [
        'pkg_resources', 'pkg_resources.py2_warn', 'pkg_resources.markers',
        'pkg_resources._vendor', 'pkg_resources._vendor.jaraco',
        'pkg_resources._vendor.jaraco.text', 'pkg_resources._vendor.jaraco.functools',
        'pkg_resources._vendor.jaraco.context', 'pkg_resources.extern',
        'jaraco', 'jaraco.text', 'jaraco.functools', 'jaraco.context',
        'importlib_resources', 'importlib_metadata',
        'encodings.utf_8', 'encodings.gbk', 'encodings.cp1252',
        'encodings.ascii', 'encodings.latin_1', 'encodings.idna',
        'atexit',  # 用于临时文件清理
    ]
    
    def __init__(self, source: str, name: str, mode: str = 'onefile',
                 noconsole: bool = False,
                 exe_icon: str = None,
                 window_icon: str = None,
                 taskbar_icon: str = None,
                 extra_data: List[str] = None,
                 cleanup_temp: bool = True):
        """
        初始化打包器
        
        Args:
            source: Python 源文件路径
            name: 输出 EXE 名称
            mode: 'onefile' 或 'onedir'
            noconsole: 是否隐藏控制台
            exe_icon: EXE 图标（PNG/ICO）
            window_icon: 窗口图标（PNG）
            taskbar_icon: 任务栏图标（PNG）
            extra_data: 额外的数据文件/目录
            cleanup_temp: 是否清理临时文件夹（onefile 模式）
        """
        self.source = source
        self.name = name
        self.mode = mode
        self.noconsole = noconsole
        self.exe_icon = exe_icon
        self.window_icon = window_icon
        self.taskbar_icon = taskbar_icon
        self.extra_data = extra_data or []
        self.cleanup_temp = cleanup_temp
        
        self.python = sys.executable
        self.source_dir = os.path.dirname(os.path.abspath(source)) or '.'
        
        # 临时文件列表（打包完成后清理）
        self._temp_files = []
        
        # 检测结果缓存
        self._detected_imports: Set[str] = set()
        self._hidden_imports: Set[str] = set()
        self._collect_packages: Set[str] = set()
        
        self._setup_encoding()
    
    def _setup_encoding(self):
        """设置 UTF-8 编码"""
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
    
    def log(self, msg: str, level: str = "INFO"):
        """输出日志"""
        prefix = {"INFO": "[Pack]", "WARN": "[WARN]", "ERROR": "[ERR]", "DEBUG": "[DBG]"}
        print(f"{prefix.get(level, '[Pack]')} {msg}")
    
    # ==================== 图标处理 ====================
    
    def prepare_exe_icon(self) -> Optional[str]:
        """
        准备 EXE 图标
        - 如果是 PNG，转换为 ICO（多尺寸）
        - 如果是 ICO，直接使用
        返回 ICO 文件路径
        """
        if not self.exe_icon:
            return None
        
        icon_path = os.path.abspath(self.exe_icon)
        if not os.path.exists(icon_path):
            self.log(f"EXE icon not found: {icon_path}", "WARN")
            return None
        
        # 如果已经是 ICO，直接返回
        if icon_path.lower().endswith('.ico'):
            self.log(f"Using ICO: {icon_path}")
            return icon_path
        
        # PNG 转 ICO
        if icon_path.lower().endswith('.png'):
            try:
                from PIL import Image
                
                self.log(f"Converting PNG to ICO: {icon_path}")
                
                img = Image.open(icon_path)
                
                # 确保是 RGBA
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # 生成多尺寸 ICO
                ico_path = os.path.join(tempfile.gettempdir(), f"{self.name}_icon.ico")
                sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                
                img.save(ico_path, format='ICO', sizes=sizes)
                
                self._temp_files.append(ico_path)
                self.log(f"Created ICO: {ico_path}")
                return ico_path
                
            except ImportError:
                self.log("Pillow not installed, cannot convert PNG to ICO", "WARN")
                return None
            except Exception as e:
                self.log(f"ICO conversion failed: {e}", "WARN")
                return None
        
        self.log(f"Unsupported icon format: {icon_path}", "WARN")
        return None
    
    def create_icon_wrapper(self) -> Optional[str]:
        """
        创建图标设置包装器代码
        
        功能：
        1. 设置 Tkinter 窗口图标
        2. 设置 PyQt/PySide 窗口图标
        3. 设置 Pygame 窗口图标
        4. 设置 Windows 任务栏图标（AppUserModelID）
        5. 清理临时文件夹（onefile 模式）
        
        返回包装器文件路径
        """
        # 读取原始源代码
        try:
            with open(self.source, 'r', encoding='utf-8') as f:
                original_code = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.source, 'r', encoding='gbk') as f:
                    original_code = f.read()
            except:
                with open(self.source, 'r', encoding='latin-1') as f:
                    original_code = f.read()
        
        # 图标文件名（打包后的相对路径）
        window_icon_name = os.path.basename(self.window_icon) if self.window_icon else ''
        taskbar_icon_name = os.path.basename(self.taskbar_icon) if self.taskbar_icon else ''
        
        # 是否需要清理代码
        cleanup_code = ''
        if self.mode == 'onefile' and self.cleanup_temp:
            cleanup_code = '''
# ==================== 临时文件夹清理（onefile 模式）====================
import sys
import os
import atexit
import shutil
import time

def _cleanup_meipass():
    """程序退出时清理 PyInstaller 临时文件夹"""
    if hasattr(sys, '_MEIPASS'):
        meipass = sys._MEIPASS
        try:
            time.sleep(0.3)  # 等待资源释放
            if os.path.exists(meipass):
                shutil.rmtree(meipass, ignore_errors=True)
        except:
            pass

if hasattr(sys, '_MEIPASS'):
    atexit.register(_cleanup_meipass)
'''
        
        # 图标设置代码
        icon_code = f'''
# ==================== 图标设置代码（自动生成）====================
import sys
import os

# 图标文件名
_WINDOW_ICON = "{window_icon_name}"
_TASKBAR_ICON = "{taskbar_icon_name}"

def _get_resource_path(filename):
    """获取资源文件的绝对路径"""
    if not filename:
        return None
    
    # PyInstaller 打包后的路径
    if hasattr(sys, '_MEIPASS'):
        path = os.path.join(sys._MEIPASS, filename)
        if os.path.exists(path):
            return path
    
    # 开发环境路径
    base = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(base, filename),
        os.path.join(os.getcwd(), filename),
        filename,
    ]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    
    return None

def _setup_windows_taskbar():
    """设置 Windows 任务栏图标（AppUserModelID）"""
    if sys.platform != 'win32':
        return
    
    try:
        import ctypes
        # 设置 AppUserModelID，使任务栏图标独立显示
        app_id = f'{{__name__}}.{{"{self.name}"}}.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except:
        pass

# 在导入时就设置任务栏
_setup_windows_taskbar()

# ==================== Tkinter 图标 Hook ====================
try:
    import tkinter as tk
    
    _original_tk_init = tk.Tk.__init__
    
    def _patched_tk_init(self, *args, **kwargs):
        _original_tk_init(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path:
                if icon_path.lower().endswith('.png'):
                    photo = tk.PhotoImage(file=icon_path)
                    self.iconphoto(True, photo)
                    # 保持引用防止被垃圾回收
                    if not hasattr(self, '_icon_refs'):
                        self._icon_refs = []
                    self._icon_refs.append(photo)
                elif icon_path.lower().endswith('.ico'):
                    self.iconbitmap(icon_path)
        except Exception as e:
            pass  # 静默失败
    
    tk.Tk.__init__ = _patched_tk_init
    
    # Toplevel 也设置图标
    _original_toplevel_init = tk.Toplevel.__init__
    
    def _patched_toplevel_init(self, *args, **kwargs):
        _original_toplevel_init(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and icon_path.lower().endswith('.png'):
                photo = tk.PhotoImage(file=icon_path)
                self.iconphoto(True, photo)
                if not hasattr(self, '_icon_refs'):
                    self._icon_refs = []
                self._icon_refs.append(photo)
        except:
            pass
    
    tk.Toplevel.__init__ = _patched_toplevel_init

except ImportError:
    pass

# ==================== PyQt5 图标 Hook ====================
try:
    from PyQt5 import QtWidgets, QtGui
    
    _original_qapp_init = QtWidgets.QApplication.__init__
    
    def _patched_qapp_init(self, *args, **kwargs):
        _original_qapp_init(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path:
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except:
            pass
    
    QtWidgets.QApplication.__init__ = _patched_qapp_init

except ImportError:
    pass

# ==================== PyQt6 图标 Hook ====================
try:
    from PyQt6 import QtWidgets, QtGui
    
    _original_qapp6_init = QtWidgets.QApplication.__init__
    
    def _patched_qapp6_init(self, *args, **kwargs):
        _original_qapp6_init(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path:
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except:
            pass
    
    QtWidgets.QApplication.__init__ = _patched_qapp6_init

except ImportError:
    pass

# ==================== PySide6 图标 Hook ====================
try:
    from PySide6 import QtWidgets, QtGui
    
    _original_pyside_init = QtWidgets.QApplication.__init__
    
    def _patched_pyside_init(self, *args, **kwargs):
        _original_pyside_init(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path:
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except:
            pass
    
    QtWidgets.QApplication.__init__ = _patched_pyside_init

except ImportError:
    pass

# ==================== Pygame 图标 Hook ====================
try:
    import pygame
    
    _original_pygame_init = pygame.init
    
    def _patched_pygame_init(*args, **kwargs):
        result = _original_pygame_init(*args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path:
                icon_surface = pygame.image.load(icon_path)
                pygame.display.set_icon(icon_surface)
        except:
            pass
        return result
    
    pygame.init = _patched_pygame_init

except ImportError:
    pass

# ==================== 原始代码开始 ====================
'''
        
        # 组合代码
        wrapper_code = cleanup_code + icon_code + '\n' + original_code
        
        # 写入临时文件
        wrapper_file = tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', suffix='.py', delete=False
        )
        wrapper_file.write(wrapper_code)
        wrapper_file.close()
        
        self._temp_files.append(wrapper_file.name)
        self.log(f"Created wrapper: {wrapper_file.name}")
        
        return wrapper_file.name
    
    # ==================== 依赖检测 ====================
    
    def analyze_source_imports(self) -> Set[str]:
        """分析源文件导入"""
        imports = set()
        
        try:
            with open(self.source, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            with open(self.source, 'r', encoding='gbk', errors='replace') as f:
                code = f.read()
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except SyntaxError as e:
            self.log(f"Syntax error: {e}", "WARN")
        
        # 正则备份
        for pat in [r'^import\s+([\w\.]+)', r'^from\s+([\w\.]+)\s+import']:
            for m in re.finditer(pat, code, re.MULTILINE):
                imports.add(m.group(1))
        
        # 过滤标准库，提取顶级模块
        third_party = set()
        for imp in imports:
            top = imp.split('.')[0]
            if top not in self.STDLIB:
                third_party.add(top)
        
        self._detected_imports = third_party
        return third_party
    
    def get_pip_dependencies(self, package: str) -> Set[str]:
        """获取包的 pip 依赖"""
        deps = set()
        pip_name = self.IMPORT_TO_PIP.get(package, package)
        
        try:
            result = subprocess.run(
                [self.python, '-m', 'pip', 'show', pip_name],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Requires:'):
                        requires = line.split(':', 1)[1].strip()
                        if requires:
                            for dep in requires.split(','):
                                dep = dep.strip().lower().replace('-', '_')
                                if dep:
                                    deps.add(dep)
        except:
            pass
        
        return deps
    
    def get_package_submodules(self, package: str) -> Set[str]:
        """获取包的所有子模块"""
        submodules = {package}
        
        try:
            mod = importlib.import_module(package)
            if hasattr(mod, '__path__'):
                for importer, modname, ispkg in pkgutil.walk_packages(
                    mod.__path__, prefix=f"{package}."
                ):
                    if 'test' not in modname.lower() and modname.count('.') <= 3:
                        submodules.add(modname)
        except:
            pass
        
        return submodules
    
    def detect_all_dependencies(self) -> Tuple[Set[str], Set[str]]:
        """检测所有依赖"""
        self.log("Analyzing dependencies...")
        
        # 分析导入
        imports = self.analyze_source_imports()
        self.log(f"Found {len(imports)} imports: {', '.join(sorted(imports)[:10])}...")
        
        # 追踪依赖链
        all_packages = set(imports)
        processed = set()
        queue = list(imports)
        
        while queue:
            pkg = queue.pop(0)
            if pkg in processed:
                continue
            processed.add(pkg)
            
            deps = self.get_pip_dependencies(pkg)
            for dep in deps:
                if dep not in processed and dep not in self.STDLIB:
                    all_packages.add(dep)
                    queue.append(dep)
        
        self.log(f"Total packages: {len(all_packages)}")
        
        # 枚举子模块
        hidden = set(self.ALWAYS_INCLUDE)
        collect = {'pkg_resources', 'jaraco'}
        
        for pkg in all_packages:
            subs = self.get_package_submodules(pkg)
            for sub in subs:
                if not any(sub.startswith(e) for e in self.ALWAYS_EXCLUDE):
                    hidden.add(sub)
            
            # 需要 collect-all 的包
            if pkg in ['certifi', 'charset_normalizer', 'cv2', 'onnxruntime']:
                collect.add(pkg)
        
        self._hidden_imports = hidden
        self._collect_packages = collect
        
        self.log(f"Hidden imports: {len(hidden)}")
        return hidden, collect
    
    # ==================== 数据文件收集 ====================
    
    def collect_data_files(self) -> List[Tuple[str, str]]:
        """收集数据文件"""
        data = []
        
        # 图标文件
        for icon in [self.window_icon, self.taskbar_icon]:
            if icon and os.path.exists(icon):
                data.append((os.path.abspath(icon), '.'))
                self.log(f"Adding icon: {os.path.basename(icon)}")
        
        # 资源文件
        extensions = [
            '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.bmp', '*.webp',
            '*.json', '*.yaml', '*.yml', '*.toml', '*.cfg', '*.ini',
            '*.txt', '*.csv', '*.xml', '*.md',
            '*.wav', '*.mp3', '*.ogg', '*.flac',
            '*.ttf', '*.otf',
            '*.onnx', '*.pb', '*.pth', '*.pt', '*.h5', '*.pkl',
        ]
        
        for ext in extensions:
            for f in glob.glob(os.path.join(self.source_dir, ext)):
                if os.path.isfile(f):
                    data.append((f, '.'))
            for f in glob.glob(os.path.join(self.source_dir, '*', ext)):
                if os.path.isfile(f):
                    rel = os.path.relpath(os.path.dirname(f), self.source_dir)
                    data.append((f, rel))
        
        # 资源目录
        for d in ['assets', 'resources', 'data', 'models', 'images', 'icons', 'fonts', 'sounds']:
            path = os.path.join(self.source_dir, d)
            if os.path.isdir(path):
                data.append((path, d))
                self.log(f"Adding directory: {d}/")
        
        # 额外文件
        for item in self.extra_data:
            if os.path.exists(item):
                if os.path.isdir(item):
                    data.append((item, os.path.basename(item)))
                else:
                    data.append((item, '.'))
        
        return data
    
    # ==================== 构建和执行 ====================
    
    def build_command(self, source: str, hidden: Set[str], collect: Set[str],
                      data_files: List[Tuple[str, str]], ico_path: str) -> List[str]:
        """构建 PyInstaller 命令"""
        cmd = [
            self.python, '-m', 'PyInstaller',
            '--clean', '--noconfirm',
            f"--{'onefile' if self.mode == 'onefile' else 'onedir'}",
            '--name', self.name,
        ]
        
        if self.noconsole:
            cmd.append('--noconsole')
        
        # EXE 图标
        if ico_path:
            cmd.extend(['--icon', ico_path])
        
        # 隐藏导入
        for h in sorted(hidden):
            cmd.extend(['--hidden-import', h])
        
        # collect-all
        for c in sorted(collect):
            cmd.extend(['--collect-all', c])
        
        # 数据文件
        sep = ';' if sys.platform == 'win32' else ':'
        for src, dst in data_files:
            cmd.extend(['--add-data', f"{os.path.abspath(src)}{sep}{dst}"])
        
        # 排除
        for e in self.ALWAYS_EXCLUDE:
            cmd.extend(['--exclude-module', e])
        
        cmd.append(source)
        return cmd
    
    def run(self) -> int:
        """执行打包"""
        start = time.time()
        
        self.log("=" * 60)
        self.log("Universal Cloud Packager v6.1 (with Icons)")
        self.log("=" * 60)
        self.log(f"Source: {self.source}")
        self.log(f"Output: {self.name}")
        self.log(f"Mode: {self.mode}")
        self.log(f"EXE Icon: {self.exe_icon or 'None'}")
        self.log(f"Window Icon: {self.window_icon or 'None'}")
        self.log(f"Taskbar Icon: {self.taskbar_icon or 'None'}")
        self.log("=" * 60)
        
        if not os.path.exists(self.source):
            self.log(f"Source not found: {self.source}", "ERROR")
            return 1
        
        try:
            # 安装依赖
            self.install_requirements()
            
            # 准备 EXE 图标
            ico_path = self.prepare_exe_icon()
            
            # 创建包装器（设置窗口/任务栏图标）
            wrapper_source = self.source
            if self.window_icon or self.taskbar_icon or (self.mode == 'onefile' and self.cleanup_temp):
                wrapper_source = self.create_icon_wrapper()
            
            # 检测依赖
            hidden, collect = self.detect_all_dependencies()
            
            # 收集数据
            data = self.collect_data_files()
            self.log(f"Data files: {len(data)}")
            
            # 构建命令
            cmd = self.build_command(wrapper_source, hidden, collect, data, ico_path)
            self.log(f"Command args: {len(cmd)}")
            
            # 执行
            self.log("-" * 60)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='replace'
            )
            
            for line in proc.stdout:
                print(f"[PYI] {line.rstrip()}")
            
            proc.wait()
            
            # 检查结果
            elapsed = time.time() - start
            
            if self.mode == 'onefile':
                exe = Path('dist') / f"{self.name}.exe"
            else:
                exe = Path('dist') / self.name / f"{self.name}.exe"
            
            self.log("-" * 60)
            
            if exe.exists():
                size = exe.stat().st_size / (1024*1024)
                self.log("=" * 60)
                self.log("SUCCESS!")
                self.log(f"Output: {exe}")
                self.log(f"Size: {size:.2f} MB")
                self.log(f"Time: {elapsed:.1f}s")
                self.log("=" * 60)
                return 0
            else:
                self.log("FAILED - Output not found", "ERROR")
                return 1
        
        finally:
            # 清理临时文件
            for f in self._temp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except:
                    pass
    
    def install_requirements(self):
        """安装依赖"""
        for req in ['requirements.txt', 'requirements-build.txt']:
            path = os.path.join(self.source_dir, req)
            if not os.path.exists(path):
                path = req
            if os.path.exists(path):
                self.log(f"Installing {req}...")
                subprocess.run(
                    [self.python, '-m', 'pip', 'install', '-r', path, '-q'],
                    capture_output=True
                )


# ==================== 命令行入口 ====================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Universal Cloud Packager v6.1 - Full Icon Support'
    )
    parser.add_argument('--source', '-s', required=True, help='Python source file')
    parser.add_argument('--name', '-n', required=True, help='Output name')
    parser.add_argument('--mode', '-m', choices=['onefile', 'onedir'], default='onefile')
    parser.add_argument('--noconsole', '-w', action='store_true', help='Hide console')
    parser.add_argument('--exe-icon', help='EXE icon (PNG/ICO, e.g. 480x480.png)')
    parser.add_argument('--window-icon', help='Window icon (PNG, e.g. 28x28.png)')
    parser.add_argument('--taskbar-icon', help='Taskbar icon (PNG, e.g. 108x108.png)')
    parser.add_argument('--data', '-d', action='append', default=[], help='Extra data')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not cleanup temp folder')
    
    args = parser.parse_args()
    
    packager = UniversalCloudPackager(
        source=args.source,
        name=args.name,
        mode=args.mode,
        noconsole=args.noconsole,
        exe_icon=args.exe_icon,
        window_icon=args.window_icon,
        taskbar_icon=args.taskbar_icon,
        extra_data=args.data,
        cleanup_temp=not args.no_cleanup,
    )
    
    sys.exit(packager.run())


if __name__ == '__main__':
    main()
