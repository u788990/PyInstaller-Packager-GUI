#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用云打包器 v6.2 - 修复版

修复内容：
1. 使用 .spec 文件代替超长命令行（解决 WinError 206）
2. 强制 UTF-8 输出（解决中文编码问题）
3. 限制隐藏导入数量，使用 collect-submodules 代替
4. 优化大型库（torch/onnxruntime/cv2）的处理
"""

# ==================== 强制 UTF-8 编码（必须在最开始）====================
import sys
import os

# 设置环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 强制重新配置 stdout/stderr 为 UTF-8
if sys.platform == 'win32':
    try:
        # Windows: 尝试设置控制台代码页为 UTF-8
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass
    
    # 重新包装 stdout/stderr
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except:
        pass

# ==================== 其余导入 ====================
import ast
import re
import time
import glob
import shutil
import tempfile
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional


def safe_print(msg: str):
    """安全打印（处理编码问题）"""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # 如果还是失败，转换为 ASCII 安全格式
        ascii_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(ascii_msg, flush=True)


class UniversalCloudPackager:
    """
    通用云打包器 v6.2 - 使用 spec 文件
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
    
    # 大型库：使用 collect_submodules 而不是逐个添加
    LARGE_PACKAGES = {
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'keras',
        'numpy', 'scipy', 'pandas',
        'cv2', 'PIL', 'skimage',
        'onnxruntime', 'onnx',
        'sklearn', 'xgboost', 'lightgbm',
        'matplotlib', 'plotly', 'seaborn',
        'transformers', 'diffusers',
        'rembg', 'imageio',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'wx', 'kivy', 'psutil',
    }
    
    # 排除的模块
    ALWAYS_EXCLUDE = [
        'numpy.array_api', 'numpy.distutils', 'numpy.f2py', 'numpy.testing',
        'scipy.testing', 'matplotlib.testing', 'matplotlib.tests',
        'torch.testing', 'torch.utils.benchmark',
        'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx',
        'setuptools', 'pip', 'wheel', 'twine', 'black', 'flake8', 'pylint',
        'mypy', 'isort', 'autopep8', 'coverage', 'tox',
    ]
    
    # 核心隐藏导入（数量有限）
    CORE_HIDDEN_IMPORTS = [
        'pkg_resources', 'pkg_resources.py2_warn', 
        'importlib_resources', 'importlib_metadata',
        'encodings.utf_8', 'encodings.gbk', 'encodings.cp1252',
        'encodings.ascii', 'encodings.latin_1', 'encodings.idna',
        'atexit', 'packaging', 'packaging.version', 'packaging.specifiers',
    ]
    
    def __init__(self, source: str, name: str, mode: str = 'onefile',
                 noconsole: bool = False,
                 exe_icon: str = None,
                 window_icon: str = None,
                 taskbar_icon: str = None,
                 extra_data: List[str] = None,
                 cleanup_temp: bool = True):
        """初始化打包器"""
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
        
        # 临时文件列表
        self._temp_files = []
        
        # 检测结果
        self._detected_imports: Set[str] = set()
        self._collect_packages: Set[str] = set()
        self._hidden_imports: Set[str] = set()
    
    def log(self, msg: str, level: str = "INFO"):
        """输出日志（安全处理编码）"""
        prefix = {"INFO": "[Pack]", "WARN": "[WARN]", "ERROR": "[ERR]", "DEBUG": "[DBG]"}
        full_msg = f"{prefix.get(level, '[Pack]')} {msg}"
        safe_print(full_msg)
    
    # ==================== 图标处理 ====================
    
    def prepare_exe_icon(self) -> Optional[str]:
        """准备 EXE 图标（PNG 转 ICO）"""
        if not self.exe_icon:
            return None
        
        icon_path = os.path.abspath(self.exe_icon)
        if not os.path.exists(icon_path):
            self.log(f"EXE icon not found: {icon_path}", "WARN")
            return None
        
        if icon_path.lower().endswith('.ico'):
            self.log(f"Using ICO: {icon_path}")
            return icon_path
        
        if icon_path.lower().endswith('.png'):
            try:
                from PIL import Image
                self.log(f"Converting PNG to ICO")
                
                img = Image.open(icon_path)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                ico_path = os.path.join(tempfile.gettempdir(), f"{self.name}_icon.ico")
                sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                img.save(ico_path, format='ICO', sizes=sizes)
                
                self._temp_files.append(ico_path)
                self.log(f"Created ICO: {ico_path}")
                return ico_path
                
            except Exception as e:
                self.log(f"ICO conversion failed: {e}", "WARN")
                return None
        
        return None
    
    def create_icon_wrapper(self) -> Optional[str]:
        """创建图标设置包装器"""
        # 读取原始源代码
        encodings_to_try = ['utf-8', 'gbk', 'cp1252', 'latin-1']
        original_code = None
        
        for enc in encodings_to_try:
            try:
                with open(self.source, 'r', encoding=enc) as f:
                    original_code = f.read()
                self.log(f"Read source with encoding: {enc}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if original_code is None:
            self.log("Failed to read source file with any encoding", "ERROR")
            return None
        
        window_icon_name = os.path.basename(self.window_icon) if self.window_icon else ''
        taskbar_icon_name = os.path.basename(self.taskbar_icon) if self.taskbar_icon else ''
        
        # 清理代码
        cleanup_code = ''
        if self.mode == 'onefile' and self.cleanup_temp:
            cleanup_code = '''
# ==================== Temp cleanup ====================
import sys, os, atexit, shutil, time

def _cleanup_meipass():
    if hasattr(sys, '_MEIPASS'):
        try:
            time.sleep(0.3)
            shutil.rmtree(sys._MEIPASS, ignore_errors=True)
        except: pass

if hasattr(sys, '_MEIPASS'):
    atexit.register(_cleanup_meipass)
'''
        
        # 图标设置代码（简化版）
        icon_code = f'''
# ==================== Icon setup ====================
import sys, os

_WINDOW_ICON = "{window_icon_name}"

def _get_resource_path(filename):
    if not filename: return None
    if hasattr(sys, '_MEIPASS'):
        path = os.path.join(sys._MEIPASS, filename)
        if os.path.exists(path): return path
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, filename), filename]:
        if os.path.exists(p): return os.path.abspath(p)
    return None

# Windows taskbar
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('{self.name}.1.0')
    except: pass

# Tkinter Hook
try:
    import tkinter as tk
    _orig_tk = tk.Tk.__init__
    def _new_tk(self, *a, **kw):
        _orig_tk(self, *a, **kw)
        try:
            icon = _get_resource_path(_WINDOW_ICON)
            if icon and icon.endswith('.png'):
                photo = tk.PhotoImage(file=icon)
                self.iconphoto(True, photo)
                self._icon_ref = photo
        except: pass
    tk.Tk.__init__ = _new_tk
except: pass

# Pygame Hook
try:
    import pygame
    _orig_pg = pygame.init
    def _new_pg(*a, **kw):
        r = _orig_pg(*a, **kw)
        try:
            icon = _get_resource_path(_WINDOW_ICON)
            if icon: pygame.display.set_icon(pygame.image.load(icon))
        except: pass
        return r
    pygame.init = _new_pg
except: pass

# ==================== Original code ====================
'''
        
        wrapper_code = cleanup_code + icon_code + '\n' + original_code
        
        # 使用唯一文件名避免冲突（避免中文路径）
        wrapper_file = os.path.join(tempfile.gettempdir(), f"wrapper_{self.name}_{int(time.time())}.py")
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        self._temp_files.append(wrapper_file)
        self.log(f"Created wrapper: {wrapper_file}")
        return wrapper_file
    
    # ==================== 依赖检测 ====================
    
    def analyze_imports(self) -> Set[str]:
        """分析导入（简化版，避免过多子模块）"""
        imports = set()
        
        encodings_to_try = ['utf-8', 'gbk', 'cp1252', 'latin-1']
        code = None
        
        for enc in encodings_to_try:
            try:
                with open(self.source, 'r', encoding=enc) as f:
                    code = f.read()
                break
            except:
                continue
        
        if code is None:
            return imports
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except SyntaxError as e:
            self.log(f"Syntax error in source: {e}", "WARN")
        
        # 正则备份
        for pat in [r'^import\s+([\w]+)', r'^from\s+([\w]+)']:
            for m in re.finditer(pat, code, re.MULTILINE):
                imports.add(m.group(1))
        
        # 过滤标准库
        third_party = {imp for imp in imports if imp not in self.STDLIB}
        self._detected_imports = third_party
        
        return third_party
    
    def prepare_dependencies(self) -> Tuple[Set[str], Set[str]]:
        """准备依赖（使用 collect_submodules 策略）"""
        imports = self.analyze_imports()
        self.log(f"Found {len(imports)} imports: {', '.join(sorted(imports)[:15])}...")
        
        # 大型库使用 collect_submodules
        collect = set()
        hidden = set(self.CORE_HIDDEN_IMPORTS)
        
        for imp in imports:
            if imp in self.LARGE_PACKAGES:
                collect.add(imp)
                self.log(f"  [collect-submodules] {imp}")
            else:
                hidden.add(imp)
        
        # 添加常见依赖的子模块收集
        collect.add('pkg_resources')
        
        # 特殊处理
        if 'cv2' in imports:
            collect.add('cv2')
        if 'PIL' in imports:
            collect.add('PIL')
        if 'numpy' in imports:
            collect.add('numpy')
        if 'torch' in imports:
            collect.add('torch')
            hidden.add('torch._C')
        if 'onnxruntime' in imports:
            collect.add('onnxruntime')
        if 'rembg' in imports:
            collect.add('rembg')
            collect.add('pooch')  # rembg dependency
        
        self._collect_packages = collect
        self._hidden_imports = hidden
        
        self.log(f"Collect submodules: {len(collect)}")
        self.log(f"Hidden imports: {len(hidden)}")
        
        return hidden, collect
    
    # ==================== 数据文件 ====================
    
    def collect_data_files(self) -> List[Tuple[str, str]]:
        """收集数据文件"""
        data = []
        collected = set()
        
        # 图标
        for icon in [self.window_icon, self.taskbar_icon]:
            if icon and os.path.exists(icon):
                abs_path = os.path.abspath(icon)
                if abs_path not in collected:
                    data.append((abs_path, '.'))
                    collected.add(abs_path)
        
        # 常见资源
        extensions = ['*.png', '*.jpg', '*.ico', '*.json', '*.yaml', '*.txt',
                      '*.wav', '*.mp3', '*.ttf', '*.onnx', '*.pth']
        
        for ext in extensions:
            for f in glob.glob(os.path.join(self.source_dir, ext)):
                if os.path.isfile(f):
                    abs_path = os.path.abspath(f)
                    if abs_path not in collected:
                        data.append((abs_path, '.'))
                        collected.add(abs_path)
        
        # 资源目录
        for d in ['assets', 'resources', 'data', 'models', 'images']:
            path = os.path.join(self.source_dir, d)
            if os.path.isdir(path):
                data.append((os.path.abspath(path), d))
        
        return data
    
    # ==================== SPEC 文件生成 ====================
    
    def generate_spec_file(self, source: str, hidden: Set[str], collect: Set[str],
                           data_files: List[Tuple[str, str]], ico_path: str) -> str:
        """生成 .spec 文件（解决命令行过长问题）"""
        
        # 数据文件格式化
        datas_list = []
        for src, dst in data_files:
            # 转义反斜杠
            src_escaped = src.replace('\\', '\\\\')
            datas_list.append(f"(r'{src_escaped}', r'{dst}')")
        datas_str = ',\n        '.join(datas_list) if datas_list else ''
        
        # 隐藏导入（限制数量）
        hidden_list = sorted(list(hidden))[:200]  # 最多200个
        hidden_str = ',\n        '.join([f"'{h}'" for h in hidden_list])
        
        # collect_submodules 调用
        collect_calls = []
        for pkg in sorted(collect):
            collect_calls.append(f"collect_submodules('{pkg}')")
        
        # 构建 hiddenimports
        if collect_calls:
            hiddenimports_expr = f"""[
        {hidden_str}
    ] + {' + '.join(collect_calls)}"""
        else:
            hiddenimports_expr = f"""[
        {hidden_str}
    ]"""
        
        # 排除模块
        excludes_str = ',\n        '.join([f"'{e}'" for e in self.ALWAYS_EXCLUDE])
        
        # 图标
        icon_line = f"icon=r'{ico_path}'," if ico_path else ""
        
        # onefile vs onedir
        if self.mode == 'onefile':
            exe_section = f"""
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='{self.name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console={not self.noconsole},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_line}
)
"""
        else:
            exe_section = f"""
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{self.name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console={not self.noconsole},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_line}
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='{self.name}',
)
"""
        
        # 源文件路径
        source_escaped = source.replace('\\', '\\\\')
        
        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
# Auto-generated by Universal Cloud Packager v6.2

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

a = Analysis(
    [r'{source_escaped}'],
    pathex=[],
    binaries=[],
    datas=[
        {datas_str}
    ],
    hiddenimports={hiddenimports_expr},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        {excludes_str}
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

{exe_section}
"""
        
        # 写入 spec 文件（放在当前目录，避免中文路径问题）
        spec_path = f"{self.name}.spec"
        with open(spec_path, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        self._temp_files.append(spec_path)
        self.log(f"Generated spec file: {spec_path}")
        
        return spec_path
    
    # ==================== 执行打包 ====================
    
    def run(self) -> int:
        """执行打包"""
        start = time.time()
        
        self.log("=" * 60)
        self.log("Universal Cloud Packager v6.2 (Spec File Mode)")
        self.log("=" * 60)
        # 使用 repr 避免中文编码问题
        self.log(f"Source: {repr(self.source)}")
        self.log(f"Output: {self.name}")
        self.log(f"Mode: {self.mode}")
        self.log(f"No Console: {self.noconsole}")
        self.log(f"EXE Icon: {self.exe_icon or 'None'}")
        self.log(f"Window Icon: {self.window_icon or 'None'}")
        self.log("=" * 60)
        
        if not os.path.exists(self.source):
            self.log(f"Source not found!", "ERROR")
            return 1
        
        try:
            # 安装依赖
            self.install_requirements()
            
            # 准备图标
            ico_path = self.prepare_exe_icon()
            
            # 创建包装器
            wrapper_source = self.source
            if self.window_icon or self.taskbar_icon or (self.mode == 'onefile' and self.cleanup_temp):
                wrapper_result = self.create_icon_wrapper()
                if wrapper_result:
                    wrapper_source = wrapper_result
            
            # 检测依赖
            hidden, collect = self.prepare_dependencies()
            
            # 收集数据文件
            data = self.collect_data_files()
            self.log(f"Data files: {len(data)}")
            
            # 生成 spec 文件
            spec_path = self.generate_spec_file(wrapper_source, hidden, collect, data, ico_path)
            
            # 执行 PyInstaller（使用 spec 文件）
            import subprocess
            cmd = [self.python, '-m', 'PyInstaller', '--clean', '--noconfirm', spec_path]
            
            self.log(f"Running PyInstaller with spec file...")
            self.log("-" * 60)
            
            # 使用 UTF-8 编码运行
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            
            # 读取输出（处理编码）
            for line in proc.stdout:
                try:
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                except:
                    decoded = str(line)
                safe_print(f"[PYI] {decoded}")
            
            proc.wait()
            
            # 检查结果
            elapsed = time.time() - start
            
            if self.mode == 'onefile':
                exe = Path('dist') / f"{self.name}.exe"
            else:
                exe = Path('dist') / self.name / f"{self.name}.exe"
            
            self.log("-" * 60)
            
            if exe.exists():
                size = exe.stat().st_size / (1024 * 1024)
                self.log("=" * 60)
                self.log("SUCCESS!")
                self.log(f"Output: {exe}")
                self.log(f"Size: {size:.2f} MB")
                self.log(f"Time: {elapsed:.1f}s")
                self.log("=" * 60)
                return 0
            else:
                self.log("FAILED - Output not found", "ERROR")
                # 列出 dist 目录帮助调试
                if Path('dist').exists():
                    self.log("Contents of dist/:")
                    for item in Path('dist').rglob('*'):
                        self.log(f"  {item}")
                return 1
        
        except Exception as e:
            self.log(f"Exception: {e}", "ERROR")
            import traceback
            traceback.print_exc()
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
        import subprocess
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
        description='Universal Cloud Packager v6.2 (Spec File Mode)'
    )
    parser.add_argument('--source', '-s', required=True, help='Python source file')
    parser.add_argument('--name', '-n', required=True, help='Output name')
    parser.add_argument('--mode', '-m', choices=['onefile', 'onedir'], default='onefile')
    parser.add_argument('--noconsole', '-w', action='store_true', help='Hide console')
    parser.add_argument('--exe-icon', help='EXE icon (PNG/ICO)')
    parser.add_argument('--window-icon', help='Window icon (PNG)')
    parser.add_argument('--taskbar-icon', help='Taskbar icon (PNG)')
    parser.add_argument('--data', '-d', action='append', default=[], help='Extra data')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not cleanup temp')
    
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
