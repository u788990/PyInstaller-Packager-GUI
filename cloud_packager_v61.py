#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用云打包器 v6.3 - 完整图标支持版

修复内容：
1. 使用 .spec 文件代替超长命令行（解决 WinError 206）
2. 强制 UTF-8 输出（解决中文编码问题）
3. 完整图标支持：Tkinter / PyQt5 / PyQt6 / PySide6 / Pygame
4. 优化大型库（torch/onnxruntime/cv2）的处理
"""

# ==================== 强制 UTF-8 编码 ====================
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

if sys.platform == 'win32':
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass
    
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except:
        pass

import ast
import re
import time
import glob
import shutil
import tempfile
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional


def safe_print(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        ascii_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(ascii_msg, flush=True)


class UniversalCloudPackager:
    
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
    
    LARGE_PACKAGES = {
        'torch', 'torchvision', 'torchaudio', 'tensorflow', 'keras',
        'numpy', 'scipy', 'pandas', 'cv2', 'PIL', 'skimage',
        'onnxruntime', 'onnx', 'sklearn', 'xgboost', 'lightgbm',
        'matplotlib', 'plotly', 'seaborn', 'transformers', 'diffusers',
        'rembg', 'imageio', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'wx', 'kivy', 'psutil',
    }
    
    ALWAYS_EXCLUDE = [
        'numpy.array_api', 'numpy.distutils', 'numpy.f2py', 'numpy.testing',
        'scipy.testing', 'matplotlib.testing', 'matplotlib.tests',
        'torch.testing', 'torch.utils.benchmark',
        'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx',
        'setuptools', 'pip', 'wheel', 'twine', 'black', 'flake8', 'pylint',
        'mypy', 'isort', 'autopep8', 'coverage', 'tox',
    ]
    
    CORE_HIDDEN_IMPORTS = [
        'pkg_resources', 'pkg_resources.py2_warn', 
        'importlib_resources', 'importlib_metadata',
        'encodings.utf_8', 'encodings.gbk', 'encodings.cp1252',
        'encodings.ascii', 'encodings.latin_1', 'encodings.idna',
        'atexit', 'packaging', 'packaging.version', 'packaging.specifiers',
    ]
    
    def __init__(self, source: str, name: str, mode: str = 'onefile',
                 noconsole: bool = False, exe_icon: str = None,
                 window_icon: str = None, taskbar_icon: str = None,
                 extra_data: List[str] = None, cleanup_temp: bool = True):
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
        
        self._temp_files = []
        self._detected_imports: Set[str] = set()
        self._collect_packages: Set[str] = set()
        self._hidden_imports: Set[str] = set()
    
    def log(self, msg: str, level: str = "INFO"):
        prefix = {"INFO": "[Pack]", "WARN": "[WARN]", "ERROR": "[ERR]"}
        safe_print(f"{prefix.get(level, '[Pack]')} {msg}")
    
    def prepare_exe_icon(self) -> Optional[str]:
        if not self.exe_icon:
            return None
        
        icon_path = os.path.abspath(self.exe_icon)
        if not os.path.exists(icon_path):
            self.log(f"EXE icon not found: {icon_path}", "WARN")
            return None
        
        if icon_path.lower().endswith('.ico'):
            return icon_path
        
        if icon_path.lower().endswith('.png'):
            try:
                from PIL import Image
                img = Image.open(icon_path)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                ico_path = os.path.join(tempfile.gettempdir(), f"{self.name}_icon.ico")
                sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                img.save(ico_path, format='ICO', sizes=sizes)
                
                self._temp_files.append(ico_path)
                self.log(f"Created ICO from PNG")
                return ico_path
            except Exception as e:
                self.log(f"ICO conversion failed: {e}", "WARN")
        
        return None
    
    def create_icon_wrapper(self) -> Optional[str]:
        """创建包含所有 GUI 框架图标 Hook 的包装器"""
        encodings_to_try = ['utf-8', 'gbk', 'cp1252', 'latin-1']
        original_code = None
        
        for enc in encodings_to_try:
            try:
                with open(self.source, 'r', encoding=enc) as f:
                    original_code = f.read()
                break
            except:
                continue
        
        if original_code is None:
            self.log("Failed to read source file", "ERROR")
            return None
        
        window_icon_name = os.path.basename(self.window_icon) if self.window_icon else ''
        taskbar_icon_name = os.path.basename(self.taskbar_icon) if self.taskbar_icon else ''
        
        cleanup_code = ''
        if self.mode == 'onefile' and self.cleanup_temp:
            cleanup_code = '''
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
        
        # 完整的图标 Hook 代码
        icon_code = f'''
# ==================== Icon Setup v6.3 ====================
import sys
import os

_WINDOW_ICON = "{window_icon_name}"
_APP_NAME = "{self.name}"

def _get_resource_path(filename):
    if not filename:
        return None
    if hasattr(sys, '_MEIPASS'):
        path = os.path.join(sys._MEIPASS, filename)
        if os.path.exists(path):
            return path
    base = os.path.dirname(os.path.abspath(__file__))
    for candidate in [os.path.join(base, filename), os.path.join(os.getcwd(), filename), filename]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None

# Windows Taskbar AppUserModelID
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f'{{_APP_NAME}}.App.1.0')
    except: pass

# ==================== PyQt5 Hook ====================
try:
    from PyQt5 import QtWidgets, QtGui
    
    _orig_qapp = QtWidgets.QApplication.__init__
    def _new_qapp(self, *args, **kwargs):
        _orig_qapp(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except: pass
    QtWidgets.QApplication.__init__ = _new_qapp
    
    _orig_qmain = QtWidgets.QMainWindow.__init__
    def _new_qmain(self, *args, **kwargs):
        _orig_qmain(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except: pass
    QtWidgets.QMainWindow.__init__ = _new_qmain
    
    _orig_qwidget_show = QtWidgets.QWidget.show
    def _new_qwidget_show(self):
        try:
            if not self.windowIcon() or self.windowIcon().isNull():
                icon_path = _get_resource_path(_WINDOW_ICON)
                if icon_path and os.path.exists(icon_path):
                    self.setWindowIcon(QtGui.QIcon(icon_path))
        except: pass
        return _orig_qwidget_show(self)
    QtWidgets.QWidget.show = _new_qwidget_show
except ImportError:
    pass

# ==================== PyQt6 Hook ====================
try:
    from PyQt6 import QtWidgets as QtWidgets6, QtGui as QtGui6
    
    _orig_qapp6 = QtWidgets6.QApplication.__init__
    def _new_qapp6(self, *args, **kwargs):
        _orig_qapp6(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                self.setWindowIcon(QtGui6.QIcon(icon_path))
        except: pass
    QtWidgets6.QApplication.__init__ = _new_qapp6
except ImportError:
    pass

# ==================== PySide6 Hook ====================
try:
    from PySide6 import QtWidgets as PySide6Widgets, QtGui as PySide6Gui
    
    _orig_pyside6 = PySide6Widgets.QApplication.__init__
    def _new_pyside6(self, *args, **kwargs):
        _orig_pyside6(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                self.setWindowIcon(PySide6Gui.QIcon(icon_path))
        except: pass
    PySide6Widgets.QApplication.__init__ = _new_pyside6
except ImportError:
    pass

# ==================== Tkinter Hook ====================
try:
    import tkinter as tk
    
    _orig_tk = tk.Tk.__init__
    def _new_tk(self, *args, **kwargs):
        _orig_tk(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                if icon_path.lower().endswith('.png'):
                    photo = tk.PhotoImage(file=icon_path)
                    self.iconphoto(True, photo)
                    self._icon_ref = photo
                elif icon_path.lower().endswith('.ico'):
                    self.iconbitmap(icon_path)
        except: pass
    tk.Tk.__init__ = _new_tk
except ImportError:
    pass

# ==================== Pygame Hook ====================
try:
    import pygame
    
    _orig_pg_init = pygame.init
    def _new_pg_init(*args, **kwargs):
        result = _orig_pg_init(*args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                pygame.display.set_icon(pygame.image.load(icon_path))
        except: pass
        return result
    pygame.init = _new_pg_init
    
    _orig_set_mode = pygame.display.set_mode
    def _new_set_mode(*args, **kwargs):
        result = _orig_set_mode(*args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                pygame.display.set_icon(pygame.image.load(icon_path))
        except: pass
        return result
    pygame.display.set_mode = _new_set_mode
except ImportError:
    pass

# ==================== wxPython Hook ====================
try:
    import wx
    
    _orig_wxframe = wx.Frame.__init__
    def _new_wxframe(self, *args, **kwargs):
        _orig_wxframe(self, *args, **kwargs)
        try:
            icon_path = _get_resource_path(_WINDOW_ICON)
            if icon_path and os.path.exists(icon_path):
                self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_PNG))
        except: pass
    wx.Frame.__init__ = _new_wxframe
except ImportError:
    pass

# ==================== Original Code ====================
'''
        
        wrapper_code = cleanup_code + icon_code + '\n' + original_code
        
        wrapper_file = os.path.join(tempfile.gettempdir(), f"wrapper_{self.name}_{int(time.time())}.py")
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        self._temp_files.append(wrapper_file)
        self.log("Created wrapper with PyQt5/PyQt6/PySide6/Tkinter/Pygame/wx hooks")
        return wrapper_file
    
    def analyze_imports(self) -> Set[str]:
        imports = set()
        code = None
        
        for enc in ['utf-8', 'gbk', 'cp1252', 'latin-1']:
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
        except:
            pass
        
        for pat in [r'^import\s+([\w]+)', r'^from\s+([\w]+)']:
            for m in re.finditer(pat, code, re.MULTILINE):
                imports.add(m.group(1))
        
        return {imp for imp in imports if imp not in self.STDLIB}
    
    def prepare_dependencies(self) -> Tuple[Set[str], Set[str]]:
        imports = self.analyze_imports()
        self.log(f"Found {len(imports)} imports: {', '.join(sorted(imports)[:10])}...")
        
        collect = {'pkg_resources'}
        hidden = set(self.CORE_HIDDEN_IMPORTS)
        
        for imp in imports:
            if imp in self.LARGE_PACKAGES:
                collect.add(imp)
            else:
                hidden.add(imp)
        
        if 'PyQt5' in imports:
            collect.add('PyQt5')
            hidden.add('PyQt5.sip')
        
        self.log(f"Collect: {len(collect)}, Hidden: {len(hidden)}")
        return hidden, collect
    
    def collect_data_files(self) -> List[Tuple[str, str]]:
        data = []
        collected = set()
        
        # 关键：添加图标文件
        for icon in [self.window_icon, self.taskbar_icon]:
            if icon and os.path.exists(icon):
                abs_path = os.path.abspath(icon)
                if abs_path not in collected:
                    data.append((abs_path, '.'))
                    collected.add(abs_path)
                    self.log(f"Adding icon: {os.path.basename(icon)}")
        
        for ext in ['*.png', '*.jpg', '*.ico', '*.json', '*.yaml', '*.txt', '*.wav', '*.mp3', '*.ttf', '*.onnx', '*.pth']:
            for f in glob.glob(os.path.join(self.source_dir, ext)):
                if os.path.isfile(f):
                    abs_path = os.path.abspath(f)
                    if abs_path not in collected:
                        data.append((abs_path, '.'))
                        collected.add(abs_path)
        
        for d in ['assets', 'resources', 'data', 'models', 'images']:
            path = os.path.join(self.source_dir, d)
            if os.path.isdir(path):
                data.append((os.path.abspath(path), d))
        
        return data
    
    def generate_spec_file(self, source: str, hidden: Set[str], collect: Set[str],
                           data_files: List[Tuple[str, str]], ico_path: str) -> str:
        
        datas_list = [f"(r'{src.replace(chr(92), chr(92)*2)}', r'{dst}')" for src, dst in data_files]
        datas_str = ',\n        '.join(datas_list)
        
        hidden_list = sorted(list(hidden))[:200]
        hidden_str = ',\n        '.join([f"'{h}'" for h in hidden_list])
        
        collect_calls = ' + '.join([f"collect_submodules('{pkg}')" for pkg in sorted(collect)])
        hiddenimports_expr = f"[\n        {hidden_str}\n    ] + {collect_calls}" if collect_calls else f"[\n        {hidden_str}\n    ]"
        
        excludes_str = ',\n        '.join([f"'{e}'" for e in self.ALWAYS_EXCLUDE])
        icon_line = f"icon=r'{ico_path}'," if ico_path else ""
        
        if self.mode == 'onefile':
            exe_section = f"""
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],
    name='{self.name}', debug=False, strip=False, upx=False,
    runtime_tmpdir=None, console={not self.noconsole}, {icon_line})
"""
        else:
            exe_section = f"""
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
    name='{self.name}', debug=False, strip=False, upx=False,
    console={not self.noconsole}, {icon_line})
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='{self.name}')
"""
        
        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
block_cipher = None
a = Analysis([r'{source.replace(chr(92), chr(92)*2)}'], pathex=[], binaries=[], 
    datas=[{datas_str}],
    hiddenimports={hiddenimports_expr},
    hookspath=[], runtime_hooks=[],
    excludes=[{excludes_str}],
    cipher=block_cipher, noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
{exe_section}
"""
        
        spec_path = f"{self.name}.spec"
        with open(spec_path, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        self._temp_files.append(spec_path)
        self.log(f"Generated spec: {spec_path}")
        return spec_path
    
    def run(self) -> int:
        start = time.time()
        
        self.log("=" * 60)
        self.log("Universal Cloud Packager v6.3 (Full Icon Support)")
        self.log("=" * 60)
        self.log(f"Source: {repr(self.source)}")
        self.log(f"Output: {self.name}")
        self.log(f"Mode: {self.mode}")
        self.log(f"EXE Icon: {self.exe_icon or 'None'}")
        self.log(f"Window Icon: {self.window_icon or 'None'}")
        self.log(f"Taskbar Icon: {self.taskbar_icon or 'None'}")
        self.log("=" * 60)
        
        if not os.path.exists(self.source):
            self.log("Source not found!", "ERROR")
            return 1
        
        try:
            self.install_requirements()
            ico_path = self.prepare_exe_icon()
            
            wrapper_source = self.source
            if self.window_icon or self.taskbar_icon or (self.mode == 'onefile' and self.cleanup_temp):
                wrapper_result = self.create_icon_wrapper()
                if wrapper_result:
                    wrapper_source = wrapper_result
            
            hidden, collect = self.prepare_dependencies()
            data = self.collect_data_files()
            self.log(f"Data files: {len(data)}")
            
            spec_path = self.generate_spec_file(wrapper_source, hidden, collect, data, ico_path)
            
            import subprocess
            cmd = [self.python, '-m', 'PyInstaller', '--clean', '--noconfirm', spec_path]
            
            self.log("Running PyInstaller...")
            self.log("-" * 60)
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            
            for line in proc.stdout:
                try:
                    safe_print(f"[PYI] {line.decode('utf-8', errors='replace').rstrip()}")
                except:
                    pass
            
            proc.wait()
            elapsed = time.time() - start
            
            exe = Path('dist') / (f"{self.name}.exe" if self.mode == 'onefile' else f"{self.name}/{self.name}.exe")
            
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
                return 1
        
        except Exception as e:
            self.log(f"Exception: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return 1
        
        finally:
            for f in self._temp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except:
                    pass
    
    def install_requirements(self):
        import subprocess
        for req in ['requirements.txt', 'requirements-build.txt']:
            path = req if os.path.exists(req) else os.path.join(self.source_dir, req)
            if os.path.exists(path):
                self.log(f"Installing {req}...")
                subprocess.run([self.python, '-m', 'pip', 'install', '-r', path, '-q'], capture_output=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Universal Cloud Packager v6.3')
    parser.add_argument('--source', '-s', required=True)
    parser.add_argument('--name', '-n', required=True)
    parser.add_argument('--mode', '-m', choices=['onefile', 'onedir'], default='onefile')
    parser.add_argument('--noconsole', '-w', action='store_true')
    parser.add_argument('--exe-icon')
    parser.add_argument('--window-icon')
    parser.add_argument('--taskbar-icon')
    parser.add_argument('--data', '-d', action='append', default=[])
    parser.add_argument('--no-cleanup', action='store_true')
    
    args = parser.parse_args()
    
    packager = UniversalCloudPackager(
        source=args.source, name=args.name, mode=args.mode,
        noconsole=args.noconsole, exe_icon=args.exe_icon,
        window_icon=args.window_icon, taskbar_icon=args.taskbar_icon,
        extra_data=args.data, cleanup_temp=not args.no_cleanup
    )
    sys.exit(packager.run())


if __name__ == '__main__':
    main()
