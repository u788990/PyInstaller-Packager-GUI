"""
云打包工具 v61 - 修复版
支持自动检测和处理特殊库（如rembg）的打包需求
"""

import os
import sys
import yaml
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# ==================== 特殊库配置 ====================
# 定义需要特殊处理的库及其打包参数
SPECIAL_LIBRARIES = {
    'rembg': {
        'collect_all': ['rembg', 'onnxruntime'],
        'hidden_imports': [
            'rembg',
            'rembg.bg',
            'rembg.session_factory',
            'rembg.sessions',
            'rembg.sessions.u2net',
            'rembg.sessions.u2netp',
            'rembg.sessions.u2net_human_seg',
            'rembg.sessions.u2net_cloth_seg',
            'rembg.sessions.silueta',
            'rembg.sessions.isnet_general_use',
            'rembg.sessions.sam',
            'onnxruntime',
            'onnxruntime.capi',
            'onnxruntime.capi._pybind_state',
            'onnxruntime.capi.onnxruntime_pybind11_state',
        ],
        'description': 'AI图像背景移除库'
    },
    'paddleocr': {
        'collect_all': ['paddleocr', 'paddle'],
        'hidden_imports': [
            'paddleocr',
            'paddle',
            'paddle.vision',
        ],
        'description': 'OCR文字识别库'
    },
    'transformers': {
        'collect_all': ['transformers'],
        'hidden_imports': [
            'transformers',
            'transformers.models',
        ],
        'description': 'Hugging Face模型库'
    },
    'torch': {
        'collect_all': ['torch', 'torchvision'],
        'hidden_imports': [
            'torch',
            'torch._C',
            'torch._VF',
            'torchvision',
        ],
        'description': 'PyTorch深度学习框架'
    },
    'tensorflow': {
        'collect_all': ['tensorflow'],
        'hidden_imports': [
            'tensorflow',
            'tensorflow.python',
        ],
        'description': 'TensorFlow深度学习框架'
    },
}


# ==================== 辅助函数 ====================

def detect_special_libraries(script_path: str) -> List[str]:
    """
    检测脚本中使用的特殊库
    
    Args:
        script_path: Python脚本路径
        
    Returns:
        检测到的特殊库名称列表
    """
    detected = []
    
    if not os.path.exists(script_path):
        print(f"[WARNING] 脚本文件不存在: {script_path}")
        return detected
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for lib_name in SPECIAL_LIBRARIES.keys():
            # 检测 import 语句
            if f'import {lib_name}' in content or f'from {lib_name}' in content:
                detected.append(lib_name)
                lib_desc = SPECIAL_LIBRARIES[lib_name].get('description', '')
                print(f"[INFO] 检测到特殊库: {lib_name} ({lib_desc})")
                
    except Exception as e:
        print(f"[WARNING] 检测库时出错: {e}")
        
    return detected


def add_special_library_options(cmd: List[str], libraries: List[str]) -> List[str]:
    """
    为特殊库添加打包选项
    
    Args:
        cmd: PyInstaller命令列表
        libraries: 检测到的特殊库列表
        
    Returns:
        添加了特殊选项的命令列表
    """
    if not libraries:
        return cmd
    
    print(f"\n[INFO] 为 {len(libraries)} 个特殊库添加打包参数...")
    
    for lib in libraries:
        config = SPECIAL_LIBRARIES.get(lib, {})
        
        # 添加 --collect-all 参数
        for module in config.get('collect_all', []):
            print(f"  └─ --collect-all {module}")
            cmd.extend(['--collect-all', module])
        
        # 添加 --hidden-import 参数
        for module in config.get('hidden_imports', []):
            print(f"  └─ --hidden-import {module}")
            cmd.extend(['--hidden-import', module])
    
    print()
    return cmd


def load_config(config_file: str = 'build-v61.yml') -> Dict:
    """
    加载YAML配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[INFO] 成功加载配置文件: {config_file}")
        return config
    except FileNotFoundError:
        print(f"[ERROR] 配置文件不存在: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] 配置文件格式错误: {e}")
        sys.exit(1)


def build_pyinstaller_command(config: Dict, detected_libs: List[str]) -> List[str]:
    """
    构建PyInstaller打包命令
    
    Args:
        config: 配置字典
        detected_libs: 检测到的特殊库列表
        
    Returns:
        完整的命令列表
    """
    cmd = ['pyinstaller']
    
    # 基础选项
    script = config.get('script')
    if not script:
        print("[ERROR] 配置文件中未指定script")
        sys.exit(1)
    
    # 清理旧的构建文件
    cmd.append('--clean')
    cmd.append('--noconfirm')
    
    # 从配置文件读取选项
    options = config.get('pyinstaller_options', {})
    
    # 单文件模式
    if options.get('onefile', False):
        cmd.append('-F')
    else:
        cmd.append('-D')
    
    # 无控制台窗口
    if options.get('noconsole', False):
        cmd.append('-w')
    
    # 图标
    icon = options.get('icon')
    if icon and os.path.exists(icon):
        cmd.extend(['--icon', icon])
    
    # 输出名称
    name = options.get('name')
    if name:
        cmd.extend(['--name', name])
    
    # 添加数据文件
    datas = options.get('datas', [])
    for data in datas:
        if isinstance(data, dict):
            src = data.get('src')
            dst = data.get('dst', '.')
            if src and os.path.exists(src):
                separator = ';' if sys.platform == 'win32' else ':'
                cmd.extend(['--add-data', f'{src}{separator}{dst}'])
    
    # 添加二进制文件
    binaries = options.get('binaries', [])
    for binary in binaries:
        if isinstance(binary, dict):
            src = binary.get('src')
            dst = binary.get('dst', '.')
            if src and os.path.exists(src):
                separator = ';' if sys.platform == 'win32' else ':'
                cmd.extend(['--add-binary', f'{src}{separator}{dst}'])
    
    # 配置文件中的 collect-all
    collect_all_list = options.get('collect_all', [])
    for module in collect_all_list:
        cmd.extend(['--collect-all', module])
    
    # 配置文件中的 hidden-imports
    hidden_imports = options.get('hidden_imports', [])
    for module in hidden_imports:
        cmd.extend(['--hidden-import', module])
    
    # 排除的模块
    excludes = options.get('excludes', [])
    for module in excludes:
        cmd.extend(['--exclude-module', module])
    
    # UPX压缩
    if options.get('upx', False):
        cmd.append('--upx-dir')
        cmd.append(options.get('upx_dir', 'upx'))
    
    # 添加hooks目录
    hooks_dir = options.get('hooks_dir')
    if hooks_dir and os.path.exists(hooks_dir):
        cmd.extend(['--additional-hooks-dir', hooks_dir])
    
    # === 关键：添加检测到的特殊库的参数 ===
    cmd = add_special_library_options(cmd, detected_libs)
    
    # 最后添加脚本文件
    cmd.append(script)
    
    return cmd


def clean_build_folders():
    """清理构建文件夹"""
    folders_to_clean = ['build', 'dist', '__pycache__']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"[INFO] 已清理: {folder}")
            except Exception as e:
                print(f"[WARNING] 清理失败 {folder}: {e}")


def run_packaging(cmd: List[str]) -> bool:
    """
    执行打包命令
    
    Args:
        cmd: 命令列表
        
    Returns:
        是否成功
    """
    print("\n" + "="*60)
    print("开始打包...")
    print("="*60)
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        # 实时输出
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ 打包成功！")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print(f"✗ 打包失败！退出代码: {result.returncode}")
            print("="*60)
            return False
            
    except Exception as e:
        print(f"\n[ERROR] 打包过程出错: {e}")
        return False


def check_dependencies():
    """检查必要的依赖"""
    try:
        import PyInstaller
        print(f"[INFO] PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("[ERROR] PyInstaller 未安装")
        print("请运行: pip install pyinstaller")
        sys.exit(1)
    
    try:
        import yaml
        print(f"[INFO] PyYAML 已安装")
    except ImportError:
        print("[ERROR] PyYAML 未安装")
        print("请运行: pip install pyyaml")
        sys.exit(1)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("云打包工具 v61 - 修复版")
    print("支持自动检测 rembg 等特殊库")
    print("="*60 + "\n")
    
    # 检查依赖
    check_dependencies()
    
    # 加载配置
    config = load_config('build-v61.yml')
    
    script_path = config.get('script')
    if not script_path or not os.path.exists(script_path):
        print(f"[ERROR] 脚本文件不存在: {script_path}")
        sys.exit(1)
    
    print(f"[INFO] 目标脚本: {script_path}\n")
    
    # 检测特殊库
    print("[INFO] 扫描脚本中的特殊库...")
    detected_libs = detect_special_libraries(script_path)
    
    if detected_libs:
        print(f"[INFO] 共检测到 {len(detected_libs)} 个特殊库")
    else:
        print("[INFO] 未检测到需要特殊处理的库")
    
    # 构建命令
    cmd = build_pyinstaller_command(config, detected_libs)
    
    # 询问是否继续
    print("\n" + "="*60)
    response = input("是否继续打包? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消打包")
        sys.exit(0)
    
    # 清理旧文件（可选）
    if config.get('clean_before_build', True):
        clean_build_folders()
    
    # 执行打包
    success = run_packaging(cmd)
    
    if success:
        # 显示输出位置
        dist_folder = Path('dist')
        if dist_folder.exists():
            files = list(dist_folder.glob('*'))
            if files:
                print(f"\n[INFO] 打包文件位置:")
                for file in files:
                    print(f"  └─ {file}")
        
        print("\n[提示] 请在没有Python环境的机器上测试EXE文件")
        print("[提示] 如果仍有问题，请查看 build/*/warn-*.txt 文件")
    else:
        print("\n[提示] 打包失败，请检查:")
        print("  1. 所有依赖是否已安装")
        print("  2. 配置文件是否正确")
        print("  3. 查看详细错误日志")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
