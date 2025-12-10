"""
游戏精灵图生成工具 - 测试版
添加了详细的调试信息，用于测试rembg是否正确打包
"""

import sys
import os
from pathlib import Path

# ==================== 调试信息 ====================
print("="*60)
print("游戏精灵图生成工具 - 启动中...")
print("="*60)

# 打印Python环境信息
print(f"\n[调试] Python版本: {sys.version}")
print(f"[调试] Python路径: {sys.executable}")
print(f"[调试] 当前工作目录: {os.getcwd()}")

# 检测是否在打包环境中运行
if getattr(sys, 'frozen', False):
    print(f"[调试] 运行模式: 打包后的EXE")
    print(f"[调试] 临时目录: {sys._MEIPASS}")
    base_path = Path(sys._MEIPASS)
else:
    print(f"[调试] 运行模式: 开发环境")
    base_path = Path(__file__).parent

print(f"[调试] 基础路径: {base_path}")

# ==================== 导入测试 ====================
print("\n" + "-"*60)
print("开始测试模块导入...")
print("-"*60)

# 测试基础库
print("\n[1/5] 测试基础库...")
try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__} 导入成功")
except Exception as e:
    print(f"  ✗ numpy 导入失败: {e}")

try:
    from PIL import Image
    print(f"  ✓ Pillow 导入成功")
except Exception as e:
    print(f"  ✗ Pillow 导入失败: {e}")

# 测试onnxruntime（rembg的核心依赖）
print("\n[2/5] 测试 onnxruntime...")
try:
    import onnxruntime as ort
    print(f"  ✓ onnxruntime {ort.__version__} 导入成功")
    print(f"  ✓ 可用的执行提供程序: {ort.get_available_providers()}")
except Exception as e:
    print(f"  ✗ onnxruntime 导入失败: {e}")
    import traceback
    traceback.print_exc()

# 测试rembg主模块
print("\n[3/5] 测试 rembg 主模块...")
try:
    import rembg
    print(f"  ✓ rembg 导入成功")
    print(f"  ✓ rembg 路径: {rembg.__file__}")
    
    # 检查rembg版本
    if hasattr(rembg, '__version__'):
        print(f"  ✓ rembg 版本: {rembg.__version__}")
    
except Exception as e:
    print(f"  ✗ rembg 导入失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n[错误] rembg模块未正确打包！")
    print("请确保使用修复后的 cloud_packager_v61.py 进行打包")
    input("\n按回车键退出...")
    sys.exit(1)

# 测试rembg子模块
print("\n[4/5] 测试 rembg 子模块...")
try:
    from rembg import remove, new_session
    print(f"  ✓ rembg.remove 导入成功")
    print(f"  ✓ rembg.new_session 导入成功")
    
    from rembg.bg import remove as bg_remove
    print(f"  ✓ rembg.bg 导入成功")
    
    from rembg.session_factory import new_session as factory_session
    print(f"  ✓ rembg.session_factory 导入成功")
    
except Exception as e:
    print(f"  ✗ rembg 子模块导入失败: {e}")
    import traceback
    traceback.print_exc()

# 测试创建session
print("\n[5/5] 测试创建 rembg session...")
try:
    # 设置模型缓存目录
    import tempfile
    cache_dir = Path(tempfile.gettempdir()) / "rembg_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ['U2NET_HOME'] = str(cache_dir)
    print(f"  ✓ 模型缓存目录: {cache_dir}")
    
    # 尝试创建session（会触发模型下载）
    print(f"  → 正在创建 u2net session...")
    session = new_session('u2net')
    print(f"  ✓ Session 创建成功！")
    print(f"  ✓ rembg 完全可用！")
    
except Exception as e:
    print(f"  ✗ Session 创建失败: {e}")
    print(f"  → 这可能是因为模型文件未下载")
    print(f"  → 首次运行需要联网下载模型")
    import traceback
    traceback.print_exc()

# ==================== 主程序 ====================
print("\n" + "="*60)
print("模块导入测试完成！")
print("="*60)

def process_image(input_path, output_path):
    """处理单张图片，移除背景"""
    try:
        print(f"\n[处理] 输入: {input_path}")
        
        # 读取图片
        with open(input_path, 'rb') as f:
            input_data = f.read()
        
        # 移除背景
        print(f"[处理] 正在移除背景...")
        output_data = remove(input_data)
        
        # 保存结果
        with open(output_path, 'wb') as f:
            f.write(output_data)
        
        print(f"[处理] 输出: {output_path}")
        print(f"[处理] ✓ 处理成功！")
        return True
        
    except Exception as e:
        print(f"[处理] ✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print("游戏精灵图生成工具")
    print("="*60)
    
    # 简单的交互式测试
    print("\n请选择操作:")
    print("1. 测试移除背景功能")
    print("2. 批量处理图片")
    print("3. 退出")
    
    choice = input("\n请输入选项 (1-3): ").strip()
    
    if choice == '1':
        print("\n[提示] 请输入图片路径（或拖拽图片到此窗口）")
        input_path = input("图片路径: ").strip().strip('"')
        
        if not os.path.exists(input_path):
            print(f"[错误] 文件不存在: {input_path}")
            input("\n按回车键退出...")
            return
        
        # 生成输出路径
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_no_bg.png"
        
        # 处理图片
        if process_image(input_path, output_path):
            print(f"\n[成功] 已保存到: {output_path}")
        else:
            print(f"\n[失败] 处理失败，请查看错误信息")
        
        input("\n按回车键退出...")
        
    elif choice == '2':
        print("\n[批量处理功能开发中...]")
        input("\n按回车键退出...")
        
    else:
        print("\n再见！")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[提示] 用户中断")
    except Exception as e:
        print(f"\n[错误] 程序异常: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
