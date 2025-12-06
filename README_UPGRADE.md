# PyInstaller-Packager-GUI v5.0 升级指南

## 🔥 重大更新内容

### 问题修复清单

| 问题 | 状态 | 说明 |
|------|------|------|
| jaraco 模块缺失 | ✅ 已修复 | 完整的 jaraco 隐藏导入配置 |
| pkg_resources 警告 | ✅ 已修复 | 添加 pkg_resources._vendor 全部子模块 |
| numpy.array_api 警告 | ✅ 已修复 | 添加到排除列表 |
| requirements.txt 未自动安装 | ✅ 已修复 | 自动检测并安装 |
| 数据文件未收集 | ✅ 已修复 | 自动收集资源文件 |
| 云模式代码混乱 | ✅ 已修复 | 独立 CloudPackager 类 |
| imageio/imageio_ffmpeg | ✅ 已修复 | 完整隐藏导入 |
| rembg/onnxruntime | ✅ 已修复 | 完整依赖链 |
| OpenCV (cv2) | ✅ 已修复 | 包含 cv2.data 等 |

---

## 📁 文件结构

```
PyInstaller-Packager-GUI/
├── main.py                          # v5.0 主程序（含 CloudPackager）
├── requirements.txt                 # 依赖列表
├── README.md                        # 说明文档
├── .github/
│   └── workflows/
│       └── cloud-build.yml          # GitHub Actions 工作流
└── 图标文件...
```

---

## 🚀 使用方法

### 方法1: 本地 GUI 模式

```bash
python main.py
```

### 方法2: 本地命令行模式

```bash
python main.py --cloud --source your_script.py --name YourApp --mode onefile --noconsole
```

### 方法3: GitHub Actions 云打包

1. Fork 本仓库
2. 上传你的 Python 文件和资源
3. 创建 `requirements.txt`（如果需要）
4. 进入 Actions → "Build EXE (Cloud Packager v5.0)"
5. 点击 "Run workflow"
6. 填写参数并运行
7. 下载 Artifacts

---

## 🔧 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cloud` | 启用云打包模式 | - |
| `--source` | Python 源文件路径 | main.py |
| `--name` | 输出 EXE 名称 | MyApp |
| `--mode` | 打包模式 (onefile/onedir) | onefile |
| `--noconsole` | 隐藏控制台窗口 | False |

---

## 📦 支持的库

CloudPackager 内置了以下库的完整隐藏导入配置：

- **图像处理**: PIL/Pillow, cv2/OpenCV, imageio, imageio_ffmpeg
- **AI/ML**: numpy, scipy, sklearn, skimage, onnxruntime, rembg
- **GUI**: tkinter, PyQt5, pygame
- **网络**: requests, aiohttp, urllib3
- **其他**: pooch, certifi, charset_normalizer

---

## ⚠️ 注意事项

1. **requirements.txt**: 必须包含所有依赖
2. **资源文件**: 放在源文件同目录或 assets/resources 子目录
3. **图标文件**: 支持 PNG/ICO，自动转换
4. **Python 版本**: 推荐 3.10-3.11

---

## 🐛 常见问题

### Q: 打包后运行报错 "No module named 'xxx'"

A: 在 `CloudPackager.HIDDEN_IMPORTS_MAP` 中添加对应模块

### Q: 资源文件找不到

A: 确保资源文件在源文件同目录，或使用相对路径

### Q: GitHub Actions 超时

A: 大型项目可能需要更长时间，考虑使用 onedir 模式

---

## 📝 更新日志

### v5.0 (2025-01)
- 完全重构云打包架构
- 新增 CloudPackager 独立类
- 自动安装 requirements.txt
- 完整的隐藏导入配置表
- 自动收集资源文件
- 修复 jaraco/pkg_resources 问题
- GitHub Actions 完全兼容

### v4.3
- 修复 numpy.array_api 警告
- 多线程依赖检测
- 缓存机制

---

## 📧 联系

作者: u788990@160.com

如有问题请提 Issue！
