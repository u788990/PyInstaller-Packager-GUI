# Universal Cloud Packager v6.1 使用指南

## 功能特性

### ✅ 动态依赖检测
- 自动分析源文件导入
- 用 `pip show` 递归追踪依赖链
- 用 `pkgutil` 枚举所有子模块
- **支持任意第三方库，无需预设配置**

### ✅ 完整图标支持
| 图标类型 | 参数 | 推荐尺寸 | 说明 |
|---------|------|---------|------|
| EXE 图标 | `--exe-icon` | 480x480 | 文件管理器中显示的图标 |
| 窗口图标 | `--window-icon` | 28x28 | 窗口标题栏左上角图标 |
| 任务栏图标 | `--taskbar-icon` | 108x108 | Windows 任务栏显示的图标 |

### ✅ GUI 框架支持
- Tkinter（自动 Hook `Tk.__init__`）
- PyQt5（自动 Hook `QApplication.__init__`）
- PyQt6（同上）
- PySide6（同上）
- Pygame（自动 Hook `pygame.init`）

### ✅ 临时文件夹清理
- onefile 模式自动清理 `_MEIPASS` 临时目录
- 使用 `atexit` 注册清理函数

---

## 命令行用法

```bash
# 基本用法
python cloud_packager_v61.py --source main.py --name MyApp

# 完整用法（含图标）
python cloud_packager_v61.py \
    --source main.py \
    --name MyGame \
    --mode onefile \
    --noconsole \
    --exe-icon 480x480.png \
    --window-icon 28x28.png \
    --taskbar-icon 108x108.png
```

### 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--source` | `-s` | Python 源文件 | 必填 |
| `--name` | `-n` | 输出名称 | 必填 |
| `--mode` | `-m` | onefile/onedir | onefile |
| `--noconsole` | `-w` | 隐藏控制台 | False |
| `--exe-icon` | | EXE 图标 | None |
| `--window-icon` | | 窗口图标 | None |
| `--taskbar-icon` | | 任务栏图标 | None |
| `--data` | `-d` | 额外数据文件 | [] |
| `--no-cleanup` | | 不清理临时目录 | False |

---

## GitHub Actions 用法

1. 将以下文件放入您的仓库：
   - `cloud_packager_v61.py`（打包器）
   - `.github/workflows/build-v61.yml`（工作流）
   - `requirements.txt`（依赖）
   - 图标文件

2. 进入 GitHub → Actions → "Build EXE v6.1"

3. 点击 "Run workflow"，填写参数

4. 等待完成，下载 Artifacts

---

## 图标文件准备

### 推荐尺寸
```
480x480.png  → EXE 图标（会自动转 ICO）
28x28.png    → 窗口标题栏图标
108x108.png  → 任务栏图标
```

### 格式要求
- PNG 格式，RGBA 模式（支持透明）
- EXE 图标会自动转换为多尺寸 ICO

---

## 自动收集的资源

打包器会自动收集以下文件：

### 文件类型
- 图片：`.png`, `.jpg`, `.gif`, `.ico`, `.bmp`, `.webp`
- 配置：`.json`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.ini`
- 文本：`.txt`, `.csv`, `.xml`, `.md`
- 音频：`.wav`, `.mp3`, `.ogg`, `.flac`
- 字体：`.ttf`, `.otf`
- 模型：`.onnx`, `.pb`, `.pth`, `.h5`, `.pkl`

### 目录
- `assets/`, `resources/`, `data/`, `models/`
- `images/`, `icons/`, `fonts/`, `sounds/`

---

## 与原版的区别

| 功能 | 原 main.py | v6.1 |
|------|-----------|------|
| 依赖检测 | 硬编码配置表 | 动态自动检测 |
| 新库支持 | 需手动添加 | 自动支持 |
| 图标设置 | 复杂的 Hook | 简化封装 |
| 代码结构 | GUI + CLI 混合 | 纯 CLI，可独立运行 |
| 云打包 | 需要特殊参数 | 原生支持 |

---

## 示例项目结构

```
my_project/
├── main.py                    # 主程序
├── requirements.txt           # 依赖
├── cloud_packager_v61.py      # 打包器
├── 480x480.png               # EXE 图标
├── 28x28.png                 # 窗口图标
├── 108x108.png               # 任务栏图标
├── assets/                   # 资源目录（自动收集）
│   ├── images/
│   └── sounds/
└── .github/
    └── workflows/
        └── build-v61.yml     # GitHub Actions
```

---

## 常见问题

### Q: 为什么图标没有显示？
A: 检查图标文件是否在源文件同目录，确保文件名正确。

### Q: 打包后启动很慢？
A: onefile 模式需要解压，建议大型项目用 onedir 模式。

### Q: 某个库的隐藏导入缺失？
A: 虽然是动态检测，但某些特殊库可能需要手动添加。可以修改 `ALWAYS_INCLUDE` 列表。

---

## 联系

基于 u788990@160.com 的项目改进
