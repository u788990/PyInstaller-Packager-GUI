# PyInstaller-Packager-GUI
小白专用的程序云打包系统

# 别快EXE2026 - 一键打包Python游戏为EXE（支持云打包！）

本地打包慢、卡、占C盘？  
现在把工具放 GitHub，以后只需要点一下鼠标，就能几分钟后下载打包好的绿色EXE！

### 功能亮点（v4.3）
- 自动分析依赖 + 一键安装（清华/阿里镜像）
- 支持单文件 / 单文件夹两种模式
- 完美设置窗口图标 + 任务栏图标（PNG自动转ICO）
- 彻底解决临时文件夹残留问题（Atexit/Bootloader双策略）
- 多线程 + 缓存 + 排除无用模块 → 打包速度大幅提升30%~70%
- 消除 numpy.array_api 等常见警告
- 支持 GitHub Actions 云打包（完全免费！）

### 云打包用法（超简单）
1. Fork 本仓库  
2. 点击 Actions → 云打包 EXE → Run workflow  
3. 上传你的 `.py` 主程序（和资源文件、图标）  
4. 填好参数 → Run → 喝杯茶  
5. 下载 Artifact 里的 dist 文件夹，就是最终EXE！

本地使用：直接双击 `main.py` 运行GUI，和以前一模一样

作者：u788990@160.com  
觉得好用求星
