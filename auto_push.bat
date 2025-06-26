@echo off
REM 自动推送脚本 - Auto Push Script (Windows)
chcp 65001 >nul

echo 🚀 开始自动推送到 GitHub...

REM 添加所有更改的文件
echo 📁 添加文件到暂存区...
git add .

REM 检查是否有更改需要提交
git diff --staged --quiet
if %errorlevel% equ 0 (
    echo ✅ 没有新的更改需要提交
    exit /b 0
)

REM 生成时间戳
for /f "tokens=1-5 delims=/ " %%a in ('date /t') do set mydate=%%c-%%a-%%b
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a:%%b
set timestamp=%mydate% %mytime%

REM 提交更改
echo 💾 提交更改...
git commit -m "Auto update: %timestamp%" -m "" -m "- 代码功能更新和优化" -m "- 添加代理支持功能" -m "- 修复API编码问题" -m "- 更新文档和指南"

REM 推送到远程仓库
echo 🌐 推送到 GitHub...
git push origin main

if %errorlevel% equ 0 (
    echo ✅ 成功推送到 GitHub!
    for /f "delims=" %%i in ('git remote get-url origin') do echo 🔗 仓库链接: %%i
) else (
    echo ❌ 推送失败，请检查网络连接或权限设置
    exit /b 1
)

echo 🎉 自动推送完成!
pause
