#!/bin/bash
# 自动推送脚本 - Auto Push Script

echo "🚀 开始自动推送到 GitHub..."

# 添加所有更改的文件
echo "📁 添加文件到暂存区..."
git add .

# 检查是否有更改需要提交
if git diff --staged --quiet; then
    echo "✅ 没有新的更改需要提交"
    exit 0
fi

# 生成时间戳
timestamp=$(date "+%Y-%m-%d %H:%M:%S")

# 提交更改
echo "💾 提交更改..."
git commit -m "Auto update: $timestamp

- 代码功能更新和优化
- 添加代理支持功能
- 修复API编码问题
- 更新文档和指南"

# 推送到远程仓库
echo "🌐 推送到 GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo "✅ 成功推送到 GitHub!"
    echo "🔗 仓库链接: $(git remote get-url origin)"
else
    echo "❌ 推送失败，请检查网络连接或权限设置"
    exit 1
fi

echo "🎉 自动推送完成!"
