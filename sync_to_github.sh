#!/bin/bash
# GitHub自动同步脚本

echo "🚀 开始同步代码到GitHub..."

# 检查Git状态
echo "📊 检查Git状态..."
git status

# 添加所有更改
echo "📝 添加更改的文件..."
git add .

# 检查是否有更改
if git diff --staged --quiet; then
    echo "✅ 没有新的更改需要提交"
    exit 0
fi

# 显示将要提交的文件
echo "📋 将要提交的文件:"
git diff --staged --name-only

# 提交更改（带时间戳）
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
echo "💾 提交更改..."
git commit -m "Auto sync: $timestamp"

# 推送到GitHub
echo "⬆️  推送到GitHub..."
if git push origin main; then
    echo "✅ 同步成功！"
else
    echo "❌ 推送失败，请检查网络连接和权限"
    exit 1
fi
