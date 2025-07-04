#!/bin/bash
# 检查同步状态脚本

echo "🔍 检查Git同步状态"
echo "==================="

# 检查远程仓库
echo "📡 远程仓库:"
git remote -v

echo ""
echo "📊 本地状态:"
git status --porcelain | head -10

echo ""
echo "📝 最近提交:"
git log --oneline -5

echo ""
echo "📏 仓库大小:"
du -sh .git

echo ""
echo "🔍 大文件检查 (>10MB):"
find . -type f -size +10M | grep -v ".git" | head -5
