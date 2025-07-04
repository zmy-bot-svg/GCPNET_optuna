#!/bin/bash
# 强制同步脚本（谨慎使用）

echo "🔨 强制同步到GitHub..."
echo "⚠️  警告：此操作会覆盖远程仓库的更改"

read -p "确定要强制推送吗? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 取消操作"
    exit 0
fi

git add .
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "Force sync: $timestamp"
git push --force origin main

echo "✅ 强制同步完成！"
