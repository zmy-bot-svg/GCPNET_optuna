#!/bin/bash
# 快速同步脚本（只同步代码文件）

echo "⚡ 快速同步代码文件..."

# 只添加代码相关文件
git add *.py *.yml *.yaml *.md *.txt *.sh utils/ model/ 2>/dev/null

# 检查是否有更改
if git diff --staged --quiet; then
    echo "✅ 没有代码文件更改"
    exit 0
fi

# 提交并推送
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "Quick sync: $timestamp"
git push origin main

echo "✅ 快速同步完成！"
