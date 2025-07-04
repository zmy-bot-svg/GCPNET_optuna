#!/bin/bash
echo "🚀 同步到本地目录..."
TARGET="C:"
mkdir -p "$TARGET"
rsync -av --progress \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='data/' \
    --exclude='output*/' \
    --exclude='*.log' \
    --exclude='*.db' \
    --exclude='__pycache__/' \
    ./ "$TARGET/"
echo "✅ 本地同步完成: $TARGET"
