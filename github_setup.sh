#!/bin/bash
# GitHub同步一键配置脚本
# 使用方法：bash github_sync_setup.sh

echo "🚀 GitHub同步配置脚本"
echo "======================"

# 检查是否在项目目录中
if [[ ! -f "main.py" && ! -f "utils/train_utils.py" ]]; then
    echo "❌ 请在GCPNet项目目录中运行此脚本"
    exit 1
fi

# 1. 配置Git用户信息
configure_git() {
    echo "📝 配置Git用户信息..."
    
    # 检查现有配置
    CURRENT_NAME=$(git config --global user.name 2>/dev/null)
    CURRENT_EMAIL=$(git config --global user.email 2>/dev/null)
    
    if [[ -n "$CURRENT_NAME" && -n "$CURRENT_EMAIL" ]]; then
        echo "✅ 已有Git配置:"
        echo "   用户名: $CURRENT_NAME"
        echo "   邮箱: $CURRENT_EMAIL"
        read -p "是否使用现有配置? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            read -p "请输入Git用户名: " GIT_NAME
            read -p "请输入Git邮箱: " GIT_EMAIL
            git config --global user.name "$GIT_NAME"
            git config --global user.email "$GIT_EMAIL"
        fi
    else
        read -p "请输入Git用户名: " GIT_NAME
        read -p "请输入Git邮箱: " GIT_EMAIL
        git config --global user.name "$GIT_NAME"
        git config --global user.email "$GIT_EMAIL"
    fi
    
    echo "✅ Git配置完成"
}

# 2. 初始化Git仓库
init_git_repo() {
    echo "📁 初始化Git仓库..."
    
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "✅ Git仓库已存在"
    else
        git init
        echo "✅ Git仓库初始化完成"
    fi
}

# 3. 创建.gitignore文件
create_gitignore() {
    echo "📝 创建.gitignore文件..."
    
    cat > .gitignore << 'EOF'
# Python相关
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# 虚拟环境
venv/
env/
ENV/
.venv
.env

# IDE和编辑器
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# 数据和模型文件（重要！避免上传大文件）
*.pth
*.pt
*.ckpt
*.h5
*.hdf5
*.pkl
*.pickle
*.npz
*.npy

# 数据集目录
data/
dataset/
datasets/
*.csv
*.json
*.zip
*.tar.gz
*.rar

# 输出和日志文件
output*/
logs/
runs/
*.log
*.out
*.err

# TensorBoard日志
tensorboard_logs/
tb_logs/

# Optuna数据库
*.db
*.sqlite
*.sqlite3

# 大型缓存文件
.cache/
cache/
tmp/
temp/

# 权重和检查点备份
*.backup*
checkpoint*/

# 系统文件
.DS_Store
Thumbs.db
*.tmp

# 配置文件中的敏感信息（如果有）
config_private.yml
secrets.yml
.env

# 编译文件
*.o
*.a
*.lib
*.so
*.dylib
*.dll

# 其他大文件
*.bin
*.data
*.index
*.meta

# 监控和同步脚本生成的文件
monitor.log
sync.log
*.pid
EOF

    echo "✅ .gitignore文件创建完成"
}

# 4. 创建同步脚本
create_sync_scripts() {
    echo "🔄 创建同步脚本..."
    
    # 自动同步脚本
    cat > sync_to_github.sh << 'EOF'
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
EOF

    # 快速同步脚本（只提交代码文件）
    cat > quick_sync.sh << 'EOF'
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
EOF

    # 强制同步脚本（解决冲突）
    cat > force_sync.sh << 'EOF'
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
EOF

    # 检查同步状态脚本
    cat > check_sync_status.sh << 'EOF'
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
EOF

    # 添加执行权限
    chmod +x sync_to_github.sh quick_sync.sh force_sync.sh check_sync_status.sh
    
    echo "✅ 同步脚本创建完成"
}

# 5. 首次提交
initial_commit() {
    echo "📦 首次提交..."
    
    # 添加所有文件
    git add .
    
    # 显示将要提交的文件
    echo "📋 将要提交的文件:"
    git diff --staged --name-only | head -20
    echo "..."
    
    # 首次提交
    git commit -m "Initial commit: GCPNet Optuna hyperparameter optimization"
    
    echo "✅ 首次提交完成"
}

# 6. 配置远程仓库
setup_remote() {
    echo "🌐 配置远程GitHub仓库..."
    
    # 检查是否已有远程仓库
    if git remote | grep -q "origin"; then
        current_remote=$(git remote get-url origin)
        echo "✅ 已有远程仓库: $current_remote"
        read -p "是否要更改远程仓库地址? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "请输入新的GitHub仓库地址: " REPO_URL
            git remote set-url origin "$REPO_URL"
        fi
    else
        echo "请先在GitHub上创建仓库，然后提供仓库地址"
        echo "GitHub仓库地址格式示例："
        echo "  HTTPS: https://github.com/username/repo-name.git"
        echo "  SSH:   git@github.com:username/repo-name.git"
        echo ""
        read -p "请输入GitHub仓库地址: " REPO_URL
        
        if [[ -n "$REPO_URL" ]]; then
            git remote add origin "$REPO_URL"
            echo "✅ 远程仓库配置完成"
        else
            echo "⚠️  跳过远程仓库配置"
            return 1
        fi
    fi
    
    return 0
}

# 7. 推送到GitHub
push_to_github() {
    echo "⬆️  推送到GitHub..."
    
    # 设置主分支
    git branch -M main
    
    # 推送到远程仓库
    if git push -u origin main; then
        echo "✅ 成功推送到GitHub！"
        return 0
    else
        echo "❌ 推送失败"
        echo "💡 可能的原因："
        echo "   1. 网络连接问题"
        echo "   2. 认证问题（需要Personal Access Token）"
        echo "   3. 仓库地址错误"
        echo ""
        echo "🔧 解决方案："
        echo "   1. 检查网络连接"
        echo "   2. 配置Personal Access Token"
        echo "   3. 使用SSH密钥认证"
        return 1
    fi
}

# 8. 创建定时同步
setup_auto_sync() {
    echo "⏰ 设置定时自动同步..."
    
    read -p "是否要设置定时自动同步? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "选择同步频率："
        echo "1. 每小时"
        echo "2. 每2小时" 
        echo "3. 每6小时"
        echo "4. 每天"
        read -p "请选择 (1-4): " -n 1 -r
        echo
        
        case $REPLY in
            1) CRON_SCHEDULE="0 * * * *" ;;
            2) CRON_SCHEDULE="0 */2 * * *" ;;
            3) CRON_SCHEDULE="0 */6 * * *" ;;
            4) CRON_SCHEDULE="0 0 * * *" ;;
            *) echo "无效选择，跳过定时同步"; return ;;
        esac
        
        PROJECT_DIR=$(pwd)
        CRON_JOB="$CRON_SCHEDULE cd $PROJECT_DIR && ./quick_sync.sh >> sync_auto.log 2>&1"
        
        # 添加到crontab
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        
        echo "✅ 定时同步设置完成"
        echo "📝 日志文件: sync_auto.log"
    fi
}

# 主要执行流程
main() {
    echo "开始GitHub同步配置..."
    echo ""
    
    # 1. 配置Git
    configure_git
    echo ""
    
    # 2. 初始化仓库
    init_git_repo
    echo ""
    
    # 3. 创建.gitignore
    create_gitignore
    echo ""
    
    # 4. 创建同步脚本
    create_sync_scripts
    echo ""
    
    # 5. 首次提交
    initial_commit
    echo ""
    
    # 6. 配置远程仓库
    if setup_remote; then
        echo ""
        
        # 7. 推送到GitHub
        if push_to_github; then
            echo ""
            
            # 8. 设置定时同步
            setup_auto_sync
        fi
    fi
    
    echo ""
    echo "🎉 GitHub同步配置完成！"
    echo "=========================="
    echo ""
    echo "📋 可用命令："
    echo "  ./sync_to_github.sh      - 完整同步"
    echo "  ./quick_sync.sh          - 快速同步（仅代码）"
    echo "  ./force_sync.sh          - 强制同步（谨慎使用）"
    echo "  ./check_sync_status.sh   - 检查状态"
    echo ""
    echo "🚀 立即使用："
    echo "  ./sync_to_github.sh"
    echo ""
    echo "📖 使用说明："
    echo "  1. 修改代码后运行: ./quick_sync.sh"
    echo "  2. 完整同步运行: ./sync_to_github.sh"
    echo "  3. 检查状态: ./check_sync_status.sh"
}

# 运行主函数
main "$@"
