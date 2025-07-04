#!/bin/bash
# 检查项目状态

echo "🔍 GCPNet 项目状态检查"
echo "======================"

# 检查监控状态
echo "📊 监控状态:"
if [[ -f "monitor.pid" ]]; then
    PID=$(cat monitor.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ 监控正在运行 (PID: $PID)"
    else
        echo "❌ 监控进程不存在"
    fi
else
    echo "❌ 未找到监控PID文件"
fi

# 检查训练进程
echo ""
echo "🔄 训练进程:"
PYTHON_PROCS=$(pgrep -f "python.*main.py" | wc -l)
if [[ $PYTHON_PROCS -gt 0 ]]; then
    echo "✅ 发现 $PYTHON_PROCS 个Python训练进程"
    pgrep -f "python.*main.py" | while read pid; do
        echo "  PID $pid: $(ps -p $pid -o args --no-headers)"
    done
else
    echo "❌ 没有发现训练进程"
fi

# 检查GPU状态
echo ""
echo "🎮 GPU状态:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=', ' read name total used free util; do
        echo "  GPU: $name"
        echo "  内存: ${used}MB/${total}MB (空闲: ${free}MB)"
        echo "  利用率: ${util}%"
    done
else
    echo "❌ NVIDIA驱动不可用"
fi

# 检查Optuna数据库
echo ""
echo "📊 Optuna状态:"
DB_FILES=$(find . -name "*.db" -type f 2>/dev/null)
if [[ -n "$DB_FILES" ]]; then
    echo "$DB_FILES" | while read db; do
        echo "✅ 数据库: $db"
        echo "  大小: $(du -h "$db" | cut -f1)"
        echo "  修改时间: $(stat -c %y "$db" 2>/dev/null || stat -f %Sm "$db" 2>/dev/null)"
    done
else
    echo "❌ 未找到Optuna数据库文件"
fi

# 检查最近的日志
echo ""
echo "📝 最近日志:"
LOG_FILES=("monitor.log" "*.log")
for pattern in "${LOG_FILES[@]}"; do
    for log in $pattern; do
        if [[ -f "$log" ]]; then
            echo "📄 $log (最后10行):"
            tail -10 "$log" | sed 's/^/  /'
            echo ""
        fi
    done
done
