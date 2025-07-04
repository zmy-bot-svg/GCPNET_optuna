#!/bin/bash
if [[ -f "monitor.pid" ]]; then
    PID=$(cat monitor.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ 监控进程已停止"
    else
        echo "⚠️  监控进程不存在"
    fi
    rm -f monitor.pid
else
    echo "⚠️  未找到监控PID文件"
fi
