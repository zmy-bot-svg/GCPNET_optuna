#!/bin/bash
echo "🚀 启动训练监控..."
python monitor.py --interval 30 --log-file monitor.log &
MONITOR_PID=$!
echo "监控进程PID: $MONITOR_PID"
echo $MONITOR_PID > monitor.pid
echo "✅ 监控已启动，日志文件: monitor.log"
echo "停止监控: kill $MONITOR_PID 或运行 ./stop_monitor.sh"
