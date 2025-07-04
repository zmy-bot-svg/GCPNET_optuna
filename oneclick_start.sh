#!/bin/bash
# 一键启动训练和监控

echo "🚀 GCPNet 一键启动"
echo "=================="

# 启动监控（后台）
if [[ -f "start_monitor.sh" ]]; then
    echo "📊 启动监控..."
    ./start_monitor.sh
    sleep 2
fi

# 检查GPU状态
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU状态:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
fi

# 显示配置
echo "📋 当前配置:"
if [[ -f "config4090.yml" ]]; then
    echo "配置文件: config4090.yml"
elif [[ -f "config.yml" ]]; then
    echo "配置文件: config.yml"
fi

echo ""
echo "🎯 可用操作:"
echo "1. 启动超参数搜索: python main.py --config config.yml --task_type hyperparameter"
echo "2. 启动训练: python main.py --config config.yml --task_type train" 
echo "3. 查看监控: tail -f monitor.log"
echo "4. 停止监控: ./stop_monitor.sh"
echo "5. 同步到GitHub: ./sync_to_github.sh"
echo ""

read -p "🤔 是否立即启动超参数搜索? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CONFIG_FILE="config4090.yml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        CONFIG_FILE="config.yml"
    fi
    
    echo "🚀 启动超参数搜索..."
    python main.py --config "$CONFIG_FILE" --task_type hyperparameter
fi
