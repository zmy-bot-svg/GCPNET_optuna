#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本
在租服务器之前测试配置是否正确
"""

import os
import sys
import time
from utils.flags import Flags

def test_config_loading():
    """测试配置文件加载"""
    print("🔧 测试配置文件加载...")
    try:
        flags = Flags()
        config = flags.updated_config
        print(f"✅ 配置加载成功")
        print(f"   Project: {config.project_name}")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Hidden features: {config.hidden_features}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def test_dataset_loading(config):
    """测试数据集加载"""
    print("\n📊 测试数据集加载...")
    try:
        from main import setup_dataset
        dataset = setup_dataset(config)
        print(f"✅ 数据集加载成功")
        print(f"   数据点数量: {len(dataset)}")
        print(f"   数据集路径: {config.dataset_path}")
        return True
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("   请检查数据集路径是否正确")
        return False

def test_model_creation(config):
    """测试模型创建"""
    print("\n🧠 测试模型创建...")
    try:
        from main import setup_dataset, setup_model
        import torch
        
        dataset = setup_dataset(config)
        net = setup_model(dataset, config)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建成功")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   隐藏特征维度: {config.hidden_features}")
        
        # 测试GPU是否可用
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("   ⚠️  警告: GPU不可用，将使用CPU")
        
        return True
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_optuna_setup():
    """测试Optuna设置"""
    print("\n🔬 测试Optuna设置...")
    try:
        import optuna
        from optuna.integration import TensorBoardCallback
        
        # 创建临时study测试
        study = optuna.create_study(direction="minimize")
        print(f"✅ Optuna设置正常")
        print(f"   Optuna版本: {optuna.__version__}")
        return True
    except Exception as e:
        print(f"❌ Optuna设置失败: {e}")
        print("   请运行: pip install optuna optuna-integration")
        return False

def estimate_training_time(config):
    """估算训练时间"""
    print("\n⏱️  估算训练时间...")
    
    # 基于用户提供的信息：RTX 4060 8分钟/epoch
    base_time_per_epoch = 8  # 分钟 (RTX 4060)
    
    if "4090" in config.project_name:
        # RTX 4090大约比4060快4-5倍
        time_per_epoch = base_time_per_epoch / 4.5
        gpu_type = "RTX 4090"
    else:
        time_per_epoch = base_time_per_epoch
        gpu_type = "RTX 4060"
    
    epochs_per_trial = config.epochs
    num_trials = config.sweep_count
    
    total_epochs = epochs_per_trial * num_trials
    total_time_minutes = total_epochs * time_per_epoch
    total_time_hours = total_time_minutes / 60
    
    print(f"   GPU类型: {gpu_type}")
    print(f"   每epoch时间: {time_per_epoch:.1f}分钟")
    print(f"   试验次数: {num_trials}")
    print(f"   每试验epoch数: {epochs_per_trial}")
    print(f"   总计时间: {total_time_hours:.1f}小时")
    
    if "4090" in config.project_name:
        cost = total_time_hours * 2  # 2元/小时
        print(f"   预计成本: {cost:.0f}元")
    
    return total_time_hours

def main():
    print("🚀 GCPNet 快速测试")
    print("="*50)
    
    # 检测配置文件
    config_file = "./config.yml"
    if "4090" in sys.argv:
        config_file = "./config_4090.yml"
        print("📝 使用4090优化配置")
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        sys.exit(1)
    
    # 运行测试
    config = test_config_loading()
    if not config:
        sys.exit(1)
    
    if not test_dataset_loading(config):
        sys.exit(1)
    
    if not test_model_creation(config):
        sys.exit(1)
    
    if not test_optuna_setup():
        sys.exit(1)
    
    # 估算时间
    estimated_hours = estimate_training_time(config)
    
    print("\n" + "="*50)
    print("✅ 所有测试通过！")
    print("\n💡 建议:")
    if estimated_hours > 20:
        print("   - 推荐使用RTX 4090服务器")
        print("   - 使用config_4090.yml配置")
    else:
        print("   - 可以在本地运行")
    
    print("\n🚀 启动超参数搜索:")
    if "4090" in config_file:
        print("   python main.py --config_file config_4090.yml --task_type hyperparameter")
    else:
        print("   python main.py --config_file config.yml --task_type hyperparameter")

if __name__ == "__main__":
    main()