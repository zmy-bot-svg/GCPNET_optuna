#!/usr/bin/python
# -*- encoding: utf-8 -*-

# 导入必要的库用于时间处理、操作系统交互和时间计算
import datetime
import os
import time
import copy  # 新增：用于深拷贝配置

# 导入PyTorch相关库用于深度学习模型构建和训练
import torch
import wandb  # 用于实验跟踪和可视化
import torch.nn as nn
import torchmetrics  # 用于计算各种评估指标
from torch_geometric.transforms import Compose  # 用于组合多个数据变换

# 新增：导入 Optuna
import optuna
import optuna.exceptions  # 新增：用于剪枝异常处理
from optuna.integration import TensorBoardCallback
import tensorboard

# 导入自定义模块
from model import GCPNet  # GCPNet模型的主要实现
from utils.keras_callbacks import WandbCallback  # Wandb回调函数
from utils.dataset_utils import MP18, dataset_split, get_dataloader  # 数据集处理工具
from utils.flags import Flags  # 配置参数管理
from utils.train_utils import KerasModel, LRScheduler  # 训练工具和学习率调度器
from utils.transforms import GetAngle, ToFloat  # 数据变换工具

# 设置NumExpr库的最大线程数为24，用于加速数值计算
os.environ["NUMEXPR_MAX_THREADS"] = "24"
# 开启调试模式，用于打印调试信息
debug = True 

# 导入日志相关库
import logging
from logging.handlers import RotatingFileHandler

# 新增：显存管理工具
import gc
import time

class GPUMemoryManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def get_memory_info(self):
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        free = total - cached
        
        
        return {"total": total, "allocated": allocated, "cached": cached, "free": free}
    
    def safe_cleanup(self, force=False):
        if not torch.cuda.is_available():
            return
        
        before = self.get_memory_info()
        gc.collect()
        torch.cuda.empty_cache()
        
        if force:
            torch.cuda.synchronize()
            time.sleep(0.1)
            torch.cuda.empty_cache()
        
        after = self.get_memory_info()
        if self.verbose:
            freed = before['cached'] - after['cached']
            if freed > 0.1:
                print(f"🧹 Cleaned {freed:.2f}GB GPU memory")

# 配置日志系统，用于记录训练过程中的重要信息
def log_config(log_file='test.log'):
    # 定义日志格式：[时间戳][日志级别]: 消息内容
    LOG_FORMAT = '[%(asctime)s][%(levelname)s]: %(message)s'
    # 设置日志级别为INFO，记录重要的训练信息
    level = logging.INFO
    # 配置基础日志设置
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # 创建文件日志处理器，支持日志轮转（最大2MB，保留3个备份文件）
    log_file_handler = RotatingFileHandler(filename=log_file, maxBytes=2*1024*1024, backupCount=3)
    # 设置日志格式化器
    formatter = logging.Formatter(LOG_FORMAT)
    log_file_handler.setFormatter(formatter)
    # 将文件处理器添加到根日志记录器
    logging.getLogger('').addHandler(log_file_handler)

# 设置随机种子，确保实验的可重复性
def set_seed(seed):
    # 导入随机数生成相关库
    import random
    import numpy as np
    # 设置Python原生random模块的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch CPU操作的随机种子
    torch.manual_seed(seed)
    # 设置PyTorch GPU操作的随机种子（适用于所有GPU）
    torch.cuda.manual_seed_all(seed)
    # 启用确定性算法，确保GPU计算结果可重复
    torch.backends.cudnn.deterministic = True
    # 禁用cudnn的benchmark模式，虽然可能影响性能但保证结果一致
    torch.backends.cudnn.benchmark = False

# 设置和初始化数据集，这是GCPNet训练的第一步
def setup_dataset(config):
    dataset = MP18(root=config.dataset_path, name=config.dataset_name, transform=Compose([GetAngle(), ToFloat(
        )]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True, points=config.points, target_name=config.target_name)
    return dataset

# 初始化GCPNet模型，配置模型的各种超参数
def setup_model(dataset, config):
    net = GCPNet(
            data=dataset,
            firstUpdateLayers=config.firstUpdateLayers,
            secondUpdateLayers=config.secondUpdateLayers,
            atom_input_features=config.atom_input_features,
            edge_input_features=config.edge_input_features,
            triplet_input_features=config.triplet_input_features,
            embedding_features=config.embedding_features,
            hidden_features=config.hidden_features,
            output_features=config.output_features,
            min_edge_distance=config.min_edge_distance,
            max_edge_distance=config.max_edge_distance,
            link=config.link,
            dropout_rate=config.dropout_rate,
        )
    return net

# 设置优化器，用于模型参数的更新
def setup_optimizer(net, config):
    optimizer = getattr(torch.optim, config.optimizer)(
        net.parameters(),
        lr=config.lr,
        **config.optimizer_args
    )
    if config.debug:
        print(f"optimizer: {optimizer}")
    return optimizer

# 设置学习率调度器，用于在训练过程中动态调整学习率
def setup_schduler(optimizer, config):
    scheduler = LRScheduler(optimizer, config.scheduler, config.scheduler_args)
    return scheduler

# 构建Keras风格的模型包装器，简化训练流程
def build_keras(net, optimizer, scheduler):
    model = KerasModel(
        net=net,
        loss_fn=nn.L1Loss(),
        metrics_dict={
            "mae": torchmetrics.MeanAbsoluteError(),
            "mape": torchmetrics.MeanAbsolutePercentageError()
        }, 
        optimizer=optimizer,
        lr_scheduler=scheduler
    )
    return model

# 主要的训练函数，执行完整的模型训练流程
def train(config, printnet=False, trial=None):  # 新增trial参数用于剪枝
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 如果启用了wandb日志记录，初始化Weights & Biases实验跟踪
    if config.log_enable:
        wandb.init(project=config.project_name, name=name, save_code=False)

    # 第1步：加载和准备数据
    dataset = setup_dataset(config)
    train_dataset, val_dataset, test_dataset = dataset_split(
        dataset, train_size=0.8, valid_size=0.1, test_size=0.1, seed=config.seed, debug=debug) 
    train_loader, val_loader, test_loader = get_dataloader(
        train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    # 第2步：加载和初始化网络模型
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)
    if config.debug and printnet:
        print(net)

    # 第3步：设置优化器和学习率调度器
    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    # 第4步：开始训练过程
    if config.log_enable:
        callbacks = [WandbCallback(project=config.project_name, config=config)]
    else:
        callbacks = None
    
    model = build_keras(net, optimizer, scheduler)
    
    # 开始训练模型（传递trial用于剪枝）
    history = model.fit(
        train_loader,
        val_loader,
        ckpt_path=os.path.join(config.output_dir, config.net+'.pth'),
        epochs=config.epochs,
        monitor='val_loss',
        mode='min',
        patience=config.patience,
        plot=True,
        callbacks=callbacks,
        trial=trial  # 新增：传递trial对象用于剪枝
    )
    
    # 在测试集上评估模型性能
    test_result = model.evaluate(test_loader)
    print(test_result)
    
    # 如果启用日志记录，记录最终的测试结果
    if config.log_enable:
        wandb.log({
            "test_mae": test_result['val_mae'],
            "test_mape": test_result['val_mape'],
            "total_params": model.total_params()
        })
        wandb.finish()

    # 返回训练历史和最佳验证误差
    best_val_mae = min(history['val_mae']) if 'val_mae' in history else float('inf')
    return best_val_mae

# 新增：Optuna 超参数搜索的目标函数
def objective(trial, base_config):
    """
    Optuna 优化的目标函数
    
    Args:
        trial: Optuna trial 对象
        base_config: 基础配置对象
    
    Returns:
        float: 要最小化的目标值（验证集 MAE）
    """
    # 深拷贝配置以避免修改原始配置
    config = copy.deepcopy(base_config)
    
    # 优化后的超参数搜索空间（更聚焦在你之前效果好的区域）
    # 学习率：以你之前好的0.001为中心，适当扩大范围
    config.lr = trial.suggest_float("lr", 0.0005, 0.002, log=True)
    
    # Dropout：降低上限，避免过高的dropout影响学习
    config.dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.2)  # 降低上限从0.3到0.2
    
    # weight_decay：适中范围
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)  # 降低上限
    config.optimizer_args = config.optimizer_args.copy()
    config.optimizer_args['weight_decay'] = weight_decay
    
    # 网络层数：保持原范围
    config.firstUpdateLayers = trial.suggest_categorical("firstUpdateLayers", [3, 4, 5])
    config.secondUpdateLayers = trial.suggest_categorical("secondUpdateLayers", [3, 4, 5])
    
    # 隐藏层维度：更倾向于大模型，增加256选项
    config.hidden_features = trial.suggest_categorical("hidden_features", [96, 128, 160])  # 增加160选项
    
    # ========================================================== #
    # 🚀 清晰显示当前试验的超参数
    # ========================================================== #
    print("\n" + "="*60)
    print(f"🚀 Starting Trial #{trial.number}")
    print("  Parameters:")
    for key, value in trial.params.items():
        print(f"    - {key}: {value}")
    print("="*60 + "\n")
    # ========================================================== #
    
    # 关闭 wandb 日志（由 Optuna 管理）
    config.log_enable = False
    
    # 为每个 trial 创建唯一的输出目录
    trial_name = f"trial_{trial.number}"
    config.output_dir = os.path.join(base_config.output_dir, trial_name)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # 执行训练并返回最佳验证 MAE
    try:
        # 清理GPU内存，确保每个trial开始时内存充足
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        best_val_mae = train(config, printnet=False, trial=trial)  # 传递trial对象用于剪枝
        
        # 训练完成后显示结果
        print(f"\n🏁 Trial #{trial.number} completed!")
        print(f"   Result: {best_val_mae:.6f}")
        print(f"   Current best: {trial.study.best_value:.6f}" if trial.study.best_value else "   First trial")
        
        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
        print("="*60)
        
        return best_val_mae
    except optuna.exceptions.TrialPruned:
        # 处理剪枝异常
        print(f"\n✂️  Trial #{trial.number} pruned!")
        print(f"   Reason: Performance not promising, stopped early")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("="*60)
        raise  # 重新抛出，让Optuna处理
    except torch.cuda.OutOfMemoryError as e:
        # 处理GPU内存不足
        print(f"\n💥 Trial #{trial.number} failed: GPU Out of Memory")
        print(f"   建议：减少batch_size或hidden_features")
        
        # 强制清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("="*60)
        return float('inf')  # 返回一个很大的值表示失败
    except Exception as e:
        print(f"❌ Trial {trial.number} failed with error: {e}")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("="*60)
        return float('inf')  # 返回一个很大的值表示失败

# 4090优化的超参数搜索目标函数
def objective_4090(trial, base_config):
    gpu_manager = GPUMemoryManager(verbose=True)
    config = copy.deepcopy(base_config)
    # 🔍 添加详细诊断
    print(f"\n🔍 Trial #{trial.number} - 详细显存诊断:")
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        free = total - cached
        print(f"   GPU总显存: {total:.2f}GB")
        print(f"   已分配: {allocated:.2f}GB") 
        print(f"   已缓存: {cached:.2f}GB")
        print(f"   可用: {free:.2f}GB")
        
        # 强制清理
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 清理后再检查
        allocated_after = torch.cuda.memory_allocated() / 1e9
        cached_after = torch.cuda.memory_reserved() / 1e9
        free_after = total - cached_after
        print(f"   清理后-已分配: {allocated_after:.2f}GB")
        print(f"   清理后-已缓存: {cached_after:.2f}GB")  
        print(f"   清理后-可用: {free_after:.2f}GB")
        
        if free_after < 10.0:  # 如果可用显存少于10GB就有问题
            print(f"   ⚠️ 警告：可用显存过少！")
            
    try:
        # 试验开始前清理显存
        gpu_manager.safe_cleanup(force=False)
        
        # 更保守的超参数范围
        config.lr = trial.suggest_float("lr", 0.0005, 0.002, log=True)
        config.dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.18)
        
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 8e-5, log=True)
        config.optimizer_args = config.optimizer_args.copy()
        config.optimizer_args['weight_decay'] = weight_decay
        
        config.firstUpdateLayers = trial.suggest_categorical("firstUpdateLayers", [3, 4, 5])
        config.secondUpdateLayers = trial.suggest_categorical("secondUpdateLayers", [3, 4, 5])
        
        # 🔧 更安全的参数范围
        config.hidden_features = trial.suggest_categorical("hidden_features", [96, 128, 160])  # 移除192, 224
        config.batch_size = trial.suggest_categorical("batch_size", [48, 64, 96])  # 移除128, 160
        
        print("\n" + "="*60)
        print(f"🚀 Starting Trial #{trial.number} (Memory Safe)")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    - {key}: {value}")
        print("="*60 + "\n")
        
        config.log_enable = False
        trial_name = f"trial_{trial.number}"
        config.output_dir = os.path.join(base_config.output_dir, trial_name)
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        
        best_val_mae = train(config,  trial=trial)
        
        print(f"\n🏁 Trial #{trial.number} completed!")
        print(f"   Result: {best_val_mae:.6f}")
        print(f"   Current best: {trial.study.best_value:.6f}" if trial.study.best_value else "   First trial")
        
        return best_val_mae
        
    except optuna.exceptions.TrialPruned:
        print(f"\n✂️  Trial #{trial.number} pruned!")
        gpu_manager.safe_cleanup(force=True)
        raise
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n💥 Trial #{trial.number} - GPU OOM")
        print(f"   配置：batch_size={config.batch_size}, hidden_features={config.hidden_features}")
        gpu_manager.safe_cleanup(force=True)
        return float('inf')
        
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        gpu_manager.safe_cleanup(force=True)
        return float('inf')
        
    finally:
        gpu_manager.safe_cleanup(force=False)
        print("="*60)

# 交叉验证训练函数，用于更可靠的模型性能评估
def train_CV(config):
    from utils.dataset_utils_strong import loader_setup_CV, split_data_CV
    dataset = setup_dataset(config)
    cv_dataset = split_data_CV(dataset, num_folds=config.num_folds, seed=config.seed)
    cv_error = []

    for index in range(0, len(cv_dataset)):
        if not(os.path.exists(output_dir := f"{config.output_dir}/{index}")):
            os.makedirs(output_dir)

        train_loader, test_loader, train_dataset, _ = loader_setup_CV(
            index, config.batch_size, cv_dataset, num_workers=config.num_workers
        )
        
        rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = setup_model(dataset, config).to(rank)
        
        optimizer = setup_optimizer(net, config)
        scheduler = setup_schduler(optimizer, config)

        model = build_keras(net, optimizer, scheduler)
        model.fit(
            train_loader, 
            None,
            ckpt_path=os.path.join(output_dir, config.net+'.pth'), 
            epochs=config.epochs,
            monitor='train_loss',
            mode='min', 
            patience=config.patience, 
            plot=True
        )

        test_error = model.evaluate(test_loader)['val_mae']
        logging.info("fold: {:d}, Test Error: {:.5f}".format(index+1, test_error)) 
        cv_error.append(test_error)
    
    import numpy as np
    mean_error = np.array(cv_error).mean()
    std_error = np.array(cv_error).std()
    logging.info("CV Error: {:.5f}, std Error: {:.5f}".format(mean_error, std_error))
    return cv_error

# 预测函数，用于在新数据上进行推理
def predict(config):
    dataset = setup_dataset(config)
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False,)

    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    model = build_keras(net, optimizer, scheduler)
    model.predict(test_loader, ckpt_path=config.model_path, test_out_path=config.output_path)

# 可视化函数，用于分析和可视化模型的特征表示
def visualize(config):
    from utils.dataset_utils_strong import MP18, dataset_split, get_dataloader
    from utils.transforms import GetAngle, ToFloat

    dataset = MP18(root=config.dataset_path,name=config.dataset_name,transform=Compose([GetAngle(),ToFloat()]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True,points=config.points,target_name=config.target_name)

    train_dataset, val_dataset, test_dataset = dataset_split(dataset,train_size=0.8,valid_size=0.1,test_size=0.1,seed=config.seed, debug=debug)
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)
    print("optimizer:",optimizer)

    model = KerasModel(net=net, loss_fn=nn.L1Loss(), metrics_dict={"mae":torchmetrics.MeanAbsoluteError(),"mape":torchmetrics.MeanAbsolutePercentageError()},optimizer=optimizer,lr_scheduler = scheduler)
    data_loader, _, _ = get_dataloader(dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)
   
    model.analysis(net_name=config.net, test_data=data_loader,ckpt_path=config.model_path,tsne_args=config.visualize_args)

    return model

# 主程序入口点，程序从这里开始执行
if __name__ == "__main__":

    # 忽略PyTorch的TypedStorage废弃警告，为了避免在使用旧版本PyTorch时出现警告信息
    import warnings
    warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
    
    # 初始化配置管理器，加载所有配置参数
    flags = Flags()
    config = flags.updated_config
    
    # 生成基于时间的唯一输出目录名称
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = os.path.join(config.output_dir, name)
    if not(os.path.exists(config.output_dir)):
        os.makedirs(config.output_dir)
    set_seed(config.seed)

    # 根据配置的任务类型执行相应的操作
    if config.task_type.lower() == 'train':
        train(config)
    
    # 修改后的超参数搜索模式：使用 Optuna 替代 Wandb
    # 修复后的超参数搜索部分 - 替换原来的elif block
    elif config.task_type.lower() == 'hyperparameter':
        print("Starting Optuna hyperparameter optimization...")
        
        # 🔧 第一步：处理数据库存储
        try:
            # 确保输出目录存在
            os.makedirs(config.output_dir, exist_ok=True)
            
            # 创建唯一的数据库文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            db_name = f"{config.project_name}_{timestamp}.db"
            db_path = os.path.abspath(os.path.join(config.output_dir, db_name))
            
            # 处理Windows路径问题
            if os.name == 'nt':  # Windows
                storage_url = f"sqlite:///{db_path.replace(os.sep, '/')}"
            else:
                storage_url = f"sqlite:///{db_path}"
            
            print(f"📁 数据库路径: {db_path}")
            
            # 测试数据库连接
            test_study = optuna.create_study(
                direction="minimize",
                study_name=f"test_{timestamp}",
                storage=storage_url
            )
            print("✅ 数据库连接成功")
            storage_name = storage_url
            
        except Exception as e:
            print(f"⚠️ 数据库设置失败: {e}")
            print("🔄 使用内存存储模式")
            storage_name = None
        
        # 🔧 第二步：创建Study
        try:
            study = optuna.create_study(
                direction="minimize",
                study_name=config.project_name,
                storage=storage_name,
                sampler=optuna.samplers.TPESampler(
                    seed=config.seed,
                    n_startup_trials=max(2, config.sweep_count // 5)
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=2,
                    n_warmup_steps=3,  # 前3个epoch不剪枝
                    interval_steps=1
                ),
                load_if_exists=True
            )
            print("✅ Optuna study创建成功")
        except Exception as e:
            print(f"❌ Study创建失败: {e}")
            # 最简单的fallback
            study = optuna.create_study(direction="minimize")
            print("🔄 使用基础配置")
        
        # 🔧 第三步：定义目标函数
        def hyperparameter_objective(trial):
            """
            超参数优化目标函数
            """
            trial_start_time = time.time()
            
            try:
                # 清理资源
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 深拷贝配置
                trial_config = copy.deepcopy(config)
                
                # 🎯 超参数搜索空间
                trial_config.lr = trial.suggest_float("lr", 0.0005, 0.003, log=True)
                trial_config.dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.25)
                
                # 优化器参数
                weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                trial_config.optimizer_args = trial_config.optimizer_args.copy()
                trial_config.optimizer_args['weight_decay'] = weight_decay
                
                # 网络结构
                trial_config.firstUpdateLayers = trial.suggest_categorical("firstUpdateLayers", [3, 4, 5])
                trial_config.secondUpdateLayers = trial.suggest_categorical("secondUpdateLayers", [3, 4, 5])
                trial_config.hidden_features = trial.suggest_categorical("hidden_features", [96, 128, 160, 192])
                
                # 根据GPU选择batch size
                if "4090" in config.project_name.lower():
                    trial_config.batch_size = trial.suggest_categorical("batch_size", [32, 48, 64, 96])
                else:
                    trial_config.batch_size = trial.suggest_categorical("batch_size", [16, 24, 32, 48])
                
                # 🚀 显示试验信息
                print(f"\n{'='*60}")
                print(f"🚀 Trial #{trial.number} 开始")
                print(f"⏱️  预计用时: {config.epochs * 2:.1f}分钟")
                print("📋 参数:")
                for key, value in trial.params.items():
                    print(f"    {key}: {value}")
                print(f"{'='*60}")
                
                # 🔧 配置调整
                trial_config.log_enable = False  # 关闭wandb
                
                # 创建试验目录
                trial_name = f"trial_{trial.number:03d}"
                trial_config.output_dir = os.path.join(config.output_dir, trial_name)
                os.makedirs(trial_config.output_dir, exist_ok=True)
                
                # 设置独特的随机种子
                trial_config.seed = config.seed + trial.number * 42
                set_seed(trial_config.seed)
                
                # 🏃‍♂️ 执行训练
                best_val_mae = train(trial_config, printnet=False, trial=trial)
                
                # 📊 显示结果
                trial_time = time.time() - trial_start_time
                print(f"\n🏁 Trial #{trial.number} 完成!")
                print(f"⏱️  耗时: {trial_time/60:.1f}分钟")
                print(f"📈 验证MAE: {best_val_mae:.6f}")
                
                # 显示排名
                try:
                    completed_trials = [t for t in trial.study.trials 
                                    if t.state == optuna.trial.TrialState.COMPLETE]
                    if len(completed_trials) > 1:
                        current_rank = sorted([t.value for t in completed_trials]).index(best_val_mae) + 1
                        print(f"🏆 当前排名: {current_rank}/{len(completed_trials)}")
                        print(f"🥇 最佳成绩: {trial.study.best_value:.6f}")
                except:
                    pass
                
                print(f"{'='*60}")
                return best_val_mae
                
            except optuna.exceptions.TrialPruned:
                trial_time = time.time() - trial_start_time
                print(f"\n✂️ Trial #{trial.number} 被剪枝")
                print(f"⏱️  节省时间: {trial_time/60:.1f}分钟")
                print(f"{'='*60}")
                raise
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n💥 Trial #{trial.number} GPU内存不足")
                print(f"🔧 建议: 减少batch_size或hidden_features")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"{'='*60}")
                return float('inf')
                
            except Exception as e:
                trial_time = time.time() - trial_start_time
                print(f"\n❌ Trial #{trial.number} 失败")
                print(f"⚠️  错误: {str(e)[:100]}...")
                print(f"⏱️  耗时: {trial_time/60:.1f}分钟")
                
                # 清理资源
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                print(f"{'='*60}")
                return float('inf')
            
            finally:
                # 最终清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # 🔧 第四步：执行优化
        print(f"\n🎯 开始超参数优化")
        print(f"📊 计划试验数: {config.sweep_count}")
        print(f"💾 存储模式: {'数据库' if storage_name else '内存'}")
        if storage_name:
            print(f"📁 数据库文件: {db_path}")
        print(f"{'='*60}")
        
        optimization_start = time.time()
        
        try:
            study.optimize(
                hyperparameter_objective,
                n_trials=config.sweep_count,
                timeout=None,
                catch=(Exception,),  # 捕获异常但继续
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print(f"\n⏹️ 用户中断优化")
        except Exception as e:
            print(f"\n💥 优化过程出错: {e}")
        
        # 🔧 第五步：结果分析
        optimization_time = time.time() - optimization_start
        
        print(f"\n{'='*60}")
        print(f"🎉 优化完成!")
        print(f"⏱️  总耗时: {optimization_time/3600:.1f}小时")
        
        # 统计信息
        all_trials = study.trials
        completed = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
        
        print(f"\n📊 试验统计:")
        print(f"   总试验: {len(all_trials)}")
        print(f"   成功: {len(completed)}")
        print(f"   剪枝: {len(pruned)} (节省 {len(pruned)/(len(all_trials) or 1)*100:.1f}%)")
        print(f"   失败: {len(failed)}")
        
        # 最佳结果
        if completed:
            best_trial = study.best_trial
            print(f"\n🏆 最佳结果:")
            print(f"   Trial: #{best_trial.number}")
            print(f"   验证MAE: {study.best_value:.6f}")
            print(f"   参数:")
            for key, value in study.best_params.items():
                print(f"     {key}: {value}")
            
            # 保存结果
            results_file = os.path.join(config.output_dir, "best_hyperparameters.txt")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(f"最佳Trial: #{best_trial.number}\n")
                f.write(f"最佳验证MAE: {study.best_value:.6f}\n")
                f.write(f"优化耗时: {optimization_time/3600:.2f}小时\n\n")
                f.write("最佳超参数:\n")
                for key, value in study.best_params.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"\n💾 结果保存至: {results_file}")
            
            # 生成配置文件
            best_config_file = os.path.join(config.output_dir, "best_config.yml")
            try:
                import yaml
                best_config_dict = {
                    'project_name': f"{config.project_name}_best",
                    'hyperParameters': {
                        'lr': study.best_params['lr'],
                        'optimizer_args': {'weight_decay': study.best_params['weight_decay']},
                        'epochs': 100,  # 用于最终训练
                    },
                    'netAttributes': {
                        'firstUpdateLayers': study.best_params['firstUpdateLayers'],
                        'secondUpdateLayers': study.best_params['secondUpdateLayers'],
                        'hidden_features': study.best_params['hidden_features'],
                        'batch_size': study.best_params.get('batch_size', config.batch_size),
                        'dropout_rate': study.best_params['dropout_rate'],
                    }
                }
                
                with open(best_config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(best_config_dict, f, default_flow_style=False, allow_unicode=True)
                print(f"🔧 最佳配置保存至: {best_config_file}")
            except:
                print("⚠️ 无法生成YAML配置文件")
        else:
            print(f"\n❌ 没有成功完成的试验")
            print(f"🔧 建议检查:")
            print(f"   - 数据路径: {config.dataset_path}")
            print(f"   - GPU内存是否充足")
            print(f"   - epochs是否太少: {config.epochs}")
        
        if storage_name and os.path.exists(db_path):
            print(f"💾 数据库保存至: {db_path}")
        
        print(f"{'='*60}")