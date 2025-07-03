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
def train(config, printnet=False):
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
    
    # 开始训练模型
    history = model.fit(
        train_loader,
        val_loader,
        ckpt_path=os.path.join(config.output_dir, config.net+'.pth'),
        epochs=config.epochs,
        monitor='val_loss',
        mode='min',
        patience=config.patience,
        plot=True,
        callbacks=callbacks
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
    
    # 定义超参数搜索空间（基于你的 config.yml 中的设置）
    config.lr = trial.suggest_float("lr", 0.0002, 0.0015, log=True)
    config.dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3)
    
    # weight_decay 需要修改 optimizer_args
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
    config.optimizer_args = config.optimizer_args.copy()
    config.optimizer_args['weight_decay'] = weight_decay
    
    config.firstUpdateLayers = trial.suggest_categorical("firstUpdateLayers", [3, 4, 5])
    config.secondUpdateLayers = trial.suggest_categorical("secondUpdateLayers", [3, 4, 5])
    config.hidden_features = trial.suggest_categorical("hidden_features", [96, 128])
    
    # 关闭 wandb 日志（由 Optuna 管理）
    config.log_enable = False
    
    # 为每个 trial 创建唯一的输出目录
    trial_name = f"trial_{trial.number}"
    config.output_dir = os.path.join(base_config.output_dir, trial_name)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # 执行训练并返回最佳验证 MAE
    try:
        best_val_mae = train(config, printnet=False)
        return best_val_mae
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')  # 返回一个很大的值表示失败

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
    elif config.task_type.lower() == 'hyperparameter':
        print("Starting Optuna hyperparameter optimization...")
        
        # 创建 TensorBoard 日志目录用于可视化
        tensorboard_dir = os.path.join(config.output_dir, "tensorboard_logs")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        
        # 创建 Optuna study，使用贝叶斯优化
        study = optuna.create_study(
            direction="minimize",  # 最小化验证 MAE
            study_name=config.project_name,
            sampler=optuna.samplers.TPESampler(seed=config.seed),  # 使用 TPE 贝叶斯优化
            # callbacks=[TensorBoardCallback(tensorboard_dir, metric_name='val_mae')]  # TensorBoard 回调
        )
        
        # 定义包装的目标函数
        def wrapped_objective(trial):
            return objective(trial, config)
        
        # 开始优化，n_trials 是总的试验次数
        print(f"Running {config.sweep_count} trials...")
        study.optimize(
            wrapped_objective,
            n_trials=config.sweep_count,
            callbacks=[TensorBoardCallback(tensorboard_dir, metric_name='val_mae')]
        )
        
        # 打印优化结果
        print("="*60)
        print("Hyperparameter optimization finished!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value (min val_mae): {study.best_value:.6f}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        print("="*60)
        
        # 保存优化结果
        results_file = os.path.join(config.output_dir, "optuna_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best value: {study.best_value:.6f}\n")
            f.write("Best parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Results saved to: {results_file}")
        print(f"TensorBoard logs saved to: {tensorboard_dir}")
        print(f"To view TensorBoard: tensorboard --logdir {tensorboard_dir}")
    
    elif config.task_type.lower() == 'visualize':
        visualize(config)
    
    elif config.task_type.lower() == 'cv':
        log_file = config.project_name + '.log'
        log_config(log_file)
        train_CV(config)
    
    elif config.task_type.lower() == 'predict':
        predict(config)
    
    else:
        raise NotImplementedError(f"Task type {config.task_type} not implemented. Supported types: train, test, cv, predict")