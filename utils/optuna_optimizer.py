#!/usr/bin/env python3
"""
Optuna超参数优化器模块
将超参数搜索逻辑模块化，增强可视化功能
"""

import os
import sys
import time
import copy
import datetime
import gc
import signal
import psutil
import torch
import optuna
import optuna.visualization as vis
from optuna.exceptions import TrialPruned


class OptunaHyperparameterOptimizer:
    """Optuna超参数优化器"""
    
    def __init__(self, config, train_function):
        """
        初始化优化器
        
        Args:
            config: 配置对象
            train_function: 训练函数
        """
        self.config = config
        self.train_function = train_function
        self.study = None
        self.storage_name = None
        self.db_path = None
        
    def setup_storage(self):
        """设置数据库存储"""
        try:
            # 确保输出目录存在
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # 检查写入权限
            if not os.access(self.config.output_dir, os.W_OK):
                raise PermissionError(f"输出目录无写入权限: {self.config.output_dir}")
            
            # 创建数据库文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            db_name = f"{self.config.project_name}_{timestamp}.db"
            self.db_path = os.path.abspath(os.path.join(self.config.output_dir, db_name))
            
            # 兼容不同操作系统的存储URL
            self.storage_name = f"sqlite:///{self.db_path}"
            
            print(f"📁 数据库路径: {self.db_path}")
            
            # 测试数据库连接
            test_study = optuna.create_study(
                direction="minimize",
                study_name=f"test_{timestamp}",
                storage=self.storage_name
            )
            print("✅ 数据库连接成功")
            
            # 清理测试study
            try:
                optuna.delete_study(study_name=f"test_{timestamp}", storage=self.storage_name)
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ 数据库设置失败: {e}")
            print("🔄 使用内存存储模式")
            self.storage_name = None
            self.db_path = None
    
    def create_study(self):
        """创建Optuna study"""
        try:
            self.study = optuna.create_study(
                direction="minimize",
                study_name=self.config.project_name,
                storage=self.storage_name,
                sampler=optuna.samplers.TPESampler(
                    seed=self.config.seed,
                    n_startup_trials=max(2, self.config.sweep_count // 5)
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=2,
                    n_warmup_steps=3,
                    interval_steps=1
                ),
                load_if_exists=True  # 支持断点续传
            )
            print("✅ Optuna study创建成功")
            
            # 如果是续传，显示已有试验信息
            existing_trials = len(self.study.trials)
            if existing_trials > 0:
                print(f"🔄 检测到已有试验: {existing_trials}个")
                print(f"📊 将从第{existing_trials + 1}个试验继续...")
                
        except Exception as e:
            print(f"❌ Study创建失败: {e}")
            self.study = optuna.create_study(direction="minimize")
            print("🔄 使用基础配置")
    
    def get_search_space_params(self, trial):
        """定义搜索空间参数"""
        params = {}
        
        # 基础超参数
        params['lr'] = trial.suggest_float("lr", 0.0005, 0.003, log=True)
        params['dropout_rate'] = trial.suggest_float("dropout_rate", 0.05, 0.25)
        params['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # 网络结构
        params['firstUpdateLayers'] = trial.suggest_categorical("firstUpdateLayers", [3, 4, 5])
        params['secondUpdateLayers'] = trial.suggest_categorical("secondUpdateLayers", [3, 4, 5])
        params['hidden_features'] = trial.suggest_categorical("hidden_features", [96, 128, 160, 192])
        
        # 动态batch size调整
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory > 40:  # A100等
                batch_options = [64, 96, 128, 160, 192]
            elif gpu_memory > 20:  # V100, A6000等
                batch_options = [48, 64, 96, 128]
            elif gpu_memory > 10:  # 4090, 3090等
                batch_options = [32, 48, 64, 96]
            else:  # 小显存卡
                batch_options = [16, 24, 32, 48]
            params['batch_size'] = trial.suggest_categorical("batch_size", batch_options)
        else:
            params['batch_size'] = trial.suggest_categorical("batch_size", [16, 24, 32])
        
        return params
    
    def apply_params_to_config(self, trial_config, params):
        """将参数应用到配置对象"""
        # 基础参数
        trial_config.lr = params['lr']
        trial_config.dropout_rate = params['dropout_rate']
        
        # 优化器参数
        trial_config.optimizer_args = trial_config.optimizer_args.copy()
        trial_config.optimizer_args['weight_decay'] = params['weight_decay']
        
        # 网络结构
        trial_config.firstUpdateLayers = params['firstUpdateLayers']
        trial_config.secondUpdateLayers = params['secondUpdateLayers']
        trial_config.hidden_features = params['hidden_features']
        trial_config.batch_size = params['batch_size']
        
        # CPU workers优化
        if hasattr(psutil, 'cpu_count'):
            cpu_cores = psutil.cpu_count(logical=False)
            max_workers = min(cpu_cores // 2, 8)
            trial_config.num_workers = min(trial_config.num_workers, max_workers)
        
        return trial_config
    
    def objective_function(self, trial):
        """优化目标函数"""
        trial_start_time = time.time()
        
        # 设置信号处理
        def cleanup_handler(signum, frame):
            print(f"\n🛑 Trial #{trial.number} 收到信号 {signum}，正在清理...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.exit(1)
        
        original_sigint = signal.signal(signal.SIGINT, cleanup_handler)
        original_sigterm = signal.signal(signal.SIGTERM, cleanup_handler)
        
        try:
            # 系统资源检查
            if hasattr(psutil, 'virtual_memory'):
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    print(f"⚠️ 系统内存使用过高: {memory.percent:.1f}%")
                    return float('inf')
            
            # 清理资源
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 深拷贝配置
            trial_config = copy.deepcopy(self.config)
            
            # 获取搜索参数
            params = self.get_search_space_params(trial)
            
            # 应用参数到配置
            trial_config = self.apply_params_to_config(trial_config, params)
            
            # 显示试验信息
            self.print_trial_info(trial, params, trial_config)
            
            # 配置调整
            trial_config.log_enable = False
            
            # 创建试验目录
            trial_name = f"trial_{trial.number:03d}"
            trial_config.output_dir = os.path.join(self.config.output_dir, trial_name)
            os.makedirs(trial_config.output_dir, exist_ok=True)
            
            # 权限检查
            if not os.access(trial_config.output_dir, os.W_OK):
                raise PermissionError(f"试验目录无写入权限: {trial_config.output_dir}")
            
            # 设置随机种子
            trial_config.seed = self.config.seed + trial.number * 42
            self.set_seed(trial_config.seed)
            
            # 执行训练
            best_val_mae = self.train_function(trial_config, printnet=False, trial=trial)
            
            # 显示结果
            self.print_trial_results(trial, best_val_mae, trial_start_time)
            
            return best_val_mae
            
        except TrialPruned:
            trial_time = time.time() - trial_start_time
            print(f"\n✂️ Trial #{trial.number} 被剪枝")
            print(f"⏱️  节省时间: {trial_time/60:.1f}分钟")
            raise
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n💥 Trial #{trial.number} GPU内存不足")
            suggested_batch = max(16, params.get('batch_size', 32) // 2)
            print(f"🔧 建议batch_size: {suggested_batch}")
            return float('inf')
            
        except Exception as e:
            trial_time = time.time() - trial_start_time
            print(f"\n❌ Trial #{trial.number} 失败")
            print(f"⚠️  错误: {str(e)[:100]}...")
            print(f"⏱️  耗时: {trial_time/60:.1f}分钟")
            return float('inf')
        
        finally:
            # 恢复信号处理器
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            
            # 最终清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def print_trial_info(self, trial, params, trial_config):
        """打印试验信息"""
        print(f"\n{'='*60}")
        print(f"🚀 Trial #{trial.number} 开始")
        
        if hasattr(psutil, 'virtual_memory'):
            memory = psutil.virtual_memory()
            print(f"🖥️  资源状态:")
            print(f"    内存使用: {memory.percent:.1f}%")
            
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"    GPU显存: {gpu_memory:.1f}GB")
            
        print(f"⏱️  预计用时: {self.config.epochs * 2:.1f}分钟")
        print("📋 超参数:")
        for key, value in params.items():
            print(f"    {key}: {value}")
        print(f"{'='*60}")
    
    def print_trial_results(self, trial, best_val_mae, trial_start_time):
        """打印试验结果"""
        trial_time = time.time() - trial_start_time
        
        print(f"\n🏁 Trial #{trial.number} 完成!")
        print(f"⏱️  耗时: {trial_time/60:.1f}分钟")
        print(f"📈 验证MAE: {best_val_mae:.6f}")
        
        if hasattr(psutil, 'virtual_memory'):
            memory = psutil.virtual_memory()
            print(f"💾 内存使用: {memory.percent:.1f}%")
        
        # 显示排名
        try:
            completed_trials = [t for t in trial.study.trials 
                              if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) > 1:
                sorted_values = sorted([t.value for t in completed_trials])
                current_rank = sorted_values.index(best_val_mae) + 1
                print(f"🏆 当前排名: {current_rank}/{len(completed_trials)}")
                print(f"🥇 最佳成绩: {trial.study.best_value:.6f}")
        except:
            pass
        
        print(f"{'='*60}")
    
    def set_seed(self, seed):
        """设置随机种子"""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def generate_visualizations(self):
        """生成可视化文件"""
        if self.study is None or len(self.study.trials) == 0:
            print("⚠️ 没有可用的试验数据，跳过可视化")
            return
        
        try:
            vis_dir = os.path.join(self.config.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            print("\n📊 生成可视化文件...")
            
            # 1. 优化历史
            try:
                fig1 = vis.plot_optimization_history(self.study)
                fig1.write_html(f"{vis_dir}/optimization_history.html")
                print(f"   ✅ 优化历史: {vis_dir}/optimization_history.html")
            except Exception as e:
                print(f"   ❌ 优化历史生成失败: {e}")
            
            # 2. 参数重要性
            try:
                if len(self.study.trials) >= 10:  # 需要足够的试验数据
                    fig2 = vis.plot_param_importances(self.study)
                    fig2.write_html(f"{vis_dir}/param_importances.html")
                    print(f"   ✅ 参数重要性: {vis_dir}/param_importances.html")
            except Exception as e:
                print(f"   ❌ 参数重要性生成失败: {e}")
            
            # 3. 平行坐标图
            try:
                if len(self.study.trials) >= 5:
                    fig3 = vis.plot_parallel_coordinate(self.study)
                    fig3.write_html(f"{vis_dir}/parallel_coordinate.html")
                    print(f"   ✅ 平行坐标图: {vis_dir}/parallel_coordinate.html")
            except Exception as e:
                print(f"   ❌ 平行坐标图生成失败: {e}")
            
            # 4. 参数关系
            try:
                if len(self.study.trials) >= 10:
                    fig4 = vis.plot_slice(self.study)
                    fig4.write_html(f"{vis_dir}/param_slice.html")
                    print(f"   ✅ 参数切片图: {vis_dir}/param_slice.html")
            except Exception as e:
                print(f"   ❌ 参数切片图生成失败: {e}")
            
        except Exception as e:
            print(f"⚠️ 可视化生成过程出错: {e}")
    
    def print_system_info(self):
        """打印系统信息"""
        print("🖥️  系统信息:")
        print(f"   操作系统: {os.name}")
        
        if hasattr(psutil, 'cpu_count'):
            print(f"   CPU核心: {psutil.cpu_count(logical=False)} 物理, {psutil.cpu_count(logical=True)} 逻辑")
            
        if hasattr(psutil, 'virtual_memory'):
            memory = psutil.virtual_memory()
            print(f"   内存: {memory.total / 1e9:.1f}GB")
            
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    def save_results(self, optimization_time):
        """保存优化结果"""
        completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed:
            print("\n❌ 没有成功完成的试验")
            return
        
        best_trial = self.study.best_trial
        
        # 保存文本结果
        results_file = os.path.join(self.config.output_dir, "best_hyperparameters.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"# 超参数优化结果\n")
            f.write(f"# 优化时间: {datetime.datetime.now()}\n")
            f.write(f"# 系统: {os.name}\n")
            if torch.cuda.is_available():
                f.write(f"# GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"\n最佳Trial: #{best_trial.number}\n")
            f.write(f"最佳验证MAE: {self.study.best_value:.6f}\n")
            f.write(f"优化耗时: {optimization_time/3600:.2f}小时\n\n")
            f.write("最佳超参数:\n")
            for key, value in self.study.best_params.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n💾 结果保存至: {results_file}")
        
        # 生成运行脚本
        script_file = os.path.join(self.config.output_dir, "run_best_config.sh")
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# 运行最佳超参数配置\n\n")
            f.write("python main.py --config_file config.yml --task_type train \\\n")
            for key, value in self.study.best_params.items():
                f.write(f"  --{key} {value} \\\n")
            f.write("  --epochs 100\n")
        
        # 设置执行权限
        try:
            os.chmod(script_file, 0o755)
            print(f"🔧 运行脚本: {script_file}")
        except:
            print(f"🔧 运行脚本: {script_file} (请手动设置执行权限)")
    
    def run(self):
        """执行完整的超参数优化流程"""
        print("Starting Optuna hyperparameter optimization...")
        
        # 打印系统信息
        self.print_system_info()
        
        # 设置存储
        self.setup_storage()
        
        # 创建study
        self.create_study()
        
        # 优化配置信息
        print(f"\n🎯 开始超参数优化")
        print(f"📊 计划试验数: {self.config.sweep_count}")
        print(f"💾 存储模式: {'SQLite数据库' if self.storage_name else '内存'}")
        if self.storage_name:
            print(f"📁 数据库: {self.db_path}")
        print(f"🔧 推荐配置:")
        print(f"   - 使用screen或tmux运行长时间任务")
        print(f"   - 监控: nvidia-smi, htop")
        print(f"{'='*60}")
        
        optimization_start = time.time()
        
        try:
            # 执行优化
            self.study.optimize(
                self.objective_function,
                n_trials=self.config.sweep_count,
                timeout=None,
                catch=(Exception,),
                show_progress_bar=True,
                gc_after_trial=True
            )
        except KeyboardInterrupt:
            print(f"\n⏹️ 用户中断优化 (Ctrl+C)")
        except Exception as e:
            print(f"\n💥 优化过程出错: {e}")
        
        # 结果分析
        optimization_time = time.time() - optimization_start
        
        print(f"\n{'='*60}")
        print(f"🎉 超参数优化完成!")
        print(f"⏱️  总耗时: {optimization_time/3600:.1f}小时")
        
        # 统计信息
        all_trials = self.study.trials
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
            best_trial = self.study.best_trial
            print(f"\n🏆 最佳结果:")
            print(f"   Trial: #{best_trial.number}")
            print(f"   验证MAE: {self.study.best_value:.6f}")
            print(f"   参数:")
            for key, value in self.study.best_params.items():
                print(f"     {key}: {value}")
            
            # 保存结果
            self.save_results(optimization_time)
            
        else:
            print(f"\n❌ 没有成功完成的试验")
            print(f"🔧 调试建议:")
            print(f"   - 检查数据路径: {self.config.dataset_path}")
            print(f"   - 检查GPU状态: nvidia-smi")
            print(f"   - 检查内存: free -h")
            print(f"   - 检查epochs设置: {self.config.epochs}")
        
        # 生成可视化
        self.generate_visualizations()
        
        if self.storage_name and self.db_path and os.path.exists(self.db_path):
            print(f"💾 数据库保存至: {self.db_path}")
            print(f"🔧 数据库权限: {oct(os.stat(self.db_path).st_mode)[-3:]}")
        
        print(f"{'='*60}")
        
        return self.study