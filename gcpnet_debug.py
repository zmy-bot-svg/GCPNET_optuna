#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
GCPNet 训练问题全面诊断与修复工具
解决 "Processing..." 后停止不训练的问题
"""

import os
import sys
import time
import psutil
import logging
import torch
import numpy as np
from tqdm import tqdm
import threading
import signal

class GCPNetDebugger:
    def __init__(self, dataset_name='jarvis_fe_15k', data_root='./data'):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.process_monitor = None
        self.monitoring = False
        
    def setup_logging(self):
        """设置详细的日志记录"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gcpnet_debug.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def monitor_system_resources(self):
        """监控系统资源使用情况"""
        def monitor():
            while self.monitoring:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                logging.info(f"内存使用: {memory_mb:.2f} MB, CPU使用: {cpu_percent:.1f}%")
                
                # 内存使用超过8GB时警告
                if memory_mb > 8000:
                    logging.warning(f"内存使用过高: {memory_mb:.2f} MB")
                
                time.sleep(10)  # 每10秒检查一次
        
        self.monitoring = True
        self.process_monitor = threading.Thread(target=monitor, daemon=True)
        self.process_monitor.start()
        
    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring = False
        
    def check_data_files(self):
        """检查数据集文件完整性"""
        print("=== 检查数据集文件 ===")
        
        dataset_path = os.path.join(self.data_root, self.dataset_name.lower())
        raw_path = os.path.join(dataset_path, 'raw')
        
        expected_files = {
            'jarvis_fe_15k': ['jarvis_fe_15k.2023.5.19.json.zip'],
            'mp18': ['mp.2018.6.1.json.zip'],
            'pt': ['pt.2023.5.19.json.zip'],
            '2d': ['2d.2023.5.19.json.zip'],
            'mof': ['mof.2023.5.19.json.zip'],
            'cubic': ['cubic.2023.7.13.json.zip']
        }
        
        if not os.path.exists(raw_path):
            logging.error(f"原始数据目录不存在: {raw_path}")
            return False
            
        if self.dataset_name.lower() in expected_files:
            for file_name in expected_files[self.dataset_name.lower()]:
                file_path = os.path.join(raw_path, file_name)
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024*1024)
                    logging.info(f"✅ 找到数据文件: {file_path} ({size_mb:.2f} MB)")
                    
                    # 检查文件是否可读
                    try:
                        with open(file_path, 'rb') as f:
                            f.read(1024)  # 读取前1KB测试
                        logging.info("文件可正常读取")
                    except Exception as e:
                        logging.error(f"文件读取失败: {e}")
                        return False
                else:
                    logging.error(f"❌ 数据文件不存在: {file_path}")
                    return False
        
        return True
        
    def test_pymatgen_parsing(self):
        """测试Pymatgen解析功能"""
        print("=== 测试Pymatgen解析 ===")
        
        try:
            import pandas as pd
            from pymatgen.core import Structure
            
            # 读取JSON文件
            raw_path = os.path.join(self.data_root, self.dataset_name.lower(), 'raw')
            json_files = [f for f in os.listdir(raw_path) if f.endswith('.json.zip')]
            
            if not json_files:
                logging.error("未找到JSON文件")
                return False
                
            json_path = os.path.join(raw_path, json_files[0])
            logging.info(f"读取JSON文件: {json_path}")
            
            df = pd.read_json(json_path)
            logging.info(f"JSON包含 {len(df)} 条记录")
            
            # 测试解析前几个结构
            success_count = 0
            for i in range(min(5, len(df))):
                try:
                    cif_str = df["structure"].iloc[i]
                    structure = Structure.from_str(cif_str, fmt="cif")
                    logging.info(f"结构 {i}: {len(structure)} 个原子, 化学式: {structure.composition.reduced_formula}")
                    success_count += 1
                except Exception as e:
                    logging.warning(f"解析结构 {i} 失败: {e}")
            
            if success_count > 0:
                logging.info(f"✅ 成功解析 {success_count}/5 个结构")
                return True
            else:
                logging.error("❌ 所有结构解析失败")
                return False
                
        except Exception as e:
            logging.error(f"Pymatgen测试失败: {e}")
            return False
            
    def create_minimal_dataset_test(self):
        """创建最小数据集测试"""
        print("=== 创建最小数据集测试 ===")
        
        test_code = """
import torch
import logging
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# 添加路径
sys.path.append('.')

from utils.dataset_utils import MP18

def test_minimal_dataset():
    logging.basicConfig(level=logging.INFO)
    
    print("开始最小数据集测试...")
    
    try:
        # 使用极小的数据集
        dataset = MP18(
            root='./data',
            name='{}',
            points=3,  # 只处理3个数据点
            r=6.0,     # 减小截断半径
            n_neighbors=8,  # 减少邻居数
            edge_steps=20,  # 减少边特征维度
            target_name='formation_energy_per_atom'
        )
        
        print(f"数据集创建成功，大小: {{len(dataset)}}")
        
        if len(dataset) > 0:
            data = dataset[0]
            print(f"第一个数据点:")
            print(f"  原子数: {{data.n_atoms}}")
            print(f"  坐标形状: {{data.pos.shape}}")
            print(f"  边数量: {{data.edge_index.shape[1]}}")
            print(f"  目标值: {{data.y}}")
            
            # 检查数据类型
            print(f"  坐标类型: {{data.pos.dtype}}")
            print(f"  原子序数类型: {{data.z.dtype}}")
            
            return True
        else:
            print("❌ 数据集为空")
            return False
            
    except Exception as e:
        print(f"❌ 数据集测试失败: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_dataset()
    sys.exit(0 if success else 1)
""".format(self.dataset_name)
        
        with open('minimal_test.py', 'w', encoding='utf-8') as f:
            f.write(test_code)
            
        logging.info("✅ 创建了最小测试脚本: minimal_test.py")
        return True
        
    def create_step_by_step_debug(self):
        """创建逐步调试脚本"""
        debug_code = """
import torch
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append('.')

def debug_step_by_step():
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== 逐步调试数据处理 ===")
    
    # 步骤1: 读取JSON
    try:
        raw_path = './data/{}/raw'
        json_files = [f for f in os.listdir(raw_path) if f.endswith('.json.zip')]
        json_path = os.path.join(raw_path, json_files[0])
        
        print(f"读取文件: {{json_path}}")
        df = pd.read_json(json_path)
        print(f"✅ 成功读取 {{len(df)}} 条记录")
        
        # 检查数据列
        print(f"数据列: {{list(df.columns)}}")
        
        # 检查目标列
        if 'formation_energy_per_atom' in df.columns:
            print("✅ 找到formation_energy_per_atom列")
        else:
            print("❌ 未找到formation_energy_per_atom列")
            print("可用的数值列:", [col for col in df.columns if df[col].dtype in [np.float64, np.int64]])
            
    except Exception as e:
        print(f"❌ 读取JSON失败: {{e}}")
        return
    
    # 步骤2: 测试结构解析
    print("\\n=== 测试结构解析 ===")
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    
    adaptor = AseAtomsAdaptor()
    
    for i in range(min(3, len(df))):
        try:
            print(f"处理结构 {{i}}...")
            cif_str = df["structure"].iloc[i]
            
            # Pymatgen解析
            structure = Structure.from_str(cif_str, fmt="cif")
            print(f"  Pymatgen结构: {{len(structure)}} 原子")
            
            # 转换为ASE
            ase_atoms = adaptor.get_atoms(structure)
            print(f"  ASE原子: {{len(ase_atoms)}} 原子")
            
            # 获取基本属性
            positions = ase_atoms.get_positions()
            cell = ase_atoms.get_cell()
            atomic_numbers = ase_atoms.get_atomic_numbers()
            
            print(f"  坐标形状: {{positions.shape}}")
            print(f"  晶胞形状: {{cell.shape}}")
            print(f"  原子序数: {{len(atomic_numbers)}}")
            
        except Exception as e:
            print(f"  ❌ 结构 {{i}} 解析失败: {{e}}")
            continue
    
    # 步骤3: 测试图构建
    print("\\n=== 测试图构建 ===")
    try:
        sys.path.append('./utils')
        from helpers import get_cutoff_distance_matrix
        
        # 使用第一个成功的结构
        cif_str = df["structure"].iloc[0]
        structure = Structure.from_str(cif_str, fmt="cif")
        ase_atoms = adaptor.get_atoms(structure)
        
        pos = torch.tensor(ase_atoms.get_positions(), dtype=torch.float)
        cell = torch.tensor(np.array(ase_atoms.get_cell()), dtype=torch.float)
        
        print(f"构建图结构，原子数: {{len(pos)}}")
        
        cd_matrix, cell_offsets = get_cutoff_distance_matrix(
            pos, cell, r=6.0, n_neighbors=8, device='cpu'
        )
        
        print(f"✅ 距离矩阵形状: {{cd_matrix.shape}}")
        print(f"✅ 非零边数: {{torch.nonzero(cd_matrix).shape[0]}}")
        
    except Exception as e:
        print(f"❌ 图构建失败: {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()
""".format(self.dataset_name)
        
        with open('debug_steps.py', 'w', encoding='utf-8') as f:
            f.write(debug_code)
            
        logging.info("✅ 创建了逐步调试脚本: debug_steps.py")
        
    def create_optimized_config(self):
        """创建优化的配置文件"""
        optimized_config = """
config:
  project_name: "GCPNet_debug"
  net: "GCPNet"
  output_dir: "./output"
  self_loop: True
  n_neighbors: 8                    # 减少邻居数
  debug: True                       # 启用调试模式

netAttributes:
  firstUpdateLayers: 2              # 减少层数
  secondUpdateLayers: 2             # 减少层数
  atom_input_features: 89           # 对应 n_neighbors=8: 80+8+1=89
  edge_input_features: 20           # 减少边特征维度
  triplet_input_features: 40
  embedding_features: 32            # 减少嵌入维度
  hidden_features: 16               # 减少隐藏层维度
  output_features: 1
  min_edge_distance: 0.0
  max_edge_distance: 6.0            # 减小截断距离
  link: "identity"
  batch_size: 16                    # 减小批次大小
  num_workers: 0                    # 单进程调试
  dropout_rate: 0.0

hyperParameters:
  lr: 0.001
  optimizer: "AdamW"
  optimizer_args:
    weight_decay: 0.00001
  scheduler: "ReduceLROnPlateau"
  scheduler_args:
    mode: "min"
    factor: 0.8
    patience: 10
    min_lr: 0.00001
    threshold: 0.0002
  seed: 666
  epochs: 10                        # 减少训练轮数
  patience: 5                       # 减少早停耐心

data:
  points: 50                        # 只使用50个数据点
  dataset_path: './data'
  dataset_name: '{}'
  target_index: 2
  target_name: 'formation_energy_per_atom'
  pin_memory: False                 # 关闭pin_memory
  num_folds: 5

predict:
  model_path: 'model.pt'
  output_path: 'output.csv'

visualize_args:
  perplexity: 50
  early_exaggeration: 12
  learning_rate: 300
  n_iter: 1000                      # 减少迭代次数
  verbose: 1
  random_state: 42

wandb:
  log_enable: False                 # 关闭wandb调试时
  sweep_count: 5
  entity: "1548532425-null"
  sweep_args:
    method: random
    parameters:
      lr: 
        distribution: log_uniform_values
        min: 0.000001
        max: 0.1
""".format(self.dataset_name)
        
        with open('config_debug.yml', 'w', encoding='utf-8') as f:
            f.write(optimized_config)
            
        logging.info("✅ 创建了优化配置文件: config_debug.yml")
        
    def run_comprehensive_diagnosis(self):
        """运行全面诊断"""
        print("=== GCPNet 全面诊断开始 ===")
        
        # 设置日志和监控
        self.setup_logging()
        self.monitor_system_resources()
        
        try:
            # 1. 检查数据文件
            if not self.check_data_files():
                logging.error("数据文件检查失败，请确认数据集完整性")
                return False
                
            # 2. 测试Pymatgen解析
            if not self.test_pymatgen_parsing():
                logging.error("Pymatgen解析测试失败")
                return False
                
            # 3. 创建测试脚本
            self.create_minimal_dataset_test()
            self.create_step_by_step_debug()
            self.create_optimized_config()
            
            print("\\n=== 诊断完成，建议执行步骤 ===")
            print("1. 运行最小测试: python minimal_test.py")
            print("2. 运行逐步调试: python debug_steps.py") 
            print("3. 使用优化配置训练: python main.py --config_file ./config_debug.yml --task_type train")
            print("4. 监控内存使用，确保不超过系统限制")
            print("5. 如果仍有问题，请检查 gcpnet_debug.log 日志文件")
            
            return True
            
        except KeyboardInterrupt:
            logging.info("用户中断诊断")
            return False
        except Exception as e:
            logging.error(f"诊断过程出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.stop_monitoring()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='GCPNet问题诊断工具')
    parser.add_argument('--dataset', default='jarvis_fe_15k', help='数据集名称')
    parser.add_argument('--data_root', default='./data', help='数据根目录')
    
    args = parser.parse_args()
    
    debugger = GCPNetDebugger(args.dataset, args.data_root)
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print("\\n正在停止诊断...")
        debugger.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    success = debugger.run_comprehensive_diagnosis()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()