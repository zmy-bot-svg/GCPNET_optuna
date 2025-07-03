#!/usr/bin/python
# -*- encoding: utf-8 -*-
# 处理晶体结构数据的脚本
'''
@File    :   dataset_utils.py
@Time    :   2023/05/05 01:41:00
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.eddu.com
'''
# 设置调试模式标志，用于控制数据集大小和调试信息输出
debug = True
# debug = False

# 导入警告模块并忽略CIF文件解析时的特定警告信息
import warnings
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")

# 导入PyTorch核心库，用于张量操作和深度学习
import torch

# 导入日志模块，用于记录数据处理过程中的信息
import logging
# 导入路径操作模块，用于文件路径处理
import os.path as osp
# 导入NumPy库，用于数值计算和数组操作
import numpy as np
# 导入进度条库，用于显示数据处理进度
from tqdm import tqdm
# 导入PyTorch Geometric的内存数据集基类和数据类
from torch_geometric.data import InMemoryDataset,Data
# 导入辅助函数模块，包含图构建和特征生成相关功能
from utils.helpers import (
    clean_up,                        # 清理数据中的临时属性
    generate_edge_features,          # 生成边特征（如径向基函数）
    generate_node_features,          # 生成节点特征
    get_cutoff_distance_matrix,      # 获取截断距离矩阵
)
# 导入稠密张量到稀疏张量的转换函数
from torch_geometric.utils import dense_to_sparse
# 导入变换组合类，用于链式应用多个数据变换
from torch_geometric.transforms import Compose
# 导入自定义变换类，用于提取目标属性
from utils.transforms import GetY
# 导入PyTorch Geometric的标准变换模块
import torch_geometric.transforms as T

# 定义MP18数据集类，继承自InMemoryDataset，用于处理材料属性预测数据集
# pyg的InMemoryDataset类用于存储和处理图数据，提供了高效的数据加载和预处理功能 
class MP18(InMemoryDataset):

    # 初始化方法，设置数据集的各种参数
    def __init__(self, root='data/', name='MP18', transform=None, pre_transform=[GetY()], r=8.0, n_neighbors=12, edge_steps=50, image_selfloop=True, points=100,target_name="formation_energy_per_atom"):
        
        # 将数据集名称转换为小写并验证是否支持
        self.name = name.lower()
        assert self.name in ['mp18', 'pt','2d','mof','surface','cubic', 'cif']
        # 设置截断半径，用于确定原子间的邻居关系
        self.r = r
        # 设置每个原子的最大邻居数量
        self.n_neighbors = n_neighbors
        # 设置边特征的径向基函数步数
        self.edge_steps = edge_steps
        # 设置是否为图像添加自环（自连接）
        self.image_selfloop = image_selfloop
        # 设置使用的数据点数量，用于限制数据集大小
        self.points = points# dataset snumbers
        # 设置目标属性名称，如形成能、带隙等
        self.target_name = target_name # target property name
        # 设置计算设备为CPU
        # 改进：如果有GPU可用，可以改为torch.device('cuda')以加速计算
        # 自动检测并使用最佳设备
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        # 调用父类初始化方法
        super(MP18, self).__init__(root, transform, pre_transform)
        # 加载预处理后的数据，weights_only=False代表加载完整数据而非仅权重
        # 将：
        # self.data, self.slices = torch.load(self.processed_paths[0])
        # 改为：
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # 属性方法：返回原始数据目录路径
    # property装饰器可以让函数像属性一样被访问，无需括号
    @property  
    def raw_dir(self):
        # 如果是CIF格式，返回空字符串（文件在根目录）
        if self.name == 'cif':
            return ''
        else:
            # 其他格式返回对应的原始数据目录
            return osp.join(self.root, self.name, 'raw')
    
    # 属性方法：返回处理后数据的存储目录路径
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    # 属性方法：返回原始数据文件名列表
    @property
    def raw_file_names(self):
        # 根据数据集类型返回对应的原始文件名
        if self.name == 'mp18':
            file_names = ['mp.2018.6.1.json.zip']
        elif self.name == 'pt':
            file_names = ['pt.2023.5.19.json.zip']
        elif self.name == 'mof':
            file_names = ['mof.2023.5.19.json.zip']
        elif self.name == '2d':
            file_names = ['2d.2023.5.19.json.zip']
        elif self.name == 'surface':
            file_names = ['surface.2023.5.19.json.zip']
        elif self.name == 'cubic':
            file_names = ['cubic.2023.7.13.json.zip']
        elif self.name == 'cif':
            # 对于CIF格式，使用glob模式匹配所有.cif文件
            # 查找并返回所有.cif文件的路径列表
            from glob import glob
            file_names = glob(f"{self.root}/*.cif")
        else:
            # 不支持的数据集类型，退出程序
            exit(1)
        return file_names

    # 属性方法：返回处理后数据文件名（将处理结果保存到哪里以及从哪里加载缓存），包含所有参数信息用于缓存
    @property
    def processed_file_names(self):
        # 根据所有参数生成唯一的处理文件名，便于缓存和版本控制
        # format方法将各个参数插入到文件名中
        processed_name = 'data_{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)
        return [processed_name]

    # 数据处理主方法，将原始数据转换为PyTorch Geometric格式
    def process(self):
        # info级别的日志记录，输出数据处理开始的信息,eg：INFO - Data found at data/mp18/raw
        logging.info("Data found at {}".format(self.raw_dir))
        # 1. 从JSON/CIF文件中读取结构数据并转换为字典格式
        # json_wrap方法将原始数据转换为字典格式的结构数据
        dict_structures, y = self.json_wrap()

        # 2. 将字典格式的结构数据转换为PyTorch Geometric的Data对象列表
        data_list = self.get_data_list(dict_structures, y) # 获取列表

        # 3. 保存处理后的数据到磁盘
        data, slices = self.collate(data_list) 
        torch.save((data, slices), self.processed_paths[0])
        logging.info("Processed data saved successfully.")
    
    # 字符串表示方法，返回数据集的参数信息
    def __str__(self):
        return '{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)
    
    # 对象表示方法，返回数据集名称
    def __repr__(self):
        return '{}()'.format(self.name)
    
    # 将Pymatgen结构对象转换为ASE原子对象的方法，ase进行原子操作是最高效的
    def pymatgen2ase(self,pymat_structure):
        from pymatgen.io.ase import AseAtomsAdaptor
        # 创建Pymatgen到ASE的适配器
        Adaptor = AseAtomsAdaptor()
        # 执行转换并返回ASE原子对象
        return Adaptor.get_atoms(pymat_structure)

    # 从JSON文件中读取和解析晶体结构数据的方法
    def json_wrap(self):
        import pandas as pd
        import os
        logging.info("Reading individual structures using Pymatgen.")

        # 导入Pymatgen的结构类，用于处理晶体结构
        from pymatgen.core import Structure
        # 根据数据集类型选择不同的数据读取方式
        # 如果数据集是CIF格式，则逐个读取CIF文件内容
        if self.name.lower() in ['cif']:
            # 处理CIF文件：逐个读取CIF文件内容
            cifFiles = []
            for i in self.raw_paths:
                with open(i, 'r') as f:
                    strContent = f.read()
                cifFiles.append(strContent)
            # 提取文件名作为材料ID，basename获取文件名（只要路径中的文件名部分）
            # split('.')获取文件名的前半部分xx.cif的xx
            ids = [os.path.basename(i).split('.')[0] for i in self.raw_paths]
            # 创建DataFrame，设置默认属性值为0.0
            df = pd.DataFrame({'structure': cifFiles, 'material_id': ids, 'property': [.0]*len(ids)})
        else:
            # 处理JSON格式：直接读取JSON文件
            df  = pd.read_json(self.raw_paths[0]) 
        # 把括号中的内容加入日志中
        logging.info("Converting data to standardized form(dict format) for downstream processing.")

        # 初始化存储转换后结构的列表
        dict_structures = []
        # 遍历DataFrame中的每个结构
        for i, s in enumerate(tqdm(df["structure"])):
            # 限制数据集大小，如果达到指定点数则停止
            if i == self.points:  # limit the dataset size
                break
            # 从CIF字符串创建Pymatgen结构对象
            s = Structure.from_str(s, fmt="cif") 
            # 将Pymatgen结构转换为ASE原子对象
            s = self.pymatgen2ase(s)
            # 创建结构数据字典
            d = {}
            # 提取原子位置坐标（N×3矩阵）
            pos = torch.tensor(s.get_positions(), dtype=torch.float)  
            # 提取晶胞参数（3×3矩阵）
            cell = torch.tensor(
                np.array(s.get_cell()), dtype=torch.float
            ) # lattice vector 3*3 
            # 提取原子序数
            atomic_numbers = torch.LongTensor(s.get_atomic_numbers())

            # 对于cubic数据集，添加A/B位点信息（钙钛矿结构特有）
            if self.name == 'cubic':
                # 获取A位点和B位点元素，A位点通常是阳离子，B位点通常是过渡金属，目的是区分不同类型的原子
                def getAB(element):
                    # 根据元素类型分配A位点(7)、B位点(8)或其他(9)
                    if df['A'][i] == element:
                        return 7
                    elif df['B'][i] == element:
                        return 8
                    else:
                        return 9
                # 为每个原子分配A/B位点标签
                d["AB"] = torch.LongTensor([getAB(i)  for i in s.get_chemical_symbols()])

            # 存储基本结构信息
            d["positions"] = pos
            d["cell"] = cell
            d["atomic_numbers"] = atomic_numbers
            d["structure_id"] = str(df['material_id'][i])

            # 生成GATGNN风格的全局特征
            # 将Pymatgen结构对象转换为ASE原子对象后，提取原子序数
            _atoms_index     = s.get_atomic_numbers()
            from utils.helpers import create_global_feat
            # 创建基于原子组成的全局特征向量
            gatgnn_glob_feat = create_global_feat(_atoms_index)
            # 将全局特征复制到每个原子（每个原子都携带相同的全局信息）
            gatgnn_glob_feat = np.repeat(gatgnn_glob_feat,len(_atoms_index),axis=0) # 作用？
            d["gatgnn_glob_feat"] = torch.Tensor(gatgnn_glob_feat).float()

            # 将处理后的结构添加到列表中
            dict_structures.append(d)

            # 提取目标属性值
            y = df[[self.target_name]].to_numpy()

            # 编译结构大小（原子数）和元素组成信息，用于统计分析
            if i == 0:
                length = [len(_atoms_index)]
                elements = [list(set(_atoms_index))]
            else:
                length.append(len(_atoms_index))
                elements.append(list(set(_atoms_index)))
            # 记录最大原子数
            n_atoms_max = max(length)
        # 统计所有出现的元素种类
        species = list(set(sum(elements, [])))
        species.sort()
        num_species = len(species)
        # 打印数据集统计信息
        print(
            "Max structure size: ", # Maximum number of unit cell atoms in the dataset
            n_atoms_max,
            "Max number of elements: ", # The number of distinct elements contained in the data set
            num_species,
        )
        return dict_structures, y
    
    # 将字典格式的结构数据转换为PyTorch Geometric Data对象列表
    def get_data_list(self, dict_structures, y):
        
        # 获取结构数量
        n_structures = len(dict_structures)
        # 为每个结构创建空的Data对象
        data_list = [Data() for _ in range(n_structures)]

        logging.info("Getting torch_geometric.data.Data() objects.")

        # 遍历每个结构字典，构建图数据
        for i, sdict in enumerate(tqdm(dict_structures)):
            # 获取目标值
            target_val = y[i]
            # 获取当前Data对象
            data = data_list[i]

            # 提取结构基本信息
            pos = sdict["positions"]
            cell = sdict["cell"]
            atomic_numbers = sdict["atomic_numbers"]
            structure_id = sdict["structure_id"]

            # 基于截断距离构建邻接矩阵和偏移向量
            cd_matrix, cell_offsets = get_cutoff_distance_matrix(
                pos,                              # 原子坐标
                cell,                             # 晶胞参数
                self.r,                           # 截断半径
                self.n_neighbors,                 # 最大邻居数
                image_selfloop=self.image_selfloop, # 是否包含自环
                device=self.device,               # 计算设备
            )

            # 将稠密邻接矩阵转换为稀疏表示（边索引和边权重）
            edge_indices, edge_weights = dense_to_sparse(cd_matrix) 

            # 设置Data对象的各种属性
            data.n_atoms = len(atomic_numbers)                    # 原子数量
            data.pos = pos                                        # 原子坐标
            data.cell = cell                                      # 晶胞参数
            data.y = torch.Tensor(np.array([target_val]))         # 目标属性值
            data.z = atomic_numbers                               # 原子序数
            # 新增下面这行，用原子序数索引作为特征
            data.x = (data.z - 1).view(-1, 1).float() 
            
            # 如果是cubic数据集，添加A/B位点信息
            if self.name == 'cubic':
                data.AB = sdict["AB"]
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...]) # 位移向量（暂时设为零）
            data.edge_index, data.edge_weight = edge_indices, edge_weights # 边索引和权重
            data.cell_offsets = cell_offsets                      # 晶胞偏移

            # 创建边描述符字典
            data.edge_descriptor = {}

            # 存储距离信息到边描述符
            data.edge_descriptor["distance"] = edge_weights
            data.distances = edge_weights
            # 存储结构ID信息
            data.structure_id = [[structure_id] * len(data.y)]

            # 存储全局特征
            data.glob_feat   = sdict["gatgnn_glob_feat"]

        # 生成节点特征（基于原子类型、坐标等）
        logging.info("Generating node features...")
        generate_node_features(data_list, self.n_neighbors, device=self.device)

        # 生成边特征（基于距离的径向基函数等）
        logging.info("Generating edge features...")
        generate_edge_features(data_list, self.edge_steps, self.r, device=self.device)

        # 应用预变换（非即时变换）
        logging.debug("Applying transforms.")

        # 确保GetY变换存在，这对下游模型是必需的
        assert self.pre_transform[0].__class__.__name__ == "GetY", "The target transform GetY is required in pre_ptransform."

        # 组合所有预变换
        composition = Compose(self.pre_transform)

        # 对每个数据对象应用变换
        for data in data_list:
            composition(data)

        # 清理临时属性，释放内存
        clean_up(data_list, ["edge_descriptor"])

        return data_list
    

# 导入PyTorch Geometric的数据加载器
from torch_geometric.loader import DataLoader

# 数据集分割函数，将数据集分为训练集、验证集和测试集
def dataset_split(
    dataset,
    train_size: float = 0.8,     # 训练集比例
    valid_size: float = 0.05,    # 验证集比例
    test_size: float = 0.15,     # 测试集比例
    seed: int = 1234,            # 随机种子，确保可重复性
    debug=True,                  # 调试模式标志
):     
    import logging
    # 检查分割比例是否合理，如果不等于1则使用默认分割
    if train_size + valid_size + test_size != 1:
        import warnings
        warnings.warn("Invalid sizes detected. Using default split of 80/5/15.")
        train_size, valid_size, test_size = 0.8, 0.05, 0.15

    # 获取数据集总大小
    dataset_size = len(dataset)

    # 计算各部分的样本数量
    train_len = int(train_size * dataset_size)
    valid_len = int(valid_size * dataset_size)
    test_len = int(test_size * dataset_size)
    # 如果不是调试模式，使用固定的数据集大小（用于性能测试）
    if debug==False:
        train_len = 60000
        valid_len =5000
        test_len = 4239
    # 计算未使用的样本数量
    unused_len = dataset_size - train_len - valid_len - test_len
    # 导入随机分割函数
    from torch.utils.data import random_split
    # 执行随机分割，使用指定的随机种子
    (train_dataset, val_dataset, test_dataset, unused_dataset) = random_split(
        dataset,
        [train_len, valid_len, test_len, unused_len],
        generator=torch.Generator().manual_seed(seed),
    )
    # 打印分割结果统计信息
    print(
      "train length:",
      train_len,
      "val length:",
      valid_len,
      "test length:",
      test_len,
      "unused length:",
      unused_len,
      "seed :",
      seed,
    )
    return train_dataset, val_dataset, test_dataset

# 创建数据加载器的函数，用于批量加载训练、验证和测试数据
def get_dataloader(
    train_dataset,   val_dataset,  test_dataset , batch_size: int, num_workers: int = 0,pin_memory=False
):

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,      # 批大小
        shuffle=True,               # 训练时打乱数据顺序
        num_workers=num_workers,    # 数据加载工作进程数
        pin_memory=pin_memory,      # 是否将数据固定在内存中
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,      # 批大小
        shuffle=False,              # 验证时不打乱数据顺序
        num_workers=num_workers,    # 数据加载工作进程数
        pin_memory=pin_memory,      # 是否将数据固定在内存中

    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,      # 批大小
        shuffle=False,              # 测试时不打乱数据顺序
        num_workers=num_workers,    # 数据加载工作进程数
        pin_memory=pin_memory,      # 是否将数据固定在内存中
    )

    return train_loader, val_loader, test_loader 

# 交叉验证数据分割函数，将数据集分成k折用于交叉验证
def split_data_CV(dataset, num_folds=5, seed=666, save=False):
    # 获取数据集总大小
    dataset_size = len(dataset)
    # 计算每折的大小
    fold_length = int(dataset_size / num_folds)
    # 计算不能整除的剩余样本数
    unused_length = dataset_size - fold_length * num_folds
    # 创建每折的大小列表
    folds = [fold_length for i in range(num_folds)]
    # 添加剩余样本作为额外一折
    folds.append(unused_length)
    # 使用随机分割创建交叉验证数据集
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    # 打印分割信息
    print("fold length :", fold_length, "unused length:", unused_length, "seed", seed)
    # 返回前num_folds个折（不包括剩余样本）
    return cv_dataset[0:num_folds]

# 交叉验证数据加载器设置函数，为指定折创建训练和测试加载器
def loader_setup_CV(index, batch_size, dataset,  num_workers=0):
    # 分割数据集：将除了指定索引外的所有折作为训练集
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    # 连接所有训练折
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    # 指定索引的折作为测试集
    test_dataset = dataset[index]

    # 初始化加载器变量
    train_loader = val_loader = test_loader = None
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,      # 批大小
        shuffle=True,               # 训练时打乱数据
        num_workers=num_workers,    # 工作进程数
        pin_memory=True,            # 固定内存
    )
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,      # 批大小
        shuffle=False,              # 测试时不打乱数据
        num_workers=num_workers,    # 工作进程数
        pin_memory=True,)           # 固定内存

    return train_loader, test_loader, train_dataset, test_dataset

# 主程序入口，用于测试数据集类的功能
if __name__ == "__main__":
    # 创建MP18数据集实例，使用pt数据集进行测试
    dataset = MP18(root="data",name='pt',transform=None, r=8.0, n_neighbors=12, edge_steps=50, image_selfloop=True, points=100,target_name="property")
    # 设置数据加载器
    if debug:
        # 在调试模式下进行数据集分割
        train_dataset, val_dataset, test_dataset = dataset_split( dataset, train_size=0.8,valid_size=0.15,test_size=0.05,seed=666)   
        # 创建数据加载器，批大小为64，24个工作进程
        train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, 64,24)
