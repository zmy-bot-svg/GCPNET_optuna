import sys
import os

print("-" * 50)
print(f"Python 版本: {sys.version}")

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"!!! PyTorch 导入失败: {e}")

try:
    import torch_geometric
    print(f"PyG 版本: {torch_geometric.__version__}")
except ImportError as e:
    print(f"!!! PyG 导入失败: {e}")

try:
    # 切换到包含编译模块的目录
    sys.path.insert(0, os.path.join(os.getcwd(), 'utils', 'functions'))
    # potnet_algorithm.py 也在 utils 目录中
    sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
    from potnet_algorithm import zeta, exp
    print("PotNet 算法模块导入成功!")
except ImportError as e:
    print(f"!!! PotNet 算法模块导入失败: {e}")

print("-" * 50)
print("环境完整性检查完毕。如果未出现'!!!'开头的失败信息，则环境配置成功!")