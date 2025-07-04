import torch
import os
import tempfile
import inspect
import yaml
from argparse import Namespace # 导入Namespace，用于将字典转换为对象

# 导入您项目中需要的模块
from utils.dataset_utils import MP18
from utils.transforms import GetAngle, ToFloat
from torch_geometric.transforms import Compose

# 这是一个独立的诊断脚本，它会提前找出所有需要被添加到PyTorch安全列表的自定义类

def find_custom_classes(module):
    """动态查找一个模块中定义的所有类"""
    custom_classes = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            custom_classes[name] = obj
    return custom_classes

def run_check():
    print("🚀 开始执行PyTorch序列化预检脚本...")
    
    # --- 1. 直接读取配置文件，不再使用Flags类 ---
    print(" step 1/4: 直接加载config4090.yml配置文件...")
    try:
        config_path = 'config.yml' # 直接指定配置文件
        if not os.path.exists(config_path):
            print(f"❌ 错误：找不到配置文件 '{config_path}'。请确保脚本和配置文件在同一目录下。")
            return
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # 将YAML中所有层级的字典合并，并转换为一个可以用“.”访问的对象
        merged_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                merged_dict.update(value)
            else:
                merged_dict[key] = value
        config = Namespace(**merged_dict)

        # 使用加载的配置创建数据集
        dataset = MP18(root=config.dataset_path, name=config.dataset_name, transform=Compose([GetAngle(), ToFloat(
        )]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True, points=config.points, target_name=config.target_name)
    
    except Exception as e:
        print(f"❌ 错误：加载配置或数据集时失败。")
        print(f"   具体错误: {e}")
        return

    # --- 2. 动态查找model.py中所有的自定义类 ---
    print(" step 2/4: 动态查找model.py中的所有自定义类...")
    import model as model_module
    all_custom_classes = find_custom_classes(model_module)
    if not all_custom_classes:
        print("❌ 错误：在model.py中没有找到任何自定义类。")
        return
    print(f"   🔍 发现的自定义类: {list(all_custom_classes.keys())}")


    # --- 3. 循环尝试，直到找出所有必需的类 ---
    print(" step 3/4: 循环测试模型加载，以识别所有必需的类...")
    
    # 准备模型和临时文件路径
    from main import setup_model
    net = setup_model(dataset, config)
    temp_file = tempfile.mktemp(suffix=".pth")
    
    required_classes = {}
    
    for i in range(len(all_custom_classes) + 1):
        try:
            if required_classes:
                torch.serialization.add_safe_globals(list(required_classes.values()))
            
            torch.save(net, temp_file)
            _ = torch.load(temp_file, weights_only=False)
            
            print(f"   ✅ 在第 {i+1} 次尝试时成功加载！")
            break

        except Exception as e:
            error_str = str(e)
            if "Unsupported global" in error_str:
                missing_class_name = error_str.split("model.")[-1].split(" ")[0].strip()
                if missing_class_name in all_custom_classes and missing_class_name not in required_classes:
                    print(f"   [识别出] 缺失的类 [第 {i+1} 轮]: {missing_class_name}")
                    required_classes[missing_class_name] = all_custom_classes[missing_class_name]
                else:
                    print(f"❌ 致命错误：无法解决的序列化问题或遇到重复错误。")
                    print(f"   原始报错: {e}")
                    return
            elif "AttributeError" in error_str and "add_safe_globals" in error_str:
                 print(f"✅ 当前PyTorch版本较旧，不需要进行序列化检查。")
                 required_classes = {}
                 break
            else:
                print(f"❌ 致命错误：发生了非序列化相关的错误。")
                print(f"   原始报错: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return

    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    # --- 4. 生成最终的修复代码 ---
    print(" step 4/4: 生成最终修复方案...")
    if not required_classes:
        print("\n🎉 检查完成！您的环境似乎不需要任何修复，或者已经修复。")
    else:
        class_names = list(required_classes.keys())
        print("\n" + "="*60)
        print("🎉 检查完成！已找到所有需要添加到安全列表的类。")
        print("请将以下完整的代码块，复制并粘贴到您的 `main_4090.py` 文件中，")
        print("放在所有 import 语句的最后面，以替换掉之前的修复代码。")
        print("="*60)
        
        print("\n# --- 开始复制 ---")
        print(f"from model import {', '.join(class_names)}")
        print(f"""
# 修复PyTorch 2.6+的序列化错误
try:
    torch.serialization.add_safe_globals([{', '.join(class_names)}])
except AttributeError:
    pass  # 旧版PyTorch不需要此设置
""")
        print("# --- 结束复制 ---\n")

if __name__ == "__main__":
    run_check()