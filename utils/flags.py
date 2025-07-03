# 导入命令行参数解析库
import argparse
# 导入YAML文件解析库，用于读取配置文件，比如config.yml
import yaml

class Flags:
    """
    配置参数管理类
    用于解析命令行参数和YAML配置文件，将两者合并为统一的配置对象
    支持命令行参数覆盖配置文件中的默认值
    """
    def __init__(self):
        '''
        初始化配置管理器
        完整的初始化流程：
        1. 创建命令行参数解析器
        2. 添加必需的命令行参数
        3. 解析命令行参数
        4. 读取并解析YAML配置文件
        5. 用命令行参数更新配置文件中的默认值
        6. 合并所有配置为最终的配置对象
        '''
        # 创建ArgumentParser实例，用于解析命令行参数
        self.parser = argparse.ArgumentParser(description="MatPlat inputs")
        
        # 添加必需的命令行参数定义
        self.add_required_args()
        
        # 解析命令行参数，返回已知参数和未知参数
        # args: 预定义的参数（如task_type, config_file）
        # unknown: 用户额外传入的参数，用于覆盖配置文件中的值
        args, unknown = self.get_args()
        
        # 解析YAML配置文件，获取默认配置
        default_config = self.parse_yml(args.config_file)
        
        # 使用命令行中的未知参数更新默认配置
        # 这允许用户通过命令行覆盖配置文件中的任何参数
        updated_config = self.update_from_args(unknown, default_config)

        # 将必需的命令行参数添加到最终配置中
        # 遍历args中的所有属性，添加到updated_config
        # setattr()函数用于动态设置对象的属性，key是属性名，val是属性值
        # 循环最后得到的updated_config包含了：
        # - 从YAML文件加载的默认配置
        # - 用户通过命令行传入的参数（如果有）
        # - 必需的命令行参数（如task_type, config_file）
        for key, val in vars(args).items():
            setattr(updated_config, key, val)
        
        # 保存最终的合并配置，供外部访问
        self.updated_config = updated_config

    def add_required_args(self):
        '''
        添加必需的命令行参数
        这些参数是程序运行的基础，必须由用户提供
        '''

        # 添加任务类型参数，决定程序执行的主要功能
        self.parser.add_argument(
            "--task_type",
            choices=["train", "test", "predict", "visualize", "hyperparameter", "CV"],  # 限制可选值
            required=True,  # 设为必需参数
            type=str,
            help="Type of task to perform: train, test, predict, visualize, hyperparameter",)
        
        # 添加配置文件路径参数，指定YAML配置文件的位置
        self.parser.add_argument(
            "--config_file",
            required=True,  # 设为必需参数
            type=str,
            help="Default parameters for training",
            default='./config.yml'  # 默认配置文件路径
        )

    def get_args(self):
        """
        解析命令行参数
        使用parse_known_args()而不是parse_args()，
        这样可以处理未预定义的参数而不报错
        
        Returns:
            args: 包含预定义参数的Namespace对象
            unknown: 未知参数的列表，格式为['--param1', 'value1', '--param2', 'value2']
        """
        # parse_known_args()返回两个值：
        # 1. 已知参数的Namespace对象
        # 2. 未知参数的字符串列表
        args, unknown = self.parser.parse_known_args()
        return args, unknown

    def parse_yml(self, yml_file):
        """
        解析YAML配置文件并将嵌套结构扁平化
        
        Args:
            yml_file: YAML配置文件的路径
            
        Returns:
            config: 包含所有配置参数的argparse.Namespace对象
        """
        
        def recursive_flatten(nestedConfig):
            """
            递归函数：将嵌套的字典结构扁平化
            特殊处理：如果键以'_args'结尾，保持其字典结构不变
            这通常用于保存复杂的配置参数（如优化器参数、调度器参数等）
            
            Args:
                nestedConfig: 嵌套的配置字典
            """
            # 遍历嵌套配置的所有键值对
            for k, v in nestedConfig.items():
                # 如果值是字典类型
                if isinstance(v, dict):
                    # 检查键是否以'_args'结尾
                    if k.split('_')[-1] == 'args':
                        # 如果是args类型，保持字典结构不变
                        # 这样可以保持optimizer_args、scheduler_args等的完整结构
                        flattenConfig[k] = v
                    else:
                        # 否则递归展开这个字典
                        recursive_flatten(v)
                else:
                    # 如果值不是字典，直接添加到扁平化配置中
                    flattenConfig[k] = v

        # 初始化扁平化配置字典
        flattenConfig = {}
        
        # 打开并读取YAML文件
        with open(yml_file, 'r', encoding='utf-8') as f:
            # 使用FullLoader加载YAML内容为嵌套字典
            nestedConfig = yaml.load(f, Loader=yaml.FullLoader)
        
        # 调用递归函数进行扁平化处理
        recursive_flatten(nestedConfig)

        # 创建argparse.Namespace对象，用于统一的属性访问方式
        config = argparse.Namespace()
        
        # 将扁平化的字典转换为Namespace对象的属性
        for key, val in flattenConfig.items():
            setattr(config, key, val)
            
        return config
    
    def update_from_args(self, unknown_args, ymlConfig):
        """
        使用命令行参数更新YAML配置
        允许用户通过命令行覆盖配置文件中的任何参数
        
        Args:
            unknown_args: 未知参数列表，格式为['--param1', 'value1', '--param2', 'value2']
            ymlConfig: 从YAML文件解析的配置对象
            
        Returns:
            ymlConfig: 更新后的配置对象
        """
        # 检查未知参数的数量是否为偶数
        # 因为每个参数都应该有对应的值：--key value
        assert len(unknown_args) % 2 == 0, f"Please Check Arguments, {' '.join(unknown_args)}"
        
        # 遍历参数对：每次取两个元素（参数名和参数值）
        # unknown_args[0::2] 获取所有偶数位置的元素（参数名）
        # unknown_args[1::2] 获取所有奇数位置的元素（参数值）
        for key, val in zip(unknown_args[0::2], unknown_args[1::2]):
            # 去除参数名前面的'--'前缀
            key = key.strip("--")
            
            # 解析参数值，尝试转换为合适的Python类型
            val = self.parse_value(val)
            
            # 设置或更新配置对象的属性
            # 这会覆盖YAML文件中的同名参数
            setattr(ymlConfig, key, val)
            
        return ymlConfig

    def parse_value(self, value):
        """
        智能解析字符串值，尝试转换为合适的Python数据类型
        
        这个函数尝试将字符串转换为Python字面量（数字、布尔值、列表等）
        如果转换失败，则保持为字符串类型
        
        Args:
            value: 待解析的字符串值
            
        Returns:
            解析后的值（可能是int、float、bool、list、dict或str）
            
        Examples:
            "123" -> 123 (int)
            "3.14" -> 3.14 (float) 
            "True" -> True (bool)
            "[1,2,3]" -> [1,2,3] (list)
            "hello" -> "hello" (str)
        """
        # 导入抽象语法树模块，用于安全地解析Python字面量
        import ast
        
        # 使用ast.literal_eval进行安全的字面量解析
        # 这比eval()更安全，只能解析字面量，不能执行任意代码
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # 如果解析失败（不是有效的Python字面量），
            # 则将其作为字符串返回
            return value
