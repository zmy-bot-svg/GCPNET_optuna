# 该脚本为Keras风格的PyTorch训练框架
# 导入PyTorch核心库，用于深度学习模型构建和训练
import torch

# 导入系统模块和日期时间模块
# sys：用于系统相关功能，如标准输出重定向
# datetime：用于处理日期和时间，记录训练时间等
import sys,datetime

# 导入tqdm库，用于在循环中显示进度条，提升用户体验
from tqdm import tqdm 

# 导入deepcopy函数，用于创建对象的深度拷贝，避免对象引用问题
from copy import deepcopy

# 导入NumPy库，用于数值计算和数组操作
import numpy as np

# 导入pandas库，用于数据处理和分析，特别是DataFrame操作
import pandas as pd

# 再次导入torch（代码中重复导入，实际可以删除这行）
import torch

# 导入Accelerator，用于支持多GPU/TPU训练和混合精度训练
from accelerate import Accelerator


# 定义一个函数用于在控制台输出彩色文本，提升训练过程中信息的可读性
def colorful(obj,color="red", display_type="plain"):
    # 定义颜色代码字典，映射颜色名称到ANSI转义序列代码
    color_dict = {"black":"30", "red":"31", "green":"32", "yellow":"33",
                    "blue":"34", "purple":"35","cyan":"36",  "white":"37"}
    # 定义显示类型字典，映射显示效果到ANSI转义序列代码
    display_type_dict = {"plain":"0","highlight":"1","underline":"4",
                "shine":"5","inverse":"7","invisible":"8"}
    # 将输入对象转换为字符串
    s = str(obj)
    # 根据指定颜色获取对应的ANSI代码，如果颜色不存在则返回空字符串
    color_code = color_dict.get(color,"")
    # 根据指定显示类型获取对应的ANSI代码，如果类型不存在则返回空字符串
    display  = display_type_dict.get(display_type,"")
    # 构造完整的ANSI转义序列：开始标记 + 文本内容 + 结束标记
    out = '\033[{};{}m'.format(display,color_code)+s+'\033[0m'
    # 返回格式化后的彩色文本字符串
    return out 


# 定义StepRunner类，负责处理训练或验证过程中的单个步骤（一个batch的处理）
class StepRunner:
    # 初始化方法，设置训练步骤所需的各种组件
    def __init__(self, net, loss_fn, accelerator, stage = "train", metrics_dict = None,
                 optimizer = None, lr_scheduler = None
                 ):
        # 保存模型、损失函数、评估指标字典和当前阶段（训练或验证）
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        # 保存优化器和学习率调度器
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        # 保存Accelerator实例，用于多设备训练
        # 这是huggingface的一个库，自动选择gpu还是cpu（如果有多个卡用多个卡）并处理分布式训练
        self.accelerator = accelerator

    # 定义__call__方法，使类实例可以像函数一样被调用，处理单个batch的数据
    def __call__(self, batch):
        # 从batch中提取特征和标签
        # features是整个batch数据，labels是batch.y（目标值）
        # .y是PyTorch Geometric中数据对象的属性，这个模型中表示材料的目标性质
        features,labels = batch,batch.y
        
        # 前向传播：将特征输入模型得到预测结果
        preds = self.net(features)
        
        # 修复预测值和标签之间的维度不匹配问题
        # 这是处理不同数据格式时常见的问题
        # 疑问：如果不是这两种情况怎么办？难道输出层指定输出形状了吗
        if preds.dim() != labels.dim():
            # 如果预测值是1维而标签是2维，则压缩标签的最后一个维度
            if preds.dim() == 1 and labels.dim() == 2:
                labels = labels.squeeze(-1)
            # 如果预测值是2维而标签是1维，则压缩预测值的最后一个维度
            elif preds.dim() == 2 and labels.dim() == 1:
                preds = preds.squeeze(-1)
        
        # 计算损失函数：比较预测值和真实标签
        # 这个函数可能在# - main.py 中的 build_keras 函数
        # - utils/model_utils.py
        # - utils/loss_utils.py  定义了损失函数
        loss = self.loss_fn(preds,labels)

        # 反向传播和参数更新（仅在训练阶段执行）
        if self.optimizer is not None and self.stage=="train":
            # 使用accelerator进行反向传播，自动处理多设备情况
            self.accelerator.backward(loss)
            # 执行优化器步骤，更新模型参数
            self.optimizer.step()
            # 清零梯度，为下一次迭代做准备
            self.optimizer.zero_grad()

        # 在多设备训练中收集所有设备上的预测值、标签和损失
        # gather函数会收集所有GPU/TPU上的数据
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        # 构造当前步骤的损失字典，键名包含阶段信息（train_loss或val_loss）
        # stage返回当前的阶段（train或val），用于区分训练和验证损失
        step_losses = {self.stage+"_loss":all_loss.item()}

        # 计算并更新评估指标（如果提供了指标字典）
        if self.metrics_dict:
            # 确保预测值和标签的形状一致，处理维度问题
            # 如果预测值和标签是多维的，去掉最后一个维度
            preds_fixed = all_preds.squeeze(-1) if all_preds.dim() > 1 else all_preds
            labels_fixed = all_labels.squeeze(-1) if all_labels.dim() > 1 else all_labels

            # 更新每个指标的累积状态（用于epoch结束时的最终计算）
            for name, metric_fn in self.metrics_dict.items():
                metric_fn.update(preds_fixed, labels_fixed)

            # 计算当前步骤的指标值，构造指标字典
            # xx.item()将张量转换为Python标量,方便记录和显示，如果不转换可能会导致显示问题
            #计算时不用 .item() - 保持梯度和张量运算能力
            #显示时使用 .item() - 更清晰的数值显示
            #存储时使用 .item() - 节省内存和支持序列化
            step_metrics = {self.stage+"_"+name:metric_fn(preds_fixed, labels_fixed).item()
                        for name, metric_fn in self.metrics_dict.items()}
        else:
            # 如果没有提供指标字典，则返回空字典
            step_metrics = {}

        # 在训练阶段记录当前学习率
        if self.stage=="train":
            if self.optimizer is not None:
                # 从优化器的状态字典中获取当前学习率
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                # 如果没有优化器，学习率设为0
                step_metrics['lr'] = 0.0
        # 返回当前步骤的损失和指标
        return step_losses,step_metrics


# 定义EpochRunner类，负责处理完整epoch的训练或验证过程
class EpochRunner:
    # 初始化方法，接收steprunner实例和静默模式标志
    def __init__(self,steprunner,quiet=False):
        # 保存steprunner实例
        self.steprunner = steprunner
        # 获取当前阶段（训练或验证）
        self.stage = steprunner.stage
        # 根据阶段设置模型模式：训练模式或评估模式
        self.steprunner.net.train() if self.stage=="train" else self.steprunner.net.eval()
        # 保存accelerator实例
        self.accelerator = self.steprunner.accelerator
        # 保存静默模式标志，控制是否显示进度条
        self.quiet = quiet

    # 定义__call__方法，处理整个数据加载器的所有batch
    def __call__(self,dataloader):
        # 创建进度条，显示训练/验证进度
        loop = tqdm(enumerate(dataloader,start=1),  # 从1开始枚举
                    total =len(dataloader),          # 总步数
                    file=sys.stdout,                 # 输出到标准输出
                    disable=not self.accelerator.is_local_main_process or self.quiet,  # 仅主进程显示或根据quiet参数
                    ncols = 100                      # 进度条宽度
                   )
        # 初始化epoch级别的损失累积字典
        epoch_losses = {}
        # 遍历数据加载器中的每个batch
        for step, batch in loop:
            # 根据阶段决定是否使用梯度计算
            if self.stage=="train":
                # 训练阶段需要计算梯度
                step_losses,step_metrics = self.steprunner(batch)
            else:
                # 验证阶段不需要计算梯度，使用torch.no_grad()提高效率
                with torch.no_grad():
                    step_losses,step_metrics = self.steprunner(batch)

            # 合并当前步骤的损失和指标到一个字典中
            step_log = dict(step_losses,**step_metrics)
            # 将batch的损失值累积到epoch_losses中，k和v分别是键和值
            for k,v in step_losses.items():
                # xx.get(k,0.0) - 如果k不存在则返回0.0
                epoch_losses[k] = epoch_losses.get(k,0.0)+v

            # 更新进度条显示信息
            if step!=len(dataloader):
                # 如果不是最后一步，显示当前步骤的指标。loop作用是更新进度条
                loop.set_postfix(**step_log)
            else:
                # 如果是最后一步，计算epoch级别的指标
                epoch_metrics = step_metrics
                
                # 计算epoch结束时的累积指标，添加安全检查避免运行时错误
                if self.steprunner.metrics_dict:
                    try:
                        # 调用每个指标的compute方法获取epoch级别的结果
                        # stage包括train或val，name是metric_dict中的指标名称（应该只有mae和mape）
                        epoch_metrics.update({self.stage+"_"+name:metric_fn.compute().item()
                                         for name,metric_fn in self.steprunner.metrics_dict.items()})
                    except RuntimeError as e:
                        # 如果计算指标时出现形状问题，打印警告并跳过
                        print(f"Warning: Could not compute epoch metrics: {e}")
                        pass
                
                # 计算epoch级别的平均损失，k是损失名称，v是累积损失值（v/step: 该损失的epoch平均值）
                # 这里的step是当前epoch的总batch数
                epoch_losses = {k:v/step for k,v in epoch_losses.items()}
                # 合并损失和指标
                epoch_log = dict(epoch_losses,**epoch_metrics)
                # 更新进度条显示最终的epoch信息
                loop.set_postfix(**epoch_log)
                
                # 重置所有指标的状态，为下一个epoch做准备
                if self.steprunner.metrics_dict:
                    for name,metric_fn in self.steprunner.metrics_dict.items():
                        metric_fn.reset()
        # 返回epoch级别的日志信息
        return epoch_log


# 定义KerasModel类，这是一个仿照Keras风格的PyTorch模型包装器
class KerasModel(torch.nn.Module):
    
    # 类属性：指定使用的StepRunner和EpochRunner类
    StepRunner,EpochRunner = StepRunner,EpochRunner
    
    # 初始化方法，设置模型的各个组件
    def __init__(self,net,loss_fn,metrics_dict=None,optimizer=None,lr_scheduler = None):
        # 调用父类初始化方法
        super().__init__()
        # 保存网络模型、损失函数和指标字典（转换为ModuleDict以便GPU移动）
        self.net,self.loss_fn,self.metrics_dict = net, loss_fn, torch.nn.ModuleDict(metrics_dict) 
        # 设置优化器，如果未提供则使用默认的Adam优化器
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.net.parameters(), lr=1e-3)
        # 保存学习率调度器
        self.lr_scheduler = lr_scheduler
        # 标记是否从零开始训练
        self.from_scratch = True

    # 加载检查点的方法
    def load_ckpt(self, ckpt_path='checkpoint.pt'):
        # 加载保存的模型权重（weights_only=False允许加载完整模型）
        self.net= torch.load(ckpt_path, weights_only=False)  #修改过，之前self.net= torch.load(ckpt_path)
        # 标记不是从零开始训练
        self.from_scratch = False

    # 前向传播方法
    def forward(self, x):
        return self.net.forward(x)
    
    # 训练方法，这是模型训练的主要入口点
    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min",
            mixed_precision='no', callbacks=None, plot=True, quiet=True, trial=None):
        """
        训练模型的主要方法，现在支持Optuna剪枝功能
        
        参数:
            trial: Optuna trial对象，用于剪枝功能
        """
        
        # 将所有局部变量保存到实例字典中，方便后续访问
        self.__dict__.update(locals())
        
        # 初始化Accelerator，设置混合精度训练
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        
        # 获取设备信息
        device = str(self.accelerator.device)
        # 根据设备类型选择显示图标
        device_type = '🐌' if 'cpu' in device else '⚡️'
        # 打印设备信息
        self.accelerator.print(
            colorful("<<<<<< " + device_type + " " + device + " is used >>>>>>"))

        # 使用accelerator准备所有组件，这会自动处理多设备分布
        self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler)
        
        # 准备数据加载器
        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)
        
        # 初始化训练历史记录字典
        self.history = {}
        # 设置回调函数列表
        callbacks = callbacks if callbacks is not None else []
        
        # 如果需要绘图，添加可视化进度回调
        if plot == True:
            from utils.keras_callbacks import VisProgress
            callbacks.append(VisProgress(self))

        # 准备回调函数
        self.callbacks = self.accelerator.prepare(callbacks)
        
        # 在主进程中执行训练开始回调
        if self.accelerator.is_local_main_process:
            for callback_obj in self.callbacks:
                callback_obj.on_fit_start(model=self)
        
        # 确定开始epoch：从零开始或从检查点继续
        start_epoch = 1 if self.from_scratch else 0
        
        # 主训练循环
        for epoch in range(start_epoch, epochs + 1):
            # 记录epoch开始时间
            import time
            start_time = time.time()
            
            # 根据quiet参数确定是否静默运行
            should_quiet = False if quiet == False else (quiet == True or epoch > quiet)
            
            # 如果不是静默模式，打印epoch信息
            if not should_quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.accelerator.print("\n" + "==========" * 8 + "%s" % nowtime)
                self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs) + "\n")

            # 1. 训练阶段 -------------------------------------------------  
            # 创建训练步骤运行器
            train_step_runner = self.StepRunner(
                net=self.net,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                stage="train",
                metrics_dict=deepcopy(self.metrics_dict),  # 深拷贝避免状态污染
                optimizer=self.optimizer if epoch > 0 else None,    # epoch 0时不更新参数
                lr_scheduler=self.lr_scheduler if epoch > 0 else None
            )

            # 创建训练epoch运行器
            train_epoch_runner = self.EpochRunner(train_step_runner, should_quiet)
        
            # 初始化训练指标并执行训练
            train_metrics = {'epoch': epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            # 将训练指标添加到历史记录中
            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            # 如果当前进程是主进程，执行训练epoch结束回调
            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_train_epoch_end(model=self)

            # 2. 验证阶段 -------------------------------------------------
            if val_dataloader:
                # 创建验证步骤运行器
                val_step_runner = self.StepRunner(
                    net=self.net,
                    loss_fn=self.loss_fn,
                    accelerator=self.accelerator,
                    stage="val",
                    metrics_dict=deepcopy(self.metrics_dict)  # 验证阶段不需要优化器
                )
                # 创建验证epoch运行器
                val_epoch_runner = self.EpochRunner(val_step_runner, should_quiet)
                # 在no_grad环境下执行验证
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)
                    # 根据调度器类型进行学习率调整
                    if self.lr_scheduler.scheduler_type == "ReduceLROnPlateau":
                        # ReduceLROnPlateau需要监控指标
                        self.lr_scheduler.step(
                            metrics=val_metrics['val_mae'])
                    else:
                        # 其他调度器按epoch调整
                        self.lr_scheduler.step()
                        
                # 将验证指标依次添加到历史记录中
                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]

            # ========== 新增Optuna剪枝逻辑开始 ==========
            # 如果提供了trial对象，则进行剪枝检查
            if trial is not None and val_dataloader:
                # 获取验证MAE指标，如果不存在则使用monitor指标
                val_mae = val_metrics.get('val_mae', val_metrics.get(monitor, float('inf')))
                
                # 向Optuna报告当前epoch的验证指标
                trial.report(val_mae, epoch)
                
                # 检查是否应该剪枝（Optuna自动决定）
                if trial.should_prune():
                    if not should_quiet:
                        self.accelerator.print(colorful(
                            f"✂️  Trial被剪枝在epoch {epoch}, {monitor}: {val_mae:.6f}"))
                    # 抛出剪枝异常
                    import optuna
                    raise optuna.exceptions.TrialPruned()
            # ========== 新增Optuna剪枝逻辑结束 ==========

            # 3. 早停机制 -------------------------------------------------
            # 等待所有设备同步
            self.accelerator.wait_for_everyone()
            # 获取监控指标的历史值
            arr_scores = self.history[monitor]
            # 记录当前最佳指标值
            self.history['best_val_mae'] = self.history.get('best_val_mae', []) + [
                np.min(arr_scores) if mode == "min" else np.max(arr_scores)]

            # 找到最佳分数的索引位置
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)

            # 如果当前epoch获得了最佳分数，保存模型
            if best_score_idx == len(arr_scores) - 1:
                self.accelerator.save(self.net, ckpt_path)
                if not should_quiet:
                    self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor, arr_scores[best_score_idx])))

            # 记录epoch结束时间
            end_time = time.time()
            self.history['time'] = self.history.get('time', []) + [end_time - start_time]
        
            # 检查是否触发早停条件
            if len(arr_scores) - best_score_idx > patience:
                self.accelerator.print(colorful(
                    "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>>"
                ).format(monitor, patience))
                break
            
            # 执行验证epoch结束回调
            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_validation_epoch_end(model=self)
        
        # 训练结束后的处理
        if self.accelerator.is_local_main_process:
            # 将历史记录转换为DataFrame
            dfhistory = pd.DataFrame(self.history)
            self.accelerator.print(dfhistory)
            
            # 执行训练结束回调
            for callback_obj in self.callbacks:
                callback_obj.on_fit_end(model=self)
        
            # 解包模型并加载最佳权重
            self.net = self.accelerator.unwrap_model(self.net)
            self.net = torch.load(ckpt_path)
            return dfhistory

        
        # 模型评估方法，用于在验证集上评估模型性能
        @torch.no_grad()  # 装饰器，确保下面的代码都不计算梯度
        def evaluate(self, val_data):
            # 如果前面没有@torch.no_grad()，则这里需要加入一个with torch.no_grad():
            # 创建新的accelerator实例
            accelerator = Accelerator()
            # 准备模型组件
            self.net,self.loss_fn,self.metrics_dict = accelerator.prepare(self.net,self.loss_fn,self.metrics_dict)
            # 准备验证数据
            val_data = accelerator.prepare(val_data)
            # 创建验证步骤运行器
            val_step_runner = self.StepRunner(net = self.net,stage="val",
                        loss_fn = self.loss_fn,metrics_dict=deepcopy(self.metrics_dict),
                        accelerator = accelerator)
            # 创建验证epoch运行器
            val_epoch_runner = self.EpochRunner(val_step_runner)
            # 执行验证并返回结果
            val_metrics = val_epoch_runner(val_data)
            return val_metrics
        
        # 模型预测方法，用于在测试集上进行预测
        @torch.no_grad()  # 装饰器，确保不计算梯度
        def predict(self, test_data,ckpt_path, test_out_path='test_out.csv'):
            # 保存检查点路径
            self.ckpt_path = ckpt_path
            # 加载模型权重
            self.load_ckpt(self.ckpt_path)
            # 设置模型为评估模式
            self.net.eval()
            # 初始化结果存储列表
            targets = []   # 真实标签
            outputs = []   # 模型预测值
            id = []        # 样本ID
            
            # 遍历测试数据
            for data in test_data:
                with torch.no_grad():
                    # 将数据移动到GPU
                    data = data.to(torch.device('cuda'))
                    # 收集真实标签
                    targets.append(data.y.cpu().numpy().tolist())
                    # 进行预测
                    output = self.net(data)
                    # 收集预测结果
                    outputs.append(output.cpu().numpy().tolist())
                    # 收集样本ID
                    id += data.structure_id
            
            # 将嵌套列表展平
            targets = sum(targets, [])
            outputs = sum(outputs, [])
            id = sum(sum(id, []),[])
            
            # 导入CSV写入模块
            import csv
            # 组合数据行
            rows = zip(
                id,
                targets,
                outputs
            )
            # 写入CSV文件
            with open(test_out_path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for row in rows:
                    writer.writerow(row)

        # cubic方法，与predict方法相同，仅默认输出文件名有区别，可能用于特定数据集
        @torch.no_grad()  # 装饰器，确保不计算梯度
        def cubic(self, test_data,ckpt_path, test_out_path='cubic_out.csv'):
            # 保存检查点路径
            self.ckpt_path = ckpt_path
            # 加载模型权重
            self.load_ckpt(self.ckpt_path)
            # 设置模型为评估模式
            self.net.eval()
            # 初始化结果存储列表
            targets = []   # 真实标签
            outputs = []   # 模型预测值
            id = []        # 样本ID
            
            # 遍历测试数据
            for data in test_data:
                with torch.no_grad():
                    # 将数据移动到GPU
                    data = data.to(torch.device('cuda'))
                    # 收集真实标签
                    targets.append(data.y.cpu().numpy().tolist())
                    # 进行预测
                    output = self.net(data)
                    # 收集预测结果
                    outputs.append(output.cpu().numpy().tolist())
                    # 收集样本ID
                    id += data.structure_id
            
            # 将嵌套列表展平
            targets = sum(targets, [])
            outputs = sum(outputs, [])
            id = sum(sum(id, []),[])
            
            # 导入CSV写入模块
            import csv
            # 组合数据行
            rows = zip(
                id,
                targets,
                outputs
            )
            # 写入CSV文件
            with open(test_out_path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for row in rows:
                    writer.writerow(row)

        # 分析方法，使用t-SNE对训练好的模型特征进行可视化分析
        @torch.no_grad()  # 装饰器，确保不计算梯度
        def analysis(self, net_name,test_data, ckpt_path, tsne_args,tsne_file_path="tsne_output.png"):
            '''
            从训练好的模型中获取图特征并使用t-SNE进行分析
            '''
            # 导入必要的分析库
            from sklearn.decomposition import PCA      # 主成分分析
            from sklearn.manifold import TSNE         # t-SNE降维
            import matplotlib.pyplot as plt           # 绘图库
            
            # 初始化输入特征存储列表
            inputs = []
            # 定义钩子函数，用于捕获中间层的输入
            def hook(module, input, output):
                inputs.append(input)

            # 保存检查点路径并加载模型
            self.ckpt_path = ckpt_path
            self.load_ckpt(self.ckpt_path)
            # 设置模型为评估模式
            self.net.eval()
            
            # 根据网络名称注册前向钩子，捕获第一个线性层的输入
            if net_name in [ "ALIGNN","CLIGNN", "GCPNet"]:
                # 对于这些网络，钩子注册到fc层
                # fc层的输入是整个网络提取的最终特征
                self.net.fc.register_forward_hook(hook)
            else:
                # 对于其他网络，钩子注册到post_lin_list的第一层
                # post_lin_list[0]的输入是特征提取器的输出
                self.net.post_lin_list[0].register_forward_hook(hook)

            # 初始化目标值存储列表（仅适用于单索引目标）
            targets = []
            # 遍历测试数据
            for data in test_data:
                with torch.no_grad():
                    # 将数据移动到GPU
                    data = data.to(torch.device('cuda'))
                    # 收集目标值
                    targets.append(data.y.cpu().numpy().tolist())
                    # 进行前向传播（触发钩子函数）
                    _ = self.net(data)

            # 处理收集到的数据
            targets = sum(targets, [])                    # 展平目标值列表
            inputs = [i for sub in inputs for i in sub]   # 展平输入特征列表
            inputs = torch.cat(inputs)                    # 连接所有输入张量
            inputs = inputs.cpu().numpy()                 # 转换为numpy数组
            
            # 打印数据信息
            print("Number of samples: ", inputs.shape[0])
            print("Number of features: ", inputs.shape[1])

            # 开始t-SNE分析
            # 使用传入的参数创建t-SNE对象
            tsne = TSNE(**tsne_args)
            # 执行t-SNE降维
            tsne_out = tsne.fit_transform(inputs)

            # 创建散点图进行可视化
            fig, ax = plt.subplots()
            # 绘制散点图，颜色映射到目标值
            main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3,cmap='coolwarm')
            # 移除坐标轴标签和刻度
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            # 添加颜色条
            cbar = plt.colorbar(main, ax=ax)
            # 计算标准差用于设置颜色范围
            stdev = np.std(targets)
            # 设置颜色条的范围为均值±2倍标准差
            cbar.mappable.set_clim(
                np.mean(targets) - 2 * np.std(targets), np.mean(targets) + 2 * np.std(targets)
            )
            # 保存图片
            plt.savefig(tsne_file_path, format="png", dpi=600)
            # 显示图片
            plt.show()

        # 返回模型总参数数量的方法
        def total_params(self):
            return self.net.total_params()


# 学习率调度器包装类，封装PyTorch的学习率调度器
class LRScheduler:
    """PyTorch学习率调度器的包装类"""

    # 初始化方法
    def __init__(self, optimizer, scheduler_type, model_parameters):
        # 保存优化器
        self.optimizer = optimizer
        # 保存调度器类型
        self.scheduler_type = scheduler_type

        # 动态获取并创建对应的调度器实例
        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_type)(
            optimizer, **model_parameters
        )

        # 保存当前学习率
        self.lr = self.optimizer.param_groups[0]["lr"]

    # 类方法：从配置创建LRScheduler实例
    @classmethod        #有装饰器则下面的第一个参数cls是LRScheduler类
    def from_config(cls, optimizer, optim_config):
        # 从配置中提取调度器类型
        scheduler_type = optim_config["scheduler_type"]
        # 从配置中提取调度器参数
        scheduler_args = optim_config["scheduler_args"]
        # 对应init中的参数，分别是要调度的优化器、调度器类型和调度器参数
        # 创建并返回LRScheduler实例
        return cls(optimizer, scheduler_type, **scheduler_args)

    # 执行学习率调度步骤
    def step(self, metrics=None, epoch=None):
        # 如果是空调度器，直接返回
        if self.scheduler_type == "Null":
            return
        # 如果是ReduceLROnPlateau类型，需要传入监控指标
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                # 如果没有提供指标，抛出异常
                raise Exception("Validation set required for ReduceLROnPlateau.")
            # 使用指标进行调度
            self.scheduler.step(metrics)
        else:
            # 其他调度器直接步进
            self.scheduler.step()

        # 更新学习率属性为当前学习率
        self.update_lr()

    # 更新学习率属性的方法
    def update_lr(self):
        # 遍历优化器的所有参数组，更新学习率，optimizer.param_group是一个列表，包含每个参数组的学习率等信息
        for param_group in self.optimizer.param_groups:
            self.lr = param_group["lr"]
