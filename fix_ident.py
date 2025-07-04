#!/usr/bin/env python3
"""
专门修复train_utils.py缩进问题的脚本
"""

def check_and_fix_indentation():
    """检查并修复train_utils.py的缩进问题"""
    
    print("🔍 分析 utils/train_utils.py 的缩进问题...")
    
    with open('utils/train_utils.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查KerasModel类的方法
    print("\n📊 KerasModel类方法检查:")
    in_keras_model = False
    keras_model_methods = []
    
    for i, line in enumerate(lines, 1):
        if 'class KerasModel' in line:
            in_keras_model = True
            print(f"行 {i}: 找到KerasModel类定义")
            continue
        
        if in_keras_model:
            # 如果遇到下一个类定义，停止
            if line.strip().startswith('class ') and 'KerasModel' not in line:
                print(f"行 {i}: 遇到下一个类 {line.strip()}")
                break
            
            # 检查方法定义
            if 'def ' in line and not line.strip().startswith('#'):
                indent_count = len(line) - len(line.lstrip())
                method_name = line.strip().split('def ')[1].split('(')[0]
                keras_model_methods.append((i, method_name, indent_count))
                print(f"行 {i}: 方法 {method_name} - 缩进: {indent_count}个空格")
    
    print(f"\n📋 KerasModel类中找到 {len(keras_model_methods)} 个方法:")
    for line_num, method_name, indent in keras_model_methods:
        status = "✅" if indent == 4 else "❌"
        print(f"  {status} {method_name}: {indent}个空格缩进")
    
    # 检查是否有缩进错误
    wrong_indent_methods = [m for m in keras_model_methods if m[2] != 4]
    
    if wrong_indent_methods:
        print(f"\n🚨 发现 {len(wrong_indent_methods)} 个缩进错误的方法")
        
        # 修复缩进
        print("🔧 开始修复缩进...")
        
        new_lines = []
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # 检查是否是需要修复的方法行
            needs_fix = False
            for wrong_line, method_name, wrong_indent in wrong_indent_methods:
                if line_num == wrong_line:
                    needs_fix = True
                    break
            
            if needs_fix:
                # 修复方法定义行的缩进
                if 'def ' in line or '@torch.no_grad()' in line:
                    new_line = '    ' + line.lstrip()
                    new_lines.append(new_line)
                    print(f"  修复行 {line_num}: {line.strip()} -> 4个空格缩进")
                else:
                    new_lines.append(line)
            else:
                # 检查是否是方法内部的代码（需要相应调整缩进）
                if line.strip() and not line.strip().startswith('#'):
                    # 如果当前行缩进过深，且在错误方法的范围内
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent > 4 and any(line_num > wrong_line for wrong_line, _, _ in wrong_indent_methods):
                        # 检查是否在方法内部
                        in_wrong_method = False
                        for wrong_line, _, _ in wrong_indent_methods:
                            if line_num > wrong_line:
                                # 检查是否还在这个方法内部
                                for j in range(i+1, min(len(lines), i+20)):
                                    if lines[j].strip().startswith('def ') or lines[j].strip().startswith('class '):
                                        if j < i + 10:  # 如果很快遇到下一个方法，说明当前行在方法内部
                                            in_wrong_method = True
                                        break
                        
                        if in_wrong_method and current_indent == 8:
                            # 将8个空格改为8个空格（方法内部代码）
                            new_line = '        ' + line.lstrip()
                            new_lines.append(new_line)
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
        
        # 写入修复后的文件
        with open('utils/train_utils.py', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print("✅ 缩进修复完成")
        
        # 验证修复结果
        print("\n🔍 验证修复结果...")
        try:
            from utils.train_utils import KerasModel
            methods = ['evaluate', 'predict', 'cubic', 'analysis', 'total_params']
            missing = []
            for method in methods:
                if hasattr(KerasModel, method):
                    print(f"  ✅ {method} 方法存在")
                else:
                    print(f"  ❌ {method} 方法不存在")
                    missing.append(method)
            
            if missing:
                print(f"\n❌ 仍有问题，缺少方法: {missing}")
                return False
            else:
                print("\n🎉 所有方法都正确存在！")
                return True
                
        except Exception as e:
            print(f"\n❌ 导入测试失败: {e}")
            return False
    
    else:
        print("\n✅ 所有方法缩进都正确")
        return True

def manual_fix_if_needed():
    """如果自动修复失败，提供手动修复方案"""
    print("\n🔧 尝试手动修复方案...")
    
    with open('utils/train_utils.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到fit方法的结束位置
    fit_end_marker = 'return dfhistory'
    lrscheduler_start_marker = 'class LRScheduler:'
    
    if fit_end_marker in content and lrscheduler_start_marker in content:
        parts = content.split(fit_end_marker)
        if len(parts) >= 2:
            before_fit_end = parts[0] + fit_end_marker
            
            # 找到LRScheduler类的开始
            remaining = fit_end_marker.join(parts[1:])
            lr_parts = remaining.split(lrscheduler_start_marker)
            
            if len(lr_parts) >= 2:
                methods_part = lr_parts[0]
                after_lr = lrscheduler_start_marker + lrscheduler_start_marker.join(lr_parts[1:])
                
                # 重写methods部分，确保正确的缩进
                fixed_methods = '''

    # 模型评估方法，用于在验证集上评估模型性能
    @torch.no_grad()
    def evaluate(self, val_data):
        accelerator = Accelerator()
        self.net, self.loss_fn, self.metrics_dict = accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict)
        val_data = accelerator.prepare(val_data)
        
        val_step_runner = self.StepRunner(
            net=self.net, stage="val",
            loss_fn=self.loss_fn, 
            metrics_dict=deepcopy(self.metrics_dict),
            accelerator=accelerator
        )
        
        val_epoch_runner = self.EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics
        
    @torch.no_grad()
    def predict(self, test_data, ckpt_path, test_out_path='test_out.csv'):
        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        
        targets = []
        outputs = []
        id = []
        
        for data in test_data:
            data = data.to(torch.device('cuda'))
            targets.append(data.y.cpu().numpy().tolist())
            output = self.net(data)
            outputs.append(output.cpu().numpy().tolist())
            id += data.structure_id
        
        targets = sum(targets, [])
        outputs = sum(outputs, [])
        id = sum(sum(id, []), [])
        
        import csv
        rows = zip(id, targets, outputs)
        with open(test_out_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for row in rows:
                writer.writerow(row)

    @torch.no_grad()
    def cubic(self, test_data, ckpt_path, test_out_path='cubic_out.csv'):
        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        
        targets = []
        outputs = []
        id = []
        
        for data in test_data:
            data = data.to(torch.device('cuda'))
            targets.append(data.y.cpu().numpy().tolist())
            output = self.net(data)
            outputs.append(output.cpu().numpy().tolist())
            id += data.structure_id
        
        targets = sum(targets, [])
        outputs = sum(outputs, [])
        id = sum(sum(id, []), [])
        
        import csv
        rows = zip(id, targets, outputs)
        with open(test_out_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for row in rows:
                writer.writerow(row)

    @torch.no_grad()
    def analysis(self, net_name, test_data, ckpt_path, tsne_args, tsne_file_path="tsne_output.png"):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        inputs = []
        def hook(module, input, output):
            inputs.append(input)

        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        
        if net_name in ["ALIGNN", "CLIGNN", "GCPNet"]:
            self.net.fc.register_forward_hook(hook)
        else:
            self.net.post_lin_list[0].register_forward_hook(hook)

        targets = []
        for data in test_data:
            data = data.to(torch.device('cuda'))
            targets.append(data.y.cpu().numpy().tolist())
            _ = self.net(data)

        targets = sum(targets, [])
        inputs = [i for sub in inputs for i in sub]
        inputs = torch.cat(inputs)
        inputs = inputs.cpu().numpy()
        
        print("Number of samples: ", inputs.shape[0])
        print("Number of features: ", inputs.shape[1])

        tsne = TSNE(**tsne_args)
        tsne_out = tsne.fit_transform(inputs)

        fig, ax = plt.subplots()
        main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3, cmap='coolwarm')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(main, ax=ax)
        stdev = np.std(targets)
        cbar.mappable.set_clim(
            np.mean(targets) - 2 * np.std(targets), 
            np.mean(targets) + 2 * np.std(targets)
        )
        plt.savefig(tsne_file_path, format="png", dpi=600)
        plt.show()

    def total_params(self):
        return self.net.total_params()

'''
                
                # 重新组合文件
                new_content = before_fit_end + fixed_methods + '\n' + after_lr
                
                with open('utils/train_utils.py', 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print("✅ 手动修复完成")
                return True
    
    print("❌ 手动修复失败")
    return False

def main():
    print("🚀 train_utils.py 缩进修复工具")
    print("=" * 50)
    
    # 备份文件
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"utils/train_utils.py.backup_indent_{timestamp}"
    shutil.copy2("utils/train_utils.py", backup_path)
    print(f"📁 已备份到: {backup_path}")
    
    # 尝试自动修复
    success = check_and_fix_indentation()
    
    if not success:
        print("\n🔧 自动修复失败，尝试手动修复...")
        success = manual_fix_if_needed()
    
    if success:
        print("\n🎉 修复成功！")
        print("🚀 现在可以运行:")
        print("   python main.py --config config_tiny.yml --task_type train")
    else:
        print("\n❌ 修复失败，请手动检查缩进问题")
        print("💡 可能需要手动编辑 utils/train_utils.py")
        print("   确保 evaluate, predict, cubic, analysis, total_params 方法")
        print("   都有正确的4个空格缩进")

if __name__ == "__main__":
    main()