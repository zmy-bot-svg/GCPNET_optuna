#!/usr/bin/env python3
import shutil
from datetime import datetime

print("🔧 修复train_utils.py缩进问题...")

# 备份
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2("utils/train_utils.py", f"utils/train_utils.py.backup_{timestamp}")

with open('utils/train_utils.py', 'r') as f:
    content = f.read()

# 找到关键位置
fit_end = 'return dfhistory'
lr_start = 'class LRScheduler:'

if fit_end in content and lr_start in content:
    before = content.split(fit_end)[0] + fit_end
    after = content.split(lr_start)
    if len(after) > 1:
        after_lr = lr_start + lr_start.join(after[1:])
        
        # 重写方法部分（确保4个空格缩进）
        fixed_methods = '''

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
        import numpy as np
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
        new_content = before + fixed_methods + '\n' + after_lr
        
        with open('utils/train_utils.py', 'w') as f:
            f.write(new_content)
        
        print("✅ 缩进修复完成")
    else:
        print("❌ 修复失败")

# 验证修复结果
try:
    from utils.train_utils import KerasModel
    methods = ['evaluate', 'predict', 'cubic', 'analysis', 'total_params']
    all_exist = True
    for method in methods:
        if hasattr(KerasModel, method):
            print(f"✅ {method} 方法存在")
        else:
            print(f"❌ {method} 方法不存在")
            all_exist = False
    
    if all_exist:
        print("🎉 所有方法都存在，修复成功！")
    else:
        print("❌ 仍有问题")
        
except Exception as e:
    print(f"❌ 验证失败: {e}")
