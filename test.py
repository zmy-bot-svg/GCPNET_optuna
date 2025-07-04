#!/usr/bin/env python3
"""
简单的手动修复脚本 - 修复重复的weights_only参数
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(file_path):
    """备份文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"✅ 已备份: {file_path} -> {backup_path}")
    return backup_path

def fix_repeated_weights_only(file_path):
    """修复重复的weights_only参数"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    print(f"🔧 修复文件: {file_path}")
    
    # 备份原文件
    backup_path = backup_file(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复重复的weights_only参数
        # 匹配: weights_only=False, weights_only=False
        content = re.sub(
            r',\s*weights_only=False,\s*weights_only=False',
            ', weights_only=False',
            content
        )
        
        # 修复其他可能的重复形式
        content = re.sub(
            r'weights_only=False,\s*weights_only=False',
            'weights_only=False',
            content
        )
        
        # 确保torch.load都有weights_only=False（但不重复添加）
        # 先找到所有torch.load调用
        torch_load_pattern = r'torch\.load\([^)]+\)'
        matches = re.findall(torch_load_pattern, content)
        
        fixes_made = []
        for match in matches:
            if 'weights_only=' not in match:
                # 只有在没有weights_only参数时才添加
                # 找到最后一个参数后添加
                if match.endswith(')'):
                    # 移除最后的)
                    new_call = match[:-1]
                    # 检查是否需要添加逗号
                    if new_call.endswith('(') or new_call.rstrip().endswith(','):
                        new_call += 'weights_only=False)'
                    else:
                        new_call += ', weights_only=False)'
                    
                    content = content.replace(match, new_call)
                    fixes_made.append(f"Added weights_only=False to: {match}")
        
        # 检查是否有变化
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 修复完成: {file_path}")
            if fixes_made:
                for fix in fixes_made:
                    print(f"   - {fix}")
            else:
                print("   - 清理了重复的weights_only参数")
            return True
        else:
            print(f"ℹ️  无需修复: {file_path}")
            os.remove(backup_path)  # 删除不必要的备份
            return True
            
    except Exception as e:
        print(f"❌ 修复失败: {file_path}")
        print(f"   错误: {e}")
        # 恢复备份
        shutil.copy2(backup_path, file_path)
        return False

def find_and_show_torch_load_calls(file_path):
    """查找并显示文件中的torch.load调用"""
    if not os.path.exists(file_path):
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\n📋 {file_path} 中的torch.load调用:")
        found_any = False
        
        for i, line in enumerate(lines, 1):
            if 'torch.load' in line:
                print(f"   行 {i}: {line.strip()}")
                found_any = True
        
        if not found_any:
            print("   无torch.load调用")
            
    except Exception as e:
        print(f"   读取失败: {e}")

def main():
    """主函数"""
    print("="*60)
    print("🔧 简单修复脚本 - 清理重复的weights_only参数")
    print("="*60)
    
    # 要检查的文件
    files_to_check = [
        'utils/train_utils.py',
        'main.py',
        'main_4090.py'
    ]
    
    print("🔍 检查当前torch.load调用状态...")
    for file_path in files_to_check:
        if os.path.exists(file_path):
            find_and_show_torch_load_calls(file_path)
    
    print("\n" + "="*60)
    print("开始修复...")
    
    success_count = 0
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if fix_repeated_weights_only(file_path):
                success_count += 1
    
    print("\n" + "="*60)
    print("修复后检查...")
    for file_path in files_to_check:
        if os.path.exists(file_path):
            find_and_show_torch_load_calls(file_path)
    
    print("\n" + "="*60)
    print("📊 修复完成!")
    print(f"✅ 成功修复 {success_count} 个文件")
    print("\n📋 接下来请:")
    print("1. 检查上面显示的torch.load调用是否正确")
    print("2. 重新运行程序测试")
    print("3. 如果还有问题，请手动检查相应文件")

if __name__ == "__main__":
    main()