#!/usr/bin/env python3
"""
GCPNet 自动修复脚本
自动修复 torch.load 相关问题
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

def fix_torch_load_in_file(file_path):
    """修复文件中的torch.load调用"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    # 备份原文件
    backup_path = backup_file(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 记录修改
        fixes_made = []
        
        # 修复模式1: torch.load(path) -> torch.load(path, weights_only=False)
        pattern1 = r'torch\.load\((.*?)\)(?!\s*,\s*weights_only=)'
        matches1 = re.findall(pattern1, content)
        if matches1:
            content = re.sub(
                pattern1,
                r'torch.load(\1, weights_only=False)',
                content
            )
            fixes_made.extend([f"torch.load({match})" for match in matches1])
        
        # 修复模式2: torch.load(path, map_location=...) -> torch.load(path, map_location=..., weights_only=False)
        pattern2 = r'torch\.load\((.*?),\s*map_location=(.*?)\)(?!\s*,\s*weights_only=)'
        matches2 = re.findall(pattern2, content)
        if matches2:
            content = re.sub(
                pattern2,
                r'torch.load(\1, map_location=\2, weights_only=False)',
                content
            )
            fixes_made.extend([f"torch.load({match[0]}, map_location={match[1]})" for match in matches2])
        
        if fixes_made:
            # 写入修复后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 修复完成: {file_path}")
            print(f"   修复了 {len(fixes_made)} 个torch.load调用:")
            for fix in fixes_made:
                print(f"   - {fix}")
            return True
        else:
            print(f"ℹ️  无需修复: {file_path}")
            # 删除不必要的备份
            os.remove(backup_path)
            return True
            
    except Exception as e:
        print(f"❌ 修复失败: {file_path}")
        print(f"   错误: {e}")
        # 恢复备份
        shutil.copy2(backup_path, file_path)
        print(f"✅ 已恢复原文件")
        return False

def add_imports_if_needed(file_path):
    """如果需要，添加必要的import"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        imports_to_add = []
        
        # 检查是否需要添加gc import
        if 'gc.collect()' in content and 'import gc' not in content:
            imports_to_add.append('import gc')
        
        if imports_to_add:
            # 找到第一个import行的位置
            lines = content.split('\n')
            import_line_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_line_idx = i
                    break
            
            # 在import区域添加新的import
            for imp in imports_to_add:
                lines.insert(import_line_idx + 1, imp)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ 添加了imports到 {file_path}: {imports_to_add}")
            
    except Exception as e:
        print(f"⚠️  添加imports失败: {e}")

def main():
    """主修复函数"""
    print("="*60)
    print("🔧 GCPNet 自动修复脚本")
    print("="*60)
    
    # 需要修复的文件列表
    files_to_fix = [
        'train_utils.py',
        'utils/train_utils.py',
        'main_4090.py'
    ]
    
    print("🔍 开始修复...")
    
    success_count = 0
    total_count = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"\n📝 处理文件: {file_path}")
            total_count += 1
            
            if fix_torch_load_in_file(file_path):
                success_count += 1
                # 添加必要的imports
                add_imports_if_needed(file_path)
            
        else:
            print(f"⚠️  跳过不存在的文件: {file_path}")
    
    print("\n" + "="*60)
    print("📊 修复结果:")
    print(f"   总计文件: {total_count}")
    print(f"   成功修复: {success_count}")
    print(f"   失败文件: {total_count - success_count}")
    
    if success_count > 0:
        print("\n🎉 修复完成！")
        print("📋 接下来请:")
        print("1. 重启Python进程以清理GPU内存")
        print("2. 重新运行超参数搜索")
        print("3. 如果问题仍然存在，请检查其他可能的torch.load调用")
    else:
        print("\n❌ 未进行任何修复")
        print("💡 可能的原因:")
        print("- 文件路径不正确")
        print("- 代码已经修复过了")
        print("- 需要手动检查torch.load调用")
    
    print("\n📁 备份文件位置:")
    print("   所有原文件都已备份为 .backup_YYYYMMDD_HHMMSS 格式")
    print("   如需回滚，请手动恢复备份文件")

if __name__ == "__main__":
    main()