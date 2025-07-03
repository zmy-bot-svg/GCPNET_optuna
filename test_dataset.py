import json
file_path = './data/jarvis_fe_15k/raw/jarvis_fe_15k.json'
try:
    with open(file_path, 'r') as f:
        # 只读取文件的前20个字符来判断它的起始符号
        content = f.read(20).strip() 
    print(f"文件的前20个字符是: '{content}'")
    if content.startswith('['):
        print(">>> 诊断结果：文件是列表（List）格式，这是导致当前报错的【根本原因】。")
    elif content.startswith('{'):
        print(">>> 诊断结果：文件是字典（Dictionary）格式，这是【正确】的格式。")
    else:
        print(">>> 诊断结果：文件格式未知或为空。")
except FileNotFoundError:
    print(f"❌ 诊断失败：找不到文件 {file_path}")