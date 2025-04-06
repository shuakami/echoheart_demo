import json
import os

def convert_to_qwen_format(input_file, output_file):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换格式
    converted_data = []
    for item in data:
        converted_item = {
            "messages": [
                {
                    "role": "user", 
                    "content": item["instruction"] + (" " + item["input"] if item["input"] else "")
                },
                {
                    "role": "assistant", 
                    "content": item["output"]
                }
            ]
        }
        converted_data.append(converted_item)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"数据已转换完成！共处理 {len(converted_data)} 条记录")
    print(f"输出文件保存在: {output_file}")

if __name__ == "__main__":
    input_file = "data/echoheart_demo.json"
    output_file = "data/converted_dataset.json"
    convert_to_qwen_format(input_file, output_file) 