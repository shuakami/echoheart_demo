import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
from pathlib import Path

# 导入 ModelScope 相关工具
try:
    import modelscope_utils
    modelscope_available = True
except ImportError:
    modelscope_available = False
    print("注意: modelscope_utils 模块未找到。如果无法连接到 Hugging Face，将无法回退到 ModelScope。")

def merge_lora_adapter(base_model_name, adapter_path, output_path):
    """Merges a LoRA adapter into the base model and saves the merged model."""
    print(f"-- Starting LoRA Merge --")
    print(f"   Base Model: {base_model_name}")
    print(f"   Adapter Path: {adapter_path}")
    print(f"   Output Path (Merged Model): {output_path}")

    adapter_path_obj = Path(adapter_path)
    output_path_obj = Path(output_path)

    # --- Optional: Load base model with quantization for potentially lower memory usage during merge --- 
    # --- However, merging usually requires loading weights in higher precision --- 
    # --- Let's try loading without quantization first for better merge quality --- 
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )
    # ---------------------------------------------------------------------------------

    try:
        # 确定适合用于合并的计算精度
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 1. 加载基础模型，优先使用 ModelScope (如果可用)
        print(f"\nLoading base model: {base_model_name}...")
        if modelscope_available:
            try:
                print("尝试使用 ModelScope 加载基础模型...")
                base_model, _ = modelscope_utils.get_model_from_either(
                    base_model_name,
                    AutoModelForCausalLM,
                    None,  # 不需要同时加载分词器
                    use_modelscope=None,  # 自动决定
                    torch_dtype=compute_dtype,
                    trust_remote_code=True,
                    device_map="auto"
                )
            except Exception as e:
                print(f"ModelScope 加载失败: {e}")
                print("回退到 Hugging Face...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=compute_dtype,
                    trust_remote_code=True,
                    device_map="auto"
                )
        else:
            # 使用原始方法
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
                device_map="auto" # Load onto available device
            )

        print(f"Loading adapter from: {adapter_path}...")
        # Load the LoRA adapter onto the base model
        model_to_merge = PeftModel.from_pretrained(base_model, adapter_path_obj)

        print("Merging adapter into the base model...")
        # Merge the adapter weights into the base model
        # This operation happens in place
        merged_model = model_to_merge.merge_and_unload()
        print("Merge complete.")

        # Ensure output directory exists
        output_path_obj.mkdir(parents=True, exist_ok=True)

        print(f"Saving merged model to: {output_path}...")
        # Save the merged model
        merged_model.save_pretrained(output_path_obj)

        # 3. 加载并保存 Tokenizer (先尝试从适配器目录加载，确保一致性)
        print(f"Saving tokenizer to: {output_path}...")
        try:
            if modelscope_available:
                print("尝试使用 ModelScope 加载 tokenizer...")
                _, tokenizer = modelscope_utils.get_model_from_either(
                    adapter_path,  # 先尝试从适配器目录加载
                    None,  # 不需要加载模型
                    AutoTokenizer,
                    use_modelscope=None,  # 自动决定
                    trust_remote_code=True
                )
                if tokenizer is None:
                    raise ValueError("tokenizer 加载失败")
            else:
                tokenizer = AutoTokenizer.from_pretrained(adapter_path_obj, trust_remote_code=True)
        except Exception as adapter_e:
            print(f"从适配器目录加载 tokenizer 失败: {adapter_e}，尝试从基础模型加载")
            # 回退到从基础模型加载
            if modelscope_available:
                try:
                    _, tokenizer = modelscope_utils.get_model_from_either(
                        base_model_name,
                        None,  # 不需要加载模型
                        AutoTokenizer,
                        use_modelscope=None,  # 自动决定
                        trust_remote_code=True
                    )
                    if tokenizer is None:
                        raise ValueError("从基础模型加载 tokenizer 失败")
                except Exception as base_e:
                    print(f"从基础模型通过 ModelScope 加载 tokenizer 失败: {base_e}")
                    print("回退到 Hugging Face...")
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                
        tokenizer.save_pretrained(output_path_obj)

        print(f"\n-- Merge Successful! --")
        print(f"   Merged model saved to: {output_path}")

    except Exception as e:
        print(f"\n-- Merge Failed! --")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name or path of the base model (e.g., Qwen/Qwen2.5-1.5B-Instruct).")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the directory containing the LoRA adapter files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model.")
    args = parser.parse_args()
    merge_lora_adapter(args.base_model_name, args.adapter_path, args.output_path) 