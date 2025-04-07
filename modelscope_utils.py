"""
modelscope_utils.py

工具函数集，用于启用 ModelScope 作为 Hugging Face 模型的备用下载源。
在无法连接到 Hugging Face 的环境中（如中国大陆），自动切换到 ModelScope。
"""

import os
import sys
import subprocess
import time
import socket
import importlib.util
from typing import Optional, Tuple, Dict, Any, List, Union

# 用于记录当前状态
_MODELSCOPE_ENABLED = False
_MS_TOKEN = None  # 存储 ModelScope 访问令牌

# 模型 ID 映射 (HF -> ModelScope)
# 添加所有需要支持的模型
MODEL_ID_MAPPING = {
    # Qwen 系列模型
    "Qwen/Qwen2.5-0.5B-Chat": "Qwen/Qwen2.5-0.5B-Chat",
    "Qwen/Qwen2.5-1.5B-Chat": "Qwen/Qwen2.5-1.5B-Chat",
    "Qwen/Qwen2.5-7B-Chat": "Qwen/Qwen2.5-7B-Chat",
    "Qwen/Qwen2.5-14B-Chat": "Qwen/Qwen2.5-14B-Chat",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    # 添加其他模型...
}

def check_huggingface_connectivity(timeout=2.0) -> bool:
    """
    检查是否可以连接到 Hugging Face 的服务器。
    
    Args:
        timeout: 连接超时时间（秒）
        
    Returns:
        bool: 如果可以连接则返回 True，否则返回 False
    """
    try:
        # 尝试连接到 Hugging Face 的主要网址
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
        return True
    except Exception:
        try:
            # 作为备选，可以尝试 CDN 或其他网址
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("cdn-lfs.huggingface.co", 443))
            return True
        except Exception:
            return False

def is_package_installed(package_name: str) -> bool:
    """检查指定的 Python 包是否已安装"""
    return importlib.util.find_spec(package_name) is not None

def install_modelscope() -> bool:
    """
    安装 ModelScope 库及其依赖。
    
    Returns:
        bool: 安装成功返回 True，否则返回 False
    """
    try:
        print("正在安装 ModelScope 及其依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        print("ModelScope 安装成功!")
        return True
    except Exception as e:
        print(f"ModelScope 安装失败: {e}")
        return False

def login_to_modelscope(token: Optional[str] = None) -> bool:
    """
    登录到 ModelScope 以便访问私有或受限的模型。
    
    Args:
        token: ModelScope 访问令牌，可从 ModelScope 获取
        
    Returns:
        bool: 登录成功返回 True，否则返回 False
    """
    global _MS_TOKEN
    
    # 如果没有提供 token 并且环境变量中有 token，则使用环境变量中的 token
    if token is None:
        token = os.environ.get("MODELSCOPE_API_TOKEN")
    
    # 如果 token 仍然是 None，说明找不到可用的 token
    if token is None:
        print("注意: 未提供 ModelScope 访问令牌，这可能会限制对某些模型的访问。您可以访问 ModelScope 网站获取令牌。")
        return False
    
    try:
        from modelscope import HubApi
        
        api = HubApi()
        api.login(token)
        _MS_TOKEN = token  # 存储 token 以备后用
        print("已成功登录 ModelScope!")
        return True
    except Exception as e:
        print(f"登录 ModelScope 失败: {e}")
        return False

def map_model_id(hf_model_id: str) -> str:
    """
    将 Hugging Face 模型 ID 映射到对应的 ModelScope 模型 ID。
    
    Args:
        hf_model_id: Hugging Face 模型 ID
        
    Returns:
        str: ModelScope 上对应的模型 ID，如果没有映射则返回原始 ID
    """
    return MODEL_ID_MAPPING.get(hf_model_id, hf_model_id)

def enable_modelscope(model_id: Optional[str] = None, token: Optional[str] = None, force: bool = False) -> bool:
    """
    启用 ModelScope 作为 Hugging Face 模型的下载源。
    
    Args:
        model_id: 要下载的模型 ID（用于映射，可选）
        token: ModelScope 访问令牌（可选）
        force: 是否强制使用 ModelScope，即使可以连接到 Hugging Face
        
    Returns:
        bool: 如果成功启用 ModelScope 则返回 True，否则返回 False
    """
    global _MODELSCOPE_ENABLED
    
    # 如果已经启用，直接返回
    if _MODELSCOPE_ENABLED and not force:
        return True
    
    # 检查是否可以连接到 Hugging Face
    can_access_hf = check_huggingface_connectivity()
    
    if can_access_hf and not force:
        print("可以访问 Hugging Face，将使用 Hugging Face 作为模型源。")
        return False
    
    # 如果无法连接到 Hugging Face 或者强制使用 ModelScope
    if not can_access_hf:
        print("无法连接到 Hugging Face，将尝试使用 ModelScope 作为备选下载源。")
    
    # 检查 ModelScope 是否已安装
    if not is_package_installed("modelscope"):
        print("未检测到 ModelScope 库，正在安装...")
        if not install_modelscope():
            print("无法安装 ModelScope，无法启用备选下载源。")
            return False
    
    # 尝试登录 ModelScope
    login_to_modelscope(token)
    
    # 设置环境变量以使用 ModelScope
    os.environ["MODELSCOPE_CACHE"] = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope/hub"))
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    os.environ["USE_MODELSCOPE_CACHE"] = "1"
    
    # 标记为已启用
    _MODELSCOPE_ENABLED = True
    
    print("已成功启用 ModelScope 作为模型下载源!")
    print(f"模型将会从 ModelScope 下载并缓存在: {os.environ['MODELSCOPE_CACHE']}")
    
    # 如果提供了模型 ID，显示映射信息
    if model_id:
        ms_model_id = map_model_id(model_id)
        if ms_model_id != model_id:
            print(f"HF 模型 ID '{model_id}' 映射到 ModelScope 模型 ID '{ms_model_id}'")
        else:
            print(f"使用原始模型 ID: {model_id} (没有特定映射)")
    
    return True

def get_model_from_either(
    model_name: str,
    model_class: Any,  # 可以是 None
    tokenizer_class: Any = None, # 可以是 None
    use_modelscope: Optional[bool] = None,
    modelscope_token: Optional[str] = None,
    **kwargs
) -> Tuple[Optional[Any], Optional[Any]]: # 返回值也可能为 None
    """
    从 Hugging Face 或 ModelScope 加载模型和/或分词器。
    如果 model_class 或 tokenizer_class 为 None，则不加载对应的部分。
    如果无法连接到 Hugging Face 或者指定了 use_modelscope=True，则优先使用 ModelScope。

    Args:
        model_name: 模型名称或路径
        model_class: 模型类，例如 AutoModelForCausalLM，如果为 None 则不加载模型
        tokenizer_class: 分词器类，例如 AutoTokenizer，如果为 None 则不加载分词器
        use_modelscope: 是否使用 ModelScope，如果为 None 则自动决定
        modelscope_token: ModelScope 访问令牌
        **kwargs: 传递给模型和分词器加载函数的参数

    Returns:
        Tuple[Optional[Any], Optional[Any]]: (model, tokenizer) 元组
    """
    model = None
    tokenizer = None
    source = "Hugging Face" # 默认源

    # 如果 use_modelscope 为 None，则自动决定
    if use_modelscope is None:
        use_modelscope = not check_huggingface_connectivity()

    # 如果需要使用 ModelScope，则启用它
    if use_modelscope:
        if enable_modelscope(model_name, modelscope_token):
            source = "ModelScope"
        else:
            print("启用 ModelScope 失败，将尝试 Hugging Face。")
            source = "Hugging Face"

    # 尝试加载
    try:
        load_path = model_name # 默认从原始名称加载
        
        if source == "ModelScope":
            # 导入 ModelScope
            try:
                from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer
                
                # 将模型映射到 ModelScope ID
                ms_model_id = map_model_id(model_name)
                
                # 如果是本地路径，直接使用；否则从 ModelScope 下载
                if os.path.exists(model_name):
                    load_path = model_name
                else:
                    print(f"正在从 ModelScope 下载资源: {ms_model_id} ...")
                    # 注意: snapshot_download 会下载整个 repo，可能包含模型和分词器
                    load_path = snapshot_download(ms_model_id)
                    print(f"资源下载完成，保存在: {load_path}")
                    
            except Exception as e:
                print(f"从 ModelScope 下载失败: {e}")
                # 如果下载失败，尝试回退到 Hugging Face
                print("将回退到 Hugging Face...")
                source = "Hugging Face"
                load_path = model_name # 重置加载路径
        
        # --- 根据 source 和 load_path 加载 --- 
        # 加载模型 (如果 model_class 不为 None)
        if model_class is not None:
            print(f"正在从 {source} 加载模型: {load_path} ...")
            # 移除可能冲突的 token 参数（如果存在于 kwargs 中）
            model_kwargs = kwargs.copy()
            model_kwargs.pop('token', None)
            model = model_class.from_pretrained(load_path, **model_kwargs)
            
        # 加载分词器 (如果 tokenizer_class 不为 None)
        if tokenizer_class is not None:
            print(f"正在从 {source} 加载分词器: {load_path} ...")
             # 移除可能冲突的 token 参数
            tokenizer_kwargs = kwargs.copy()
            tokenizer_kwargs.pop('token', None)
            tokenizer = tokenizer_class.from_pretrained(load_path, **tokenizer_kwargs)
            
        return model, tokenizer

    except Exception as e:
        print(f"从 {source} 加载失败: {e}")
        # 如果 ModelScope 失败，并且最初尝试的就是 ModelScope，则尝试 Hugging Face
        if source == "ModelScope":
            print("ModelScope 加载失败，尝试回退到 Hugging Face...")
            source = "Hugging Face"
            load_path = model_name
            try:
                # 加载模型 (如果 model_class 不为 None)
                if model_class is not None:
                    print(f"正在从 {source} 加载模型: {load_path} ...")
                    model_kwargs = kwargs.copy()
                    model_kwargs.pop('token', None)
                    model = model_class.from_pretrained(load_path, **model_kwargs)
                    
                # 加载分词器 (如果 tokenizer_class 不为 None)
                if tokenizer_class is not None:
                    print(f"正在从 {source} 加载分词器: {load_path} ...")
                    tokenizer_kwargs = kwargs.copy()
                    tokenizer_kwargs.pop('token', None)
                    tokenizer = tokenizer_class.from_pretrained(load_path, **tokenizer_kwargs)
                    
                return model, tokenizer
            except Exception as hf_e:
                print(f"从 Hugging Face 也加载失败: {hf_e}")
                raise hf_e # 如果两种方式都失败，则抛出最终异常
        else:
             # 如果最初尝试的就是 Hugging Face 并且失败了，直接抛出异常
             raise e 