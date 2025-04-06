import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None, capture_output=False):
    print(f"> Running command: {' '.join(command)}")
    try:
        process = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture_output, encoding='utf-8', errors='replace')
        if capture_output:
            print(process.stdout)
        return process.returncode
    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        if capture_output:
            print(f"Output:\n{e.output}")
            print(f"Stderr:\n{e.stderr}")
        return e.returncode
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

def main(model_dir, output_file, llama_cpp_dir="llama.cpp", out_type="f16"):
    print(f"-- Starting GGUF Conversion --")
    print(f"   Input HF Model Dir: {model_dir}")
    print(f"   Output GGUF File: {output_file}")
    print(f"   Output Type: {out_type}")
    print(f"   Llama.cpp Dir: {llama_cpp_dir}")

    model_path = Path(model_dir).resolve() # Use absolute path
    output_path = Path(output_file).resolve()
    llama_cpp_path = Path(llama_cpp_dir).resolve()

    # Ensure model directory exists
    if not model_path.is_dir():
        print(f"Error: Input model directory not found: {model_path}")
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Clone llama.cpp if it doesn't exist
    if not llama_cpp_path.exists():
        print(f"\nCloning llama.cpp into {llama_cpp_path}...")
        ret_code = run_command(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp_path)])
        if ret_code != 0:
            print("Failed to clone llama.cpp. Aborting conversion.")
            sys.exit(1)
    else:
        print(f"\nUsing existing llama.cpp directory: {llama_cpp_path}")
        # Optional: Update llama.cpp
        # print("Pulling latest changes for llama.cpp...")
        # run_command(["git", "pull"], cwd=llama_cpp_path)

    # 2. Install llama.cpp requirements (specifically gguf)
    print("\nInstalling conversion script requirements (gguf)...")
    # Check for different possible requirement file names
    req_files = ["requirements-convert.txt", "requirements.txt"]
    installed_reqs = False
    for req_file in req_files:
        req_path = llama_cpp_path / req_file
        if req_path.exists():
            # Try installing only gguf first, then the full file if needed
            ret_code_gguf = run_command([sys.executable, "-m", "pip", "install", "gguf"])
            # We might need more, install the whole file if gguf wasn't enough or errors happen later
            # ret_code_req = run_command([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
            # if ret_code_req == 0:
            #    installed_reqs = True
            #    break
            if ret_code_gguf == 0:
                 installed_reqs = True # Assume gguf is enough for now
                 break 
    if not installed_reqs:
        print("Warning: Could not find or install requirements for llama.cpp conversion script.")
        # Attempting to continue anyway...

    # 3. Find the conversion script
    convert_script_path = llama_cpp_path / "convert.py"
    if not convert_script_path.exists():
        convert_script_path = llama_cpp_path / "convert-hf-to-gguf.py" # Older name
        if not convert_script_path.exists():
             print(f"Error: Could not find conversion script (convert.py or convert-hf-to-gguf.py) in {llama_cpp_path}")
             sys.exit(1)
    print(f"Using conversion script: {convert_script_path}")

    # 4. Run the conversion script
    print(f"\nStarting conversion process...")
    cmd = [
        sys.executable, # Use the current python interpreter
        str(convert_script_path),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", out_type, 
        # Let convert.py auto-detect vocab type for Qwen if possible
    ]

    ret_code = run_command(cmd) # Run in default CWD, script should handle paths

    if ret_code == 0:
        print(f"\n-- Conversion successful! --")
        print(f"   GGUF file saved to: {output_path}")
    else:
        print(f"\n-- Conversion failed! --")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Hugging Face model to GGUF format using llama.cpp.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained Hugging Face model (e.g., output/qwen-ft).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output GGUF file (e.g., output/qwen-ft/gguf-model-f16.gguf).")
    parser.add_argument("--out_type", type=str, default="f16", help="Type of output quantization (e.g., f16, q4_0, q4_k_m, q8_0). Default: f16 (no quantization). See llama.cpp for options.")
    args = parser.parse_args()
    main(args.model_dir, args.output_file, out_type=args.out_type) 