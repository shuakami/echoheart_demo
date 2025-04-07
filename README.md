# Qwen QLoRA Fine-tuning on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shuakami/echoheart_demo/blob/master/run_in_colab.ipynb)

**Goal:** Provide an easy-to-use, end-to-end solution for fine-tuning Qwen language models using QLoRA directly within Google Colab, producing ready-to-use LoRA adapters, merged models, and GGUF files.


## 🚀 Quick Start (Google Colab)

Get fine-tuning in minutes:

1.  **Click the Badge:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shuakami/echoheart_demo/blob/master/run_in_colab.ipynb)
2.  **Configure:** In the first code cell, set your desired `base_model_name` and `dataset_file`.
3.  **Run:** Execute all notebook cells (`Runtime` -> `Run all`).

The notebook handles setup, dependency installation, training, model merging, and optional GGUF conversion.

## ✨ Highlights

*   **Colab-Native:** Designed for a seamless experience in Google Colab.
*   **Efficient QLoRA:** Fine-tune effectively on limited GPU resources (e.g., free Colab tiers).
*   **Easy Configuration:** Centralized configuration for model, data, and hyperparameters within the notebook.
*   **Automatic Optimization:** Dynamically determines a safe batch size to prevent OOM errors.
*   **Multiple Outputs:** Generates LoRA adapters, full merged models, and GGUF files for inference.

## 🔧 How It Works

The core workflow within the `run_in_colab.ipynb` notebook involves:

1.  **Setup:** Environment preparation and dependency installation.
2.  **QLoRA Training (`train.py`):** Loads the base model (4-bit quantized), applies LoRA adapters, and fine-tunes using your dataset and specified hyperparameters. It automatically determines a viable batch size.
3.  **Model Merging (`merge_lora.py`):** Combines the trained LoRA weights with the original base model weights.
4.  **GGUF Conversion (`convert_to_gguf.py`):** Transforms the merged model into the GGUF format suitable for `llama.cpp`.

## ⚙️ Configuration

Fine-tune the process by editing the **first code cell** in `run_in_colab.ipynb`:

*   **Required:**
    *   `base_model_name`: Hugging Face identifier (e.g., `Qwen/Qwen2-1.5B-Instruct`).
    *   `dataset_file`: Path to your JSON dataset.
*   **Optional Hyperparameters:**
    *   Epochs, learning rate, LoRA rank/alpha, etc.
*   **Optional Output Path:**
    *   Set `custom_output_dir` to override the default location.

Detailed comments within the cell guide you through the options.

## ▶️ Output Files

By default, the process generates:

*   **LoRA Adapter:** In `output/<model-name>-qlora-ft/`
*   **Merged Model:** In `output/<model-name>-merged-ft/`
*   **GGUF Model:** Inside the merged model directory (e.g., `output/<model-name>-merged-ft/gguf-model-f16.gguf`)

## 💻 Local Usage (Advanced)

While designed for Colab, the Python scripts (`train.py`, `merge_lora.py`, `convert_to_gguf.py`) can be run locally if you have the necessary environment and hardware. Run them with the `-h` flag to see available command-line arguments.

## 📦 Dependencies

All required packages are listed in `requirements.txt` and installed automatically by the Colab notebook. 