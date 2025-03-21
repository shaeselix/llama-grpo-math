# llama-grpo-math
Fine-tuning Llama models for mathematical reasoning using GSM8K and other math datasets.

## Overview
This project implements supervised fine-tuning and evaluation for Llama-3.2-3B models on mathematical reasoning tasks. The code focuses on improving the model's ability to solve step-by-step math problems from datasets like GSM8K.

## Features
- Supervised fine-tuning using LoRA
- Structured prompting for step-by-step math reasoning
- Comprehensive evaluation suite for mathematical reasoning
- Visualization of evaluation results
- Type-safe implementation with mypy support

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)
- Hugging Face access to Llama-3.2 models

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/llama-grpo-math.git
cd llama-grpo-math
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Usage

### Fine-tuning
To fine-tune the model on GSM8K dataset:
```bash
python -m src.train \
  --base_model meta-llama/Llama-3.2-3B \
  --output_dir ./model_math_finetuned \
  --num_train_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --sample_size 1000 \
  --evaluate_after_training
```

### Evaluation
To evaluate a model on math datasets:
```bash
python -m src.evaluate \
  --model_name_or_path ./model_math_finetuned \
  --test_gsm8k \
  --max_samples 50 \
  --visualize
```

```bash
uv run --with-requirements requirements.txt python -m src.evaluate --model_name_or_path /home/ubuntu/grpo/llama-grpo-math/models/lora/2025-03-12_22-08-34/checkpoint-200
 --test_gsm8k --max_samples 2000 --visualize --vllm
```

## Development

### Code Quality Tools
This project uses several tools to maintain code quality:

1. **pre-commit hooks**: Automatically runs checks before each commit
   - Install: `pip install pre-commit`
   - Setup: `pre-commit install`
   - Manual run: `pre-commit run --all-files`

2. **Ruff**: Fast Python linter
   - Run: `ruff check src/`
   - Auto-fix: `ruff check --fix src/`

3. **mypy**: Static type checking
   - Run: `mypy src/`

4. **pytest**: Testing framework
   - Run all tests: `python -m pytest tests/`
   - Run specific tests: `python -m pytest tests/src/test_evaluate.py -v`

### Pre-commit Hooks
The pre-commit configuration includes:
- Code formatting with ruff
- Type checking with mypy
- Import sorting
- Various syntax and security checks

To run the pre-commit hooks manually:
```bash
pre-commit run --all-files
```

## License
[Insert License Information]

## Contributors
[Insert Contributor Information]
