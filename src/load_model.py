import argparse
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example: for demonstration, a placeholder model_name
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B"


def load_llama_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    disable_quantization: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token if not defined (needed for batched inputs)
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Default dtype settings
    dtype = torch.float32  # Use float32 for testing to avoid precision issues

    # Load the model with or without quantization
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",  # Let the library decide optimal device mapping
    }

    if not disable_quantization:
        try:
            # Try to use 8-bit quantization to reduce memory usage
            print("Attempting to load model in 8-bit mode to save memory...")
            model_kwargs["load_in_8bit"] = True
        except Exception as e:
            print(f"Error setting up quantization: {e}")
            print("Falling back to standard loading")

    # By default, trust_remote_code=False unless needed for custom architectures.
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device != "auto" and not model_kwargs.get("device_map"):
        model = model.to(device)
    print("Model loaded with config:", model.config)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and test Llama3.2-3B.")
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model ID"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model",
    )
    args = parser.parse_args()

    _model, _tokenizer = load_llama_model(model_name=args.model_name, device=args.device)
    # Optionally, do a quick check
    prompt = "Hello, how are you?"
    inputs = _tokenizer(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=20)
    print("\n" + 50 * "-" + "\nSample output:\n")
    print(_tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
