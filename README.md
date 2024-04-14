# Dual-GPT Implementations

This repository contains two distinct implementations of the multi-head attention mechanism used in GPT models. The attention mechanism is a critical component that allows the model to focus on different parts of the input sequence when predicting the output.

## Implementation Details

### Single-Matrix-Multiplication Multi-Head GPT (SM-GPT)

The `gpt_.py` contains an implementation of the GPT model that utilizes a single matrix multiplication to compute multi-head attention. This method reshapes the input tensor to allow for parallel computation of all heads within the attention mechanism. The main advantage of this approach is efficiency; by performing one large matrix multiplication instead of multiple smaller ones, we reduce computational overhead and take better advantage of hardware acceleration.

Key features:
- Efficient computation of attention with fewer sequential steps.
- Utilization of batch matrix multiplication for parallel processing of heads.
- Reduction in Python-level for loop overheads.

### Loop-Based Multi-Head GPT (LB-GPT)

The `gpt.py` directory contains an implementation of the GPT model that computes each head's attention separately within a loop and then concatenates the results. This method is more straightforward and easier to understand, as each attention head is computed one after another, which can be beneficial for debugging and educational purposes.

Key features:
- Simplicity in understanding and debugging the attention mechanism.
- Direct control over each head's computation, allowing for custom head-level manipulations.
- Each head's output can be independently analyzed before concatenation.

# Comparison

- `gpt_.py` is optimized for performance and is best suited for scenarios where computational resources and efficiency are paramount.
- `gpt.py` is optimized for clarity and is best suited for educational purposes, debugging, or research where individual attention head behavior needs to be studied.

Both implementations achieve the same theoretical result, but they differ in execution and performance characteristics.

# Tokenize
data use minbpe as the tokenizer

# Trained components
You can find:
- trained models in model_dict dict
- trained tokenizers in tokenizers dict
- dataset in data dict
- generated text in generate_text dict

You can modify config:
- config in config.py

# Usage

Please refer to the individual directories for detailed instructions on how to run each implementation.


# Requirements

```bash
conda create -n gpt python=3.10 -y
pip install -r requirements.txt
```
## train or generate
```bash
python gpt.py # or gpt_.py
```

# Future work

prepare to add rlhf module