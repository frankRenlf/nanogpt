#!/bin/bash

scripts=("gpt.py" "gpt_.py" "gpt_cur3.py" "gpt_cur4.py" "gpt_llama.py")

# 使用 for 循环依次执行每个脚本
for script in "${scripts[@]}"
do
  echo "Running $script..."
  CUDA_VISIBLE_DEVICES=4 python $script
  echo "$script finished."
done

echo "all training done"
