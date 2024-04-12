# hyperparameters
import os
import torch


batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 8
n_layer = 6
dropout = 0.2
vocab_size = 2048
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


from minbpe import RegexTokenizer

file = "pg2701"


def get_data():
    with open(f"data/{file}.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text


tokenizer = RegexTokenizer()
if not os.path.exists(f"tokenizers/{file}.model"):
    tokenizer.train(get_data(), vocab_size=vocab_size)
    tokenizer.save(f"tokenizers/{file}")  # writes tok32k.model and tok32k.vocab
tokenizer.load(f"tokenizers/{file}.model")  # loads the model back from disk
print(tokenizer.decode(tokenizer.encode("hello world")) == "hello world")


def encode(t):
    return tokenizer.encode(t)


def decode(t):
    return tokenizer.decode(t)


mode = "train"
