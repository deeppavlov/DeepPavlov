
def load_vocab(vocab_path):
    with open(vocab_path) as f:
        return f.read().split()

