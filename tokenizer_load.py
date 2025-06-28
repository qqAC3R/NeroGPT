from tokenizers import Tokenizer

# open file storing the text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text)

tokenizer = Tokenizer.from_file("tokenizer.json")
text_encoded = tokenizer.encode(text).ids
print(text_encoded)
vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

text = tokenizer.decode(text_encoded)
print(text)