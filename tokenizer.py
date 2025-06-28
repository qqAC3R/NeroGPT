from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = ByteLevel()          # splits text into UTF-8 bytes
tokenizer.decoder       = ByteLevelDecoder()   # knows how to merge bytes back

trainer = BpeTrainer(special_tokens=[""], vocab_size=1024)
tokenizer.train(files=["input.txt"], trainer=trainer)


tokenizer_json = tokenizer.to_str(pretty=False)
tokenizer = Tokenizer.from_str(tokenizer_json)

#tokenizer.model.save("my_bpe")           # creates my_bpe/vocab.json & merges.txt
tokenizer.save("tokenizer.json")
