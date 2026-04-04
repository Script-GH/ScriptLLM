class SimpleTokenizer:

    def __init__(self, text):

        words = sorted(list(set(text.split())))

        self.word_to_id = {w:i for i,w in enumerate(words)}
        self.id_to_word = {i:w for w,i in self.word_to_id.items()}

    def encode(self, text):
        return [self.word_to_id[w] for w in text.split()]

    def decode(self, ids):
        return " ".join([self.id_to_word[i] for i in ids])

class CharTokenizer:

    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.char_to_id = {c:i for i,c in enumerate(chars)}
        self.id_to_char = {i:c for c,i in self.char_to_id.items()}

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join([self.id_to_char.get(i, "") for i in ids])