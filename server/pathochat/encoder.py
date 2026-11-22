import os
import unicodedata

import numpy as np

LAYERS = 6
HEADS = 12
MAX_TOKENS = 256
EPS = 1e-12


class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        self.unk = self.vocab["[UNK]"]
        self.cls = self.vocab["[CLS]"]
        self.sep = self.vocab["[SEP]"]

    @staticmethod
    def is_punctuation(ch):
        cp = ord(ch)
        if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
            return True
        return unicodedata.category(ch).startswith("P")

    @staticmethod
    def is_cjk(ch):
        cp = ord(ch)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
                or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)

    def words(self, text):
        cleaned = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
                continue
            if self.is_cjk(ch):
                cleaned.append(" " + ch + " ")
            elif ch in "\t\n\r" or unicodedata.category(ch) == "Zs":
                cleaned.append(" ")
            else:
                cleaned.append(ch)
        out = []
        for word in "".join(cleaned).lower().split():
            word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
            piece = ""
            for ch in word:
                if self.is_punctuation(ch):
                    if piece:
                        out.append(piece)
                        piece = ""
                    out.append(ch)
                else:
                    piece += ch
            if piece:
                out.append(piece)
        return out

    def encode(self, text):
        ids = [self.cls]
        for word in self.words(text):
            if len(word) > 100:
                ids.append(self.unk)
                continue
            start, pieces = 0, []
            while start < len(word):
                end = len(word)
                found = None
                while start < end:
                    piece = ("##" if start else "") + word[start:end]
                    if piece in self.vocab:
                        found = self.vocab[piece]
                        break
                    end -= 1
                if found is None:
                    pieces = [self.unk]
                    break
                pieces.append(found)
                start = end
            ids.extend(pieces)
        return ids[: MAX_TOKENS - 1] + [self.sep]


def erf(x):
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-a * a))


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def layer_norm(x, weight, bias):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight + bias


class Encoder:
    def __init__(self, folder):
        self.tokenizer = Tokenizer(os.path.join(folder, "vocab.txt"))
        with np.load(os.path.join(folder, "weights.npz")) as data:
            self.w = {k: data[k].astype(np.float32) for k in data.files}

    def dense(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def encode(self, text):
        w = self.w
        ids = self.tokenizer.encode(text)
        n = len(ids)
        x = w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:n] + w["embeddings.token_type_embeddings.weight"][0]
        x = layer_norm(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"])
        size = x.shape[1] // HEADS
        for i in range(LAYERS):
            p = f"encoder.layer.{i}."
            split = lambda t: t.reshape(n, HEADS, size).transpose(1, 0, 2)
            q = split(self.dense(x, p + "attention.self.query"))
            k = split(self.dense(x, p + "attention.self.key"))
            v = split(self.dense(x, p + "attention.self.value"))
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(size)
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            scores /= scores.sum(-1, keepdims=True)
            ctx = (scores @ v).transpose(1, 0, 2).reshape(n, -1)
            x = layer_norm(x + self.dense(ctx, p + "attention.output.dense"), w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"])
            hidden = gelu(self.dense(x, p + "intermediate.dense"))
            x = layer_norm(x + self.dense(hidden, p + "output.dense"), w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"])
        pooled = x.mean(0)
        return pooled / max(np.linalg.norm(pooled), 1e-12)
