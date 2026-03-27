"""
Tokenizer Implementation
Educational implementation from scratch using PyTorch only

This implements a BPE tokenizer that can load from HuggingFace's tokenizer.json format.
"""

import json
import re
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path


class Tokenizer:
    """
    BPE Tokenizer that loads from tokenizer.json format.

    This is a simplified implementation for educational purposes.
    For production, consider using the `tokenizers` library.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_tokens: Dict[str, int] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>"
    ):
        """
        Initialize tokenizer.

        Args:
            vocab: Token to ID mapping
            merges: List of BPE merge pairs
            special_tokens: Additional special tokens
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            pad_token: Padding token
        """
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}

        # Special tokens
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.special_tokens = special_tokens or {}

        # Get special token IDs
        self.unk_token_id = vocab.get(unk_token, 0)
        self.bos_token_id = vocab.get(bos_token)
        self.eos_token_id = vocab.get(eos_token)
        self.pad_token_id = vocab.get(pad_token, 0)

        # Cache for BPE encoding
        self.cache = {}

        # Byte encoder/decoder for handling any Unicode
        self.byte_encoder = self._create_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @staticmethod
    def _create_byte_encoder() -> Dict[int, str]:
        """
        Create byte-to-unicode mapping.

        This maps bytes (0-255) to unicode characters.
        Printable ASCII maps to itself, others map to special Unicode.
        """
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))

        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        return {b: chr(c) for b, c in zip(bs, cs)}

    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Get pairs of consecutive symbols in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token: str) -> str:
        """
        Apply BPE to a single token.

        Iteratively merges the most frequent pairs.
        """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            # Find the pair with lowest rank (highest priority)
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float('inf'))
            )

            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                # Find next occurrence of first symbol
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                i = j

                # Check if this is our bigram
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break

            pairs = self._get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # Convert text to bytes and then to byte-encoded string
        text_bytes = text.encode('utf-8')
        text_encoded = ''.join(self.byte_encoder[b] for b in text_bytes)

        # Split into words (simple whitespace tokenization)
        # In practice, you'd want a proper pre-tokenizer
        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        try:
            import regex
            tokens = regex.findall(pattern, text_encoded)
        except ImportError:
            # Fallback to simple split
            tokens = text_encoded.split()

        # Apply BPE to each token
        bpe_tokens = []
        for token in tokens:
            bpe_result = self._bpe(token)
            bpe_tokens.extend(bpe_result.split())

        # Convert to IDs
        ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.unk_token_id)

        # Add special tokens
        if add_special_tokens:
            if self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
            if self.eos_token_id is not None:
                ids = ids + [self.eos_token_id]

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        # Get special token IDs to skip
        special_ids = set()
        if skip_special_tokens:
            for token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                if token_id is not None:
                    special_ids.add(token_id)
            for token, token_id in self.special_tokens.items():
                special_ids.add(token_id)

        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in special_ids:
                continue
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])

        # Join tokens
        text = ''.join(tokens)

        # Convert byte-encoded string back to bytes and decode
        byte_list = []
        for char in text:
            if char in self.byte_decoder:
                byte_list.append(self.byte_decoder[char])

        try:
            decoded = bytes(byte_list).decode('utf-8', errors='replace')
        except Exception:
            decoded = text

        return decoded

    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token IDs."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Tokenizer":
        """
        Load tokenizer from a pretrained model directory.

        Args:
            model_path: Path to model directory containing tokenizer.json

        Returns:
            Tokenizer instance
        """
        model_path = Path(model_path)

        # Try to load tokenizer.json
        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_path}")

        with open(tokenizer_file, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        # Extract vocabulary
        vocab = {}
        if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
            vocab = tokenizer_data["model"]["vocab"]
        elif "vocab" in tokenizer_data:
            vocab = tokenizer_data["vocab"]

        # Extract merges
        merges = []
        if "model" in tokenizer_data and "merges" in tokenizer_data["model"]:
            for merge in tokenizer_data["model"]["merges"]:
                if isinstance(merge, str):
                    parts = merge.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
                elif isinstance(merge, list) and len(merge) == 2:
                    merges.append((merge[0], merge[1]))

        # Extract special tokens
        special_tokens = {}
        if "added_tokens" in tokenizer_data:
            for token_info in tokenizer_data["added_tokens"]:
                if isinstance(token_info, dict):
                    content = token_info.get("content", "")
                    token_id = token_info.get("id", -1)
                    if content and token_id >= 0:
                        special_tokens[content] = token_id

        # Get special token names from config
        unk_token = "<unk>"
        bos_token = "<s>"
        eos_token = "</s>"
        pad_token = "<pad>"

        # Try to load from tokenizer_config.json
        config_file = model_path / "tokenizer_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                unk_token = config.get("unk_token", unk_token)
                bos_token = config.get("bos_token", bos_token)
                eos_token = config.get("eos_token", eos_token)
                pad_token = config.get("pad_token", pad_token)

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token
        )


class SimpleTokenizer:
    """
    Simplified tokenizer that uses pre-built vocabulary.

    For educational purposes - directly loads vocab and decodes/encodes
    without full BPE implementation.
    """

    def __init__(self, vocab: Dict[str, int], special_tokens: Dict[str, int] = None):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or {}
        self.vocab_size = len(vocab)

    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text."""
        import torch

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)

        # Join and clean up
        text = ''.join(tokens)

        # Handle byte-encoded characters (Qwen tokenizer style)
        # Replace special markers and clean up
        text = text.replace('Ġ', ' ')  # GPT-2 style space marker
        text = text.replace('▁', ' ')  # SentencePiece style space marker
        text = text.strip()

        return text

    def batch_decode(
        self,
        batch_ids: Union[List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token IDs."""
        import torch

        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.tolist()

        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SimpleTokenizer":
        """Load from pretrained model directory."""
        model_path = Path(model_path)
        tokenizer_file = model_path / "tokenizer.json"

        with open(tokenizer_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get vocab
        vocab = {}
        if "model" in data and "vocab" in data["model"]:
            vocab = data["model"]["vocab"]

        # Get special tokens
        special_tokens = {}
        if "added_tokens" in data:
            for token_info in data["added_tokens"]:
                if isinstance(token_info, dict):
                    content = token_info.get("content", "")
                    if content:
                        special_tokens[content] = token_info.get("id", -1)

        return cls(vocab=vocab, special_tokens=special_tokens)


if __name__ == "__main__":
    # Test tokenizer
    print("Testing SimpleTokenizer with dummy vocab:")

    # Create a simple test vocabulary
    test_vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "Ġhello": 6,
        "Ġworld": 7,
        "!": 8
    }

    tokenizer = SimpleTokenizer(vocab=test_vocab, special_tokens={"<pad>": 0, "<s>": 1, "</s>": 2})

    # Test decode
    test_ids = [1, 6, 7, 8, 2]  # <s> Ġhello Ġworld ! </s>
    decoded = tokenizer.decode(test_ids, skip_special_tokens=True)
    print(f"Token IDs: {test_ids}")
    print(f"Decoded: '{decoded}'")

    # Test batch decode
    batch = [[1, 6, 7, 2], [1, 4, 5, 2]]
    decoded_batch = tokenizer.batch_decode(batch, skip_special_tokens=True)
    print(f"\nBatch decoded: {decoded_batch}")
