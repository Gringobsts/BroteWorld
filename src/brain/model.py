from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from brain.data import Prompt
from brain.encoding import EOS, PAD, VOCAB_SIZE, decode_answer, encode_prompt


@dataclass(frozen=True)
class ModelConfig:
    embedding_size: int = 48
    hidden_size: int = 128
    layers: int = 1


class ByteGRU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_size, padding_idx=PAD)
        self.recurrent = nn.GRU(
            config.embedding_size, config.hidden_size, config.layers, batch_first=True
        )
        self.output = nn.Linear(config.hidden_size, VOCAB_SIZE)

    def forward(self, tokens, hidden=None):
        states, hidden = self.recurrent(self.embedding(tokens), hidden)
        return self.output(states), hidden

    @torch.inference_mode()
    def predict(self, prompts: list[Prompt], max_new_bytes: int = 80) -> list[str]:
        if not prompts:
            return []
        if max_new_bytes < 1:
            raise ValueError("El límite de generación debe ser positivo")
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            prefixes = [encode_prompt(x) for x in prompts]
            lengths = torch.tensor(list(map(len, prefixes)))
            tokens = torch.full(
                (len(prompts), int(lengths.max())), PAD, dtype=torch.long, device=device
            )
            for i, prefix in enumerate(prefixes):
                tokens[i, : len(prefix)] = torch.tensor(prefix, device=device)
            packed = pack_padded_sequence(
                self.embedding(tokens), lengths, batch_first=True, enforce_sorted=False
            )
            # Each call and each batch member begins with a zero recurrent state.
            _, hidden = self.recurrent(packed, None)
            logits = self.output(hidden[-1])
            done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
            generated = [[] for _ in prompts]
            for _ in range(max_new_bytes):
                next_token = logits.argmax(-1)
                for i, token in enumerate(next_token.tolist()):
                    if not done[i].item():
                        generated[i].append(token)
                done |= next_token == EOS
                if bool(done.all()):
                    break
                logits, hidden = self(next_token[:, None], hidden)
                logits = logits[:, -1]
            # A truncated response must not accidentally count as a valid completed answer.
            return [
                decode_answer(row) if EOS in row else decode_answer(row) + "�" for row in generated
            ]
        finally:
            self.train(was_training)
