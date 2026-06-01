"""Real-data language modeling with RetNet-Engram.

Supports character-level tokenization on plain-text datasets.
First target: TinyStories (roneneldan/TinyStories on HuggingFace).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models import AnamnesisModel
from src.models.anamnesis import AnamnesisConfig
from src.models.transformer_baseline import TransformerConfig, TransformerLM
from src.models.linear_attention import LinearAttentionModel

CACHE_DIR = Path("experiments/data")

SPECIAL_TOKENS = ["<pad>", "<eos>"]

TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_valid.txt",
}

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class WordTokenizer:
    """Simple whitespace tokenizer for semantic-level tokenization."""

    def __init__(self, text: str) -> None:
        import re
        words = sorted(set(re.findall(r"\S+", text.lower())))
        self.vocab = SPECIAL_TOKENS + words
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for i, w in enumerate(self.vocab)}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        import re
        return [self.stoi.get(w, self.pad_id) for w in re.findall(r"\S+", text.lower())]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos.get(i, "") for i in ids)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> WordTokenizer:
        vocab = json.loads(path.read_text(encoding="utf-8"))
        tok = cls.__new__(cls)
        tok.vocab = vocab
        tok.stoi = {w: i for i, w in enumerate(vocab)}
        tok.itos = {i: w for i, w in enumerate(vocab)}
        tok.pad_id = tok.stoi["<pad>"]
        tok.eos_id = tok.stoi["<eos>"]
        return tok


class CharTokenizer:
    """Minimal character-level tokenizer."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab = SPECIAL_TOKENS + chars
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.pad_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        vocab = json.loads(path.read_text(encoding="utf-8"))
        tok = cls.__new__(cls)
        tok.vocab = vocab
        tok.stoi = {c: i for i, c in enumerate(vocab)}
        tok.itos = {i: c for i, c in enumerate(vocab)}
        tok.pad_id = tok.stoi["<pad>"]
        tok.eos_id = tok.stoi["<eos>"]
        return tok


class BPETokenizer:
    """BPE tokenizer using HuggingFace tokenizers library.

    Trains a byte-level BPE vocabulary on the training text, enabling
    subword-level tokenization that generalizes to large-vocab scenarios.
    """

    def __init__(
        self,
        text: str,
        vocab_size: int = 4096,
    ) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        self._tokenizer = Tokenizer(BPE(unk_token="<pad>"))
        self._tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        chunks = [text[i : i + 8192] for i in range(0, len(text), 8192)]
        self._tokenizer.train_from_iterator(chunks, trainer=trainer)
        self.pad_id = self._tokenizer.token_to_id("<pad>")
        self.eos_id = self._tokenizer.token_to_id("<eos>")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def save(self, path: Path) -> None:
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        from tokenizers import Tokenizer

        tok = cls.__new__(cls)
        tok._tokenizer = Tokenizer.from_file(str(path))
        tok.pad_id = tok._tokenizer.token_to_id("<pad>")
        tok.eos_id = tok._tokenizer.token_to_id("<eos>")
        return tok


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Fixed-length sequence dataset from tokenized text."""

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return (
            torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_text(url: str, cache_path: Path, max_chars: int | None = None) -> str:
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        import requests

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        chunks: list[str] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=1 << 16, decode_unicode=True):
            chunks.append(chunk)
            total += len(chunk)
            if max_chars and total >= max_chars:
                break
        text = "".join(chunks)
        if max_chars:
            text = text[:max_chars]
        cache_path.write_text(text, encoding="utf-8")
        print(f"Cached {len(text)} chars to {cache_path}")
    return text


def load_dataset(
    name: str,
    split: str,
    max_chars: int | None = None,
) -> str:
    if name == "tinystories":
        return _load_tinystories(split, max_chars)
    elif name == "shakespeare":
        suffix = f".{max_chars}" if max_chars else ""
        cache_path = CACHE_DIR / f"shakespeare_{split}{suffix}.txt"
        return download_text(SHAKESPEARE_URL, cache_path, max_chars)
    elif name == "wikitext103":
        return _load_wikitext(split, "wikitext", "wikitext-103-raw-v1", max_chars)
    elif name == "wikitext2":
        return _load_wikitext(split, "wikitext", "wikitext-2-raw-v1", max_chars)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_tinystories(split: str, max_chars: int | None = None) -> str:
    """Load TinyStories via HuggingFace datasets library."""
    from datasets import load_dataset as hf_load

    hf_split = "train" if split == "train" else "validation"
    suffix = f".{max_chars}" if max_chars else ""
    cache_path = CACHE_DIR / f"tinystories_{split}{suffix}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    print(f"Loading TinyStories [{hf_split}] from HuggingFace...")
    ds = hf_load("roneneldan/TinyStories", split=hf_split, streaming=True)
    texts: list[str] = []
    total = 0
    for row in ds:
        texts.append(row["text"])
        total += len(row["text"])
        if max_chars and total >= max_chars:
            break
    text = "\n".join(texts)
    if max_chars:
        text = text[:max_chars]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    print(f"Cached {len(text)} chars to {cache_path}")
    return text


def _load_wikitext(split: str, dataset: str, config: str, max_chars: int | None = None) -> str:
    """Load WikiText via HuggingFace datasets library."""
    from datasets import load_dataset as hf_load

    hf_split = "train" if split == "train" else "validation"
    suffix = f".{max_chars}" if max_chars else ""
    cache_path = CACHE_DIR / f"wikitext_{config}_{split}{suffix}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    print(f"Loading {dataset}/{config} [{hf_split}] from HuggingFace...")
    ds = hf_load(dataset, config, split=hf_split, streaming=True)
    texts: list[str] = []
    total = 0
    for row in ds:
        texts.append(row["text"])
        total += len(row["text"])
        if max_chars and total >= max_chars:
            break
    text = "\n".join(texts)
    if max_chars:
        text = text[:max_chars]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    print(f"Cached {len(text)} chars to {cache_path}")
    return text


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, vocab_size: int) -> AnamnesisModel | TransformerLM | LinearAttentionModel:
    if getattr(args, "model_type", "anamnesis") == "transformer":
        config = TransformerConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff or args.d_model * 4,
            max_seq_len=args.seq_len * 2,  # support RAG context beyond training length
            dropout=args.dropout,
        )
        return TransformerLM(config)

    if getattr(args, "model_type", "anamnesis") == "linear_attention":
        return LinearAttentionModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff or args.d_model * 4,
        )

    config = AnamnesisConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff or args.d_model * 4,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        position_encoding_type=getattr(args, "position_encoding", "learned"),
        token_copy_sinusoidal_pos=getattr(args, "position_encoding", "learned") == "sinusoidal",
        input_dependent_gamma=bool(getattr(args, "input_dependent_gamma", False)),
        use_token_copy_buffer=args.use_token_copy_buffer,
        use_milestone_snapshots=args.use_milestones,
        milestone_token_ids=(),
        branch_init_scale=args.branch_init_scale,
        attnres_every=args.attnres_every,
        engram_layers=(args.engram_layer,) if args.use_engram else (),
        engram_num_slots=args.engram_slots,
        engram_max_ngram=args.engram_max_ngram,
        engram_hash_heads=args.engram_hash_heads,
        engram_use_conv=args.engram_use_conv,
        engram_vector_gate=args.engram_vector_gate,
        layerwise_gamma=bool(getattr(args, "layerwise_gamma", False)),
        layerwise_gamma_spread=float(getattr(args, "layerwise_gamma_spread", 1.0)),
    )
    return AnamnesisModel(config)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: AnamnesisModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, return_metrics=True)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=0,
        )
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    print(f"Loading {args.dataset} dataset...")
    train_text = load_dataset(args.dataset, "train", args.max_train_chars)
    valid_text = load_dataset(args.dataset, "valid", args.max_valid_chars)

    tokenizer = (
        BPETokenizer(train_text, vocab_size=args.bpe_vocab_size)
        if args.tokenizer == "bpe"
        else WordTokenizer(train_text)
        if args.tokenizer == "word"
        else CharTokenizer(train_text)
    )
    tok_path = Path(args.output_dir) / "tokenizer.json"
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tok_path)
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    print(f"Train tokens: {len(train_ids):,}  Valid tokens: {len(valid_ids):,}")

    train_ds = TextDataset(train_ids, args.seq_len)
    valid_ds = TextDataset(valid_ids, args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_model(args, tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    data_iter = iter(train_loader)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, metrics = model(x, return_metrics=True)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            tokens_per_sec = step * args.batch_size * args.seq_len / max(elapsed, 1e-9)
            row: dict[str, float | int | str] = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": scheduler.get_last_lr()[0],
                "tokens_per_sec": tokens_per_sec,
            }

            if step % args.eval_interval == 0 or step == args.steps:
                val_loss = evaluate(model, valid_loader, device, args.eval_batches)
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                row["val_loss"] = val_loss
                row["val_ppl"] = val_ppl
                print(
                    f"step={step:5d} loss={row['loss']:.4f} "
                    f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} "
                    f"lr={row['lr']:.2e} tok/s={tokens_per_sec:.0f}"
                )
            else:
                print(
                    f"step={step:5d} loss={row['loss']:.4f} "
                    f"lr={row['lr']:.2e} tok/s={tokens_per_sec:.0f}"
                )

            results.append(row)

    csv_path = out_dir / "results.csv"
    if results:
        all_keys = sorted({k for r in results for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    print(f"Results saved to {csv_path}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Model saved to {out_dir / 'model.pt'}")

    model.eval()
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], device=device)
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(200):
            output = model(input_ids, return_metrics=False)
            logits = output[0] if isinstance(output, tuple) else output
            next_logits = logits[0, -1]
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if input_ids.shape[1] >= args.seq_len:
                input_ids = input_ids[:, -args.seq_len // 2 :]

    sample = tokenizer.decode(generated)
    print(f"\n--- Generated sample ---\n{sample}\n--- End ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-data LM training")

    p.add_argument("--model-type", choices=["anamnesis", "transformer", "linear_attention"], default="anamnesis")
    p.add_argument("--tokenizer", choices=["char", "word", "bpe"], default="char")
    p.add_argument("--bpe-vocab-size", type=int, default=4096,
                   help="BPE vocabulary size (only used with --tokenizer bpe).")
    p.add_argument("--dataset", choices=["tinystories", "shakespeare", "wikitext103", "wikitext2"], default="tinystories")
    p.add_argument("--max-train-chars", type=int, default=10_000_000)
    p.add_argument("--max-valid-chars", type=int, default=500_000)
    p.add_argument("--output-dir", type=str, default="experiments/results/real")

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--position-encoding", choices=["learned", "sinusoidal"], default="sinusoidal")
    p.add_argument("--input-dependent-gamma", action="store_true")
    p.add_argument("--layerwise-gamma", action="store_true",
                   help="Shift gamma range per layer: shallow=short memory, deep=long memory.")
    p.add_argument("--layerwise-gamma-spread", type=float, default=1.0,
                   help="Widen shallow-deep gamma contrast (1.0=default).")
    p.add_argument("--use-milestones", action="store_true")
    p.add_argument("--use-token-copy-buffer", action="store_true")
    p.add_argument("--attnres-every", type=int, default=4)
    p.add_argument("--branch-init-scale", type=float, default=1e-4)
    p.add_argument("--use-engram", action="store_true")
    p.add_argument("--engram-layer", type=int, default=2)
    p.add_argument("--engram-slots", type=int, default=4096)
    p.add_argument("--engram-max-ngram", type=int, default=3)
    p.add_argument("--engram-hash-heads", type=int, default=4)
    p.add_argument("--engram-use-conv", action="store_true", default=True)
    p.add_argument("--no-engram-conv", action="store_false", dest="engram_use_conv")
    p.add_argument("--engram-vector-gate", action="store_true", default=False)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=50)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)
