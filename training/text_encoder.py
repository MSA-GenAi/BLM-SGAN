"""BERT text encoder of BLM-SGAN (BERT-DAMSM).

  * BERT (bert-base-uncased) reads the caption text, up to 40 word pieces;
  * the sentence vector is the masked mean of BERT's token states (BERT's pooled output, trained for next-sentence
    prediction, gives almost the same vector for every caption);
  * the 768 -> 256 projections are trained together with the top BERT layers and a DAMSM image encoder
    (train_damsm.py), so text and images share one embedding space;
  * word features are aligned with the caption length the DAMSM word loss uses.

Saved file (text_encoder.pth): {"config": {...}, "state_dict": only the trainable tensors}. Loading
rebuilds BERT from the pretrained weights and overlays them, so the file is ~60 MB instead of 440 MB.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast


class BertTextEncoder(nn.Module):
    def __init__(self, bert: str = "bert-base-uncased", emb_dim: int = 256, max_len: int = 40,
                 finetune_layers: int = 2):
        super().__init__()
        self.config = {"bert": str(bert), "emb_dim": emb_dim, "max_len": max_len,
                       "finetune_layers": finetune_layers}
        self.tokenizer = BertTokenizerFast.from_pretrained(str(bert))
        self.bert = BertModel.from_pretrained(str(bert), add_pooling_layer=False)
        hidden = self.bert.config.hidden_size
        for p in self.bert.parameters():
            p.requires_grad = False
        n = self.bert.config.num_hidden_layers
        for layer in self.bert.encoder.layer[n - finetune_layers:] if finetune_layers > 0 else []:
            for p in layer.parameters():
                p.requires_grad = True
        self.proj_words = nn.Linear(hidden, emb_dim)
        self.proj_sent = nn.Linear(hidden, emb_dim)

    @property
    def device(self) -> torch.device:
        return self.proj_sent.weight.device

    def forward(self, captions: list[str]):
        """captions: list of B strings ->
        words    B x emb_dim x L   (word-piece features between [CLS] and [SEP]; zero past cap_lens)
        sent     B x emb_dim
        cap_lens B                 (number of word pieces per caption, >= 1)
        """
        tok = self.tokenizer(list(captions), padding=True, truncation=True,
                             max_length=self.config["max_len"], return_tensors="pt").to(self.device)
        h = self.bert(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).last_hidden_state
        m = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
        sent = self.proj_sent((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        cap_lens = (tok["attention_mask"].sum(1) - 2).clamp(min=1)
        words = self.proj_words(h[:, 1:-1])                            # drop [CLS] and the last column
        if words.size(1) == 0:                                         # every caption empty
            words = self.proj_words(h[:, 1:2])
        valid = torch.arange(words.size(1), device=self.device)[None] < cap_lens[:, None]
        words = words * valid.unsqueeze(-1).to(words.dtype)
        return words.transpose(1, 2).contiguous(), sent, cap_lens

    # ------------------------------------------------------------- save / load
    def trainable_state_dict(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def save(self, path, **extra) -> None:
        torch.save({"config": self.config, "state_dict": self.trainable_state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, bert: str | None = None, map_location="cpu") -> "BertTextEncoder":
        """bert: override the BERT location stored in the file (e.g. a local offline copy)."""
        ck = torch.load(path, map_location=map_location, weights_only=True)
        cfg = dict(ck["config"])
        if bert:
            cfg["bert"] = str(bert)
        enc = cls(**cfg)
        _, unexpected = enc.load_state_dict(ck["state_dict"], strict=False)
        assert not unexpected, f"unexpected keys in {path}: {unexpected[:3]}"
        need = {n for n, p in enc.named_parameters() if p.requires_grad}
        assert need <= set(ck["state_dict"]), f"{path} lacks {sorted(need - set(ck['state_dict']))[:3]}"
        return enc
