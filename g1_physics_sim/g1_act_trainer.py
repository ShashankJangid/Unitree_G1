from __future__ import annotations
import argparse
import math
import os
import pathlib
import sys
import time
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

base_dir = pathlib.Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from g1_constants import (
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    NOMINAL_STAND_Q,
    NUM_BODY_MOTORS,
)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class G1ActionChunkingTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 16,
        chunk_size: int = 50,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.state_proj = nn.Linear(state_dim, d_model)
        self.query_embed = nn.Embedding(chunk_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 2:
            state = state.unsqueeze(1)
        bsz = state.size(0)
        enc_in = self.state_proj(state)
        enc_out = self.transformer_encoder(enc_in)
        queries = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        dec_out = self.transformer_decoder(queries, enc_out)
        pred_actions = self.action_head(dec_out)
        return pred_actions


class ParquetChunkDataset(Dataset):
    def __init__(self, parquet_path: str, chunk_size: int = 50, max_episodes: int = 50):
        t = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "episode_index"],
        )
        states_raw = t["observation.state"].to_pylist()
        actions_raw = t["action"].to_pylist()
        episodes_raw = t["episode_index"].to_pylist()

        self.samples = []
        n = len(states_raw)
        i = 0
        while i < n:
            curr_ep = episodes_raw[i]
            ep_start = i
            while i < n and episodes_raw[i] == curr_ep:
                i += 1
            ep_end = i
            if curr_ep > max_episodes:
                continue
            ep_len = ep_end - ep_start
            if ep_len <= chunk_size:
                continue
            for s in range(ep_start, ep_end - chunk_size, 5):
                st = np.array(states_raw[s], dtype=np.float32)
                act_chunk = np.array(actions_raw[s : s + chunk_size], dtype=np.float32)
                self.samples.append((st, act_chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s, a = self.samples[idx]
        return torch.from_numpy(s), torch.from_numpy(a)


def train_act_model(
    parquet_path: str,
    output_dir: str = "models/g1_act",
    epochs: int = 5,
    batch_size: int = 32,
    chunk_size: int = 50,
):
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Loading dataset from: {parquet_path}")
    ds = ParquetChunkDataset(parquet_path, chunk_size=chunk_size, max_episodes=40)
    print(f"[TRAIN] Created {len(ds)} training chunks. Device: {device}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = G1ActionChunkingTransformer(state_dim=16, action_dim=16, chunk_size=chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count = 0
        t0 = time.time()
        for batch_s, batch_a in loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            optimizer.zero_grad()
            preds = model(batch_s)
            loss = criterion(preds, batch_a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        elapsed = time.time() - t0
        avg_loss = total_loss / max(1, count)
        print(f"[TRAIN] Epoch {epoch}/{epochs} | L1 Loss: {avg_loss:.5f} | Time: {elapsed:.2f}s")

    model_path = out_dir / "act_policy_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "chunk_size": chunk_size,
            "state_dim": 16,
            "action_dim": 16,
        },
        str(model_path),
    )
    print(f"[TRAIN] Model saved to {model_path}")
    return model_path


class G1ACTAutonomousRunner:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(weights_path, map_location=self.device)
        self.chunk_size = ckpt.get("chunk_size", 50)
        self.model = G1ActionChunkingTransformer(
            state_dim=ckpt.get("state_dim", 16),
            action_dim=ckpt.get("action_dim", 16),
            chunk_size=self.chunk_size,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.action_buffer: list[np.ndarray] = []
        self.buffer_idx = 0

    def query_action(self, full_robot_q: np.ndarray) -> np.ndarray:
        left_arm = full_robot_q[LEFT_ARM_INDICES]
        right_arm = full_robot_q[RIGHT_ARM_INDICES]
        left_gripper = 0.0
        right_gripper = 0.0
        obs_16 = np.concatenate([left_arm, right_arm, [left_gripper, right_gripper]]).astype(np.float32)

        if not self.action_buffer or self.buffer_idx >= len(self.action_buffer):
            with torch.no_grad():
                st_t = torch.from_numpy(obs_16).unsqueeze(0).to(self.device)
                pred_chunk = self.model(st_t).squeeze(0).cpu().numpy()
            self.action_buffer = [pred_chunk[i] for i in range(len(pred_chunk))]
            self.buffer_idx = 0

        act_16 = self.action_buffer[self.buffer_idx]
        self.buffer_idx += 1

        out_q29 = full_robot_q.copy()
        out_q29[LEFT_ARM_INDICES] = act_16[:7]
        out_q29[RIGHT_ARM_INDICES] = act_16[7:14]
        return out_q29


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="models/g1_act")
    args = parser.parse_args()

    parquet_file = (
        pathlib.Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--unitreerobotics--G1_Dex1_PickPlaceRedBlock_Dataset_Sim"
        / "snapshots"
        / "77684d6e7e5589b80c8b68f35f221efeac3f19a4"
        / "data"
        / "chunk-000"
        / "file-000.parquet"
    )

    if args.train:
        train_act_model(str(parquet_file), output_dir=args.output_dir, epochs=args.epochs)
