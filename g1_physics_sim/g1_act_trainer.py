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

try:  # package and flat-script imports both work
    from .g1_constants import (    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    NOMINAL_STAND_Q,
    NUM_BODY_MOTORS)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (LEFT_ARM_JOINTS,
RIGHT_ARM_JOINTS,
NOMINAL_STAND_Q,
NUM_BODY_MOTORS)


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
    def __init__(self, parquet_path: str, chunk_size: int = 50, max_episodes: int = 150):
        t = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "episode_index", "task_index"],
        )
        states_raw = t["observation.state"].to_pylist()
        actions_raw = t["action"].to_pylist()
        episodes_raw = t["episode_index"].to_pylist()
        tasks_raw = t["task_index"].to_pylist()

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
            task_id = float(tasks_raw[ep_start])
            for s in range(ep_start, ep_end - chunk_size, 4):
                st = np.array(states_raw[s], dtype=np.float32)
                st_conditioned = np.concatenate([st, [task_id]]).astype(np.float32)
                act_chunk = np.array(actions_raw[s : s + chunk_size], dtype=np.float32)
                self.samples.append((st_conditioned, act_chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s, a = self.samples[idx]
        return torch.from_numpy(s), torch.from_numpy(a)


def collect_multi_robot_experience(num_robots: int = 4, steps_per_robot: int = 100) -> list[tuple[np.ndarray, np.ndarray]]:
    from g1_env import G1PhysicsEnv
    envs = [G1PhysicsEnv(render_mode="none", fix_base=True) for _ in range(num_robots)]
    experiences: list[tuple[np.ndarray, np.ndarray]] = []
    chunk_len = 50
    for env in envs:
        env.reset()
        recent_s: list[np.ndarray] = []
        recent_a: list[np.ndarray] = []
        for _ in range(steps_per_robot):
            act_7 = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(act_7)
            q = env.robot.get_joint_positions()
            left_arm = q[LEFT_ARM_JOINTS]
            right_arm = q[RIGHT_ARM_JOINTS]
            s_16 = np.concatenate([left_arm, right_arm, [0.0, 0.0]]).astype(np.float32)
            a_16 = np.concatenate([act_7, np.zeros(7, dtype=np.float32), [0.0, 0.0]]).astype(np.float32)
            recent_s.append(s_16)
            recent_a.append(a_16)
            if term or trunc:
                env.reset()
        for idx in range(0, len(recent_s) - chunk_len, 5):
            st_cond = np.concatenate([recent_s[idx], [2.0]]).astype(np.float32)
            chunk = np.array(recent_a[idx : idx + chunk_len], dtype=np.float32)
            experiences.append((st_cond, chunk))
        env.close()
    return experiences


def train_act_model(
    parquet_path: str,
    output_dir: str = "models/g1_act",
    epochs: int = 5,
    batch_size: int = 32,
    chunk_size: int = 50,
    num_robots: int = 4,
    log_weight_telemetry: bool = True,
):
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[TRAIN] Loading dataset from: {parquet_path}")
    ds = ParquetChunkDataset(parquet_path, chunk_size=chunk_size, max_episodes=80)
    
    if num_robots > 1:
        print(f"[TRAIN] Running {num_robots} parallel G1 robots in simulation to collect interactive rollouts...")
        extra_exp = collect_multi_robot_experience(num_robots=num_robots, steps_per_robot=150)
        ds.samples.extend(extra_exp)
        print(f"[TRAIN] Added {len(extra_exp)} interactive rollout chunks from {num_robots} parallel robots.")

    print(f"[TRAIN] Total training dataset: {len(ds)} chunks. Device: {device}{gpu_info}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = G1ActionChunkingTransformer(state_dim=17, action_dim=16, chunk_size=chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()

    weight_history_file = out_dir / "weight_training_telemetry.json"
    telemetry_records: list[dict] = []

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

        weight_stats: dict[str, dict[str, float]] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                w_norm = float(param.data.norm().item())
                g_norm = float(param.grad.norm().item()) if param.grad is not None else 0.0
                weight_stats[name] = {"norm": round(w_norm, 4), "grad": round(g_norm, 6)}

        rec = {
            "epoch": epoch,
            "l1_loss": round(avg_loss, 5),
            "time_sec": round(elapsed, 2),
            "num_parallel_robots": num_robots,
            "weights": weight_stats,
        }
        telemetry_records.append(rec)

        summary_keys = ["state_proj.weight", "action_head.weight"]
        summary_str = " | ".join(
            f"{k.split('.')[0]}: W={weight_stats[k]['norm']:.2f},G={weight_stats[k]['grad']:.4f}"
            for k in summary_keys
            if k in weight_stats
        )
        print(f"[TRAIN] Epoch {epoch}/{epochs} | L1 Loss: {avg_loss:.5f} | Time: {elapsed:.2f}s | {summary_str}")

    import json
    with open(weight_history_file, "w", encoding="utf-8") as f:
        json.dump(telemetry_records, f, indent=2)
    print(f"[TRAIN] Weight telemetry logged for every layer to: {weight_history_file}")

    model_path = out_dir / "act_policy_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "chunk_size": chunk_size,
            "state_dim": 17,
            "action_dim": 16,
            "num_parallel_robots": num_robots,
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
        self.state_dim = ckpt.get("state_dim", 17)
        self.model = G1ActionChunkingTransformer(
            state_dim=self.state_dim,
            action_dim=ckpt.get("action_dim", 16),
            chunk_size=self.chunk_size,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.action_buffer: list[np.ndarray] = []
        self.buffer_idx = 0

    def query_action(self, full_robot_q: np.ndarray, task_id: float = 0.0) -> np.ndarray:
        left_arm = full_robot_q[LEFT_ARM_JOINTS]
        right_arm = full_robot_q[RIGHT_ARM_JOINTS]
        left_gripper = 0.0
        right_gripper = 0.0
        if self.state_dim == 17:
            obs = np.concatenate([left_arm, right_arm, [left_gripper, right_gripper, task_id]]).astype(np.float32)
        else:
            obs = np.concatenate([left_arm, right_arm, [left_gripper, right_gripper]]).astype(np.float32)

        if not self.action_buffer or self.buffer_idx >= len(self.action_buffer):
            with torch.no_grad():
                st_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                pred_chunk = self.model(st_t).squeeze(0).cpu().numpy()
            self.action_buffer = [pred_chunk[i] for i in range(len(pred_chunk))]
            self.buffer_idx = 0

        act_16 = self.action_buffer[self.buffer_idx]
        self.buffer_idx += 1

        out_q29 = full_robot_q.copy()
        out_q29[LEFT_ARM_JOINTS] = act_16[:7]
        out_q29[RIGHT_ARM_JOINTS] = act_16[7:14]
        return out_q29


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-robots", type=int, default=4, help="Number of parallel robots to collect training data from")
    parser.add_argument("--batch-size", type=int, default=64)
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
        train_act_model(
            str(parquet_file),
            output_dir=args.output_dir,
            epochs=args.epochs,
            num_robots=args.num_robots,
            batch_size=args.batch_size,
        )
