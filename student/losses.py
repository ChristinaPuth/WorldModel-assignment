"""Student one-step plus rollout loss with noise injection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states, actions, normalizer):
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(model, states, actions, normalizer, warmup_steps, horizon):
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            f"train_sequence_length too short: need {needed_states - 1} actions "
            f"for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0

    sub_states = states[:, start: start + needed_states]
    sub_actions = actions[:, start: start + int(warmup_steps) + int(horizon)]

    # 噪声注入：对 warmup 起始状态加小噪声，提升鲁棒性
    noise_scale = 0.02
    noisy_states = sub_states.clone()
    noisy_states[:, 0, :] = sub_states[:, 0, :] + torch.randn_like(sub_states[:, 0, :]) * noise_scale

    preds = open_loop_rollout(
        model, noisy_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon
    )
    targets = sub_states[:, warmup_steps + 1: warmup_steps + 1 + horizon]

    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    # 后期步骤权重更大
    T = pred_norm.shape[1]
    weights = torch.linspace(1.0, 4.0, T, device=pred_norm.device)
    weights = weights / weights.mean()

    per_step = F.mse_loss(pred_norm, target_norm, reduction="none").mean(dim=(0, 2))
    return (per_step * weights).mean()


def compute_loss(model, batch, normalizer, cfg):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    one = one_step_delta_loss(model, states, actions, normalizer)

    horizon = int(loss_cfg.get("rollout_train_horizon", 70))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    roll = rollout_loss(model, states, actions, normalizer,
                        warmup_steps=warmup, horizon=horizon)

    one_w = float(loss_cfg.get("one_step_weight", 0.8))
    roll_w = float(loss_cfg.get("rollout_weight", 3.0))
    total = one_w * one + roll_w * roll

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }