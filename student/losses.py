# """Student one-step plus rollout loss with noise injection."""

# from __future__ import annotations

# import torch
# import torch.nn.functional as F

# from .rollout import open_loop_rollout


# def one_step_delta_loss(model, states, actions, normalizer):
#     obs = states[:, :-1].reshape(-1, states.shape[-1])
#     act = actions.reshape(-1, actions.shape[-1])
#     target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
#     obs_norm = normalizer.normalize_obs(obs)
#     act_norm = normalizer.normalize_act(act)
#     target_norm = normalizer.normalize_delta(target_delta)
#     pred_norm, _ = model(obs_norm, act_norm, None)
#     return F.mse_loss(pred_norm, target_norm)


# def rollout_loss(model, states, actions, normalizer, warmup_steps, horizon):
#     needed_states = int(warmup_steps) + int(horizon) + 1
#     if states.shape[1] < needed_states:
#         raise ValueError(
#             f"train_sequence_length too short: need {needed_states - 1} actions "
#             f"for warmup={warmup_steps}, horizon={horizon}."
#         )

#     max_start = states.shape[1] - needed_states
#     start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0

#     sub_states = states[:, start: start + needed_states]
#     sub_actions = actions[:, start: start + int(warmup_steps) + int(horizon)]

#     preds = open_loop_rollout(
#         model, sub_states, sub_actions, normalizer,
#         warmup_steps=warmup_steps,
#         horizon=horizon,
#     )

#     targets = sub_states[:, warmup_steps + 1: warmup_steps + 1 + horizon]

#     pred_norm = normalizer.normalize_obs(preds)
#     target_norm = normalizer.normalize_obs(targets)

#     # per-step normalized MSE: [B, T]
#     err = ((pred_norm - target_norm) ** 2).mean(dim=-1)

#     # cap extreme drift so exploded trajectories do not dominate training
#     err = torch.clamp(err, max=2.0)

#     # mild later-step weighting, not too aggressive
#     T = err.shape[1]
#     weights = torch.linspace(1.0, 2.0, T, device=err.device)
#     weights = weights / weights.mean()

#     return (err * weights[None, :]).mean()
# def compute_loss(model, batch, normalizer, cfg):
#     loss_cfg = cfg["loss"]
#     states = batch["states"]
#     actions = batch["actions"]

#     one = one_step_delta_loss(model, states, actions, normalizer)

#     max_horizon = int(loss_cfg.get("rollout_train_horizon", 60))
#     min_horizon = int(loss_cfg.get("rollout_min_horizon", 10))

#     horizon = int(torch.randint(
#         min_horizon,
#         max_horizon + 1,
#         (),
#         device=states.device,
#     ).item())

#     warmup = int(cfg["eval"].get("warmup_steps", 10))

#     roll = rollout_loss(
#         model,
#         states,
#         actions,
#         normalizer,
#         warmup_steps=warmup,
#         horizon=horizon,
#     )

#     one_w = float(loss_cfg.get("one_step_weight", 1.0))
#     roll_w = float(loss_cfg.get("rollout_weight", 1.0))

#     total = one_w * one + roll_w * roll

#     return total, {
#         "loss/total": float(total.detach().cpu()),
#         "loss/one_step": float(one.detach().cpu()),
#         "loss/rollout": float(roll.detach().cpu()),
#     }



"""Student one-step plus rollout loss with curriculum horizon."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout

# 全局步数计数器，因为训练脚本不传step给我们
_global_step = 0


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
    # 加这3行保护
    max_possible = states.shape[1] - int(warmup_steps) - 1
    horizon = min(int(horizon), max_possible)
    if horizon <= 0:
        return torch.tensor(0.0, device=states.device, requires_grad=True)
    
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

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps,
        horizon=horizon,
    )

    targets = sub_states[:, warmup_steps + 1: warmup_steps + 1 + horizon]

    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    err = ((pred_norm - target_norm) ** 2).mean(dim=-1)  # [B, T]

    # 更保守的截断，防止爆炸轨迹主导梯度
    err = torch.clamp(err, max=4.0)

    # 指数加权：后面步骤权重更高，鼓励长期稳定
    T = err.shape[1]
    t_idx = torch.arange(T, device=err.device, dtype=torch.float32)
    weights = torch.exp(0.5 * t_idx / T)
    weights = weights / weights.mean()

    return (err * weights[None, :]).mean()


def compute_loss(model, batch, normalizer, cfg):
    global _global_step
    _global_step += 1

    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    one = one_step_delta_loss(model, states, actions, normalizer)

    max_horizon = int(loss_cfg.get("rollout_train_horizon", 150))
    min_horizon = int(loss_cfg.get("rollout_min_horizon", 10))
    total_updates = int(cfg["training"].get("updates", 30000))

    # 课程学习：前30%训练用短horizon，后期逐渐增长
    curriculum_ratio = min(1.0, _global_step / (total_updates * 0.3))
    current_max = int(min_horizon + curriculum_ratio * (max_horizon - min_horizon))
    current_max = max(current_max, min_horizon)
    current_max = min(current_max, states.shape[1] - warmup - 1)

    horizon = int(torch.randint(
        min_horizon,
        current_max + 1,
        (),
        device=states.device,
    ).item())

    warmup = int(cfg["eval"].get("warmup_steps", 10))

    roll = rollout_loss(
        model, states, actions, normalizer,
        warmup_steps=warmup,
        horizon=horizon,
    )

    one_w = float(loss_cfg.get("one_step_weight", 1.0))
    roll_w = float(loss_cfg.get("rollout_weight", 2.0))

    total = one_w * one + roll_w * roll

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/horizon": float(horizon),
        "loss/curriculum": float(curriculum_ratio),
    }