"""
ATEC2026 仿真挑战赛 —— 提交入口 solution.py (Worker 7 scaffold)
=================================================================
本文件是【唯一】的本地测试与在线提交入口，文件名不可更改。

接口契约（来自 readme.md §3.3 与 demo/server.py）：
  - 类名必须是 AlgSolution
  - 方法 predicts(self, obs, current_score) -> {"action": <List>, "giveup": <bool>}
      * obs            : dict，结构见下方"obs 结构"说明
      * current_score  : float，当前累计得分（play_atec_task.py 里传的是 total_episode_reward）
      * 返回 action 必须是 Python List（server 端会 torch.tensor(...).view(1,-1)）
      * giveup=True 会立即终止评测；本骨架默认 False

obs 结构（由 demo/server.py 组装，注意 Task A/D 与 Task E 不同）：
  Task A / B / D (移动类，head_rgb 存在时):
    obs = {
      'proprio': Tensor(1, N),                 # 本体感知，已 cuda
      'extero' : Tensor(1, M) 或 None,          # LiDAR 高度扫描
      'image'  : {'head_rgb','head_depth','ee_rgb','ee_depth'}
    }
  Task E (桌面操作类，head_rgb 为 None 时):
    obs = {
      'proprio': Tensor(1, 24),
      'extero' : None,
      'image'  : {'video_rgb','video_depth','ee_rgb','ee_depth'}
    }

proprio 布局（Task A/B/D，按 readme §3.4 顺序，全部已注入噪声/保序/分组拼接）：
    [ base_lin_vel(3), base_ang_vel(3), velocity_commands(3), projected_gravity(3),
      joint_pos(action_dim), joint_vel(action_dim), prev_actions(action_dim) ]
  => proprio.shape[-1] == 12 + 3*action_dim
  => action_dim = (proprio.shape[-1] - 12) // 3
  机器人自由度: b2_piper=20, b2w_piper=24, G1=33, tron1a_piper=16, piper=8

动作空间：所有任务都是关节位置控制 (JointPositionActionCfg, scale=0.5,
use_default_offset=True, preserve_order=True)。返回的 action 维度必须等于 action_dim。

=================================================================
当前策略（L0 解锁优先）
=================================================================
赛道1 (徒步) L0 = Task A：本骨架默认装载官方 RL locomotion baseline
  (demo/policy.pt, B2 flat 步态)，让机器人前进，目标是 L0 得分 >=0.01 解锁 L1。
  这复用了 solution_rl.py 的核心逻辑，并增加了容错（policy 加载失败时回退到
  安全的"原地保持"动作，保证 predicts 永远返回合法格式、不崩）。

赛道2 (桌面整理) L0 = Task E：如要切换，把下方 MODE 改为 "act"，
  并按 Dockerfile 注释把 act/ 目录与 policy_act.pt 一并 COPY 进镜像。
  Task E 的实现见 solution_act.py（ACT/CVAE，需要 RGB 图像）。

提交（readme + 选手指南）：
  - 方式一：上传源码由平台自动构建镜像；方式二：自行构建 Docker 镜像推送。
  - 镜像内 server.py 会 `from solution import AlgSolution`，因此本文件必须无副作用即可 import。
  - L0 规则：任一 L0 任务有过一次提交且得分 >=0.01 即视为完成，解锁 L1。
"""

import os
import sys
import torch
import math

# 选择 L0 入门策略："rl" -> Task A 徒步 (locomotion baseline)；"act" -> Task E 桌面整理。
MODE = os.environ.get("ATEC_SOLUTION_MODE", "rl")

_HERE = os.path.dirname(os.path.abspath(__file__))

# 兼容两种加载方式：
#   1) 提交/server.py：cwd=solution 目录，本文件作为顶层模块 `solution` 导入，
#      `import solution_act` / `import act.*` 直接可用；
#   2) 本地评估/play_atec_task.py：cwd=仓库根，本文件作为 `demo.solution` 子模块导入，
#      此时 demo/ 不在 sys.path 上，顶层名 `solution_act` / `act` 无法解析。
# 把本文件所在目录(demo/)注入 sys.path，使 ACT delegate 与 act/ 子包在两种方式下都能 import。
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class AlgSolution:
    """Task A 徒步 L0 baseline：装载官方 locomotion policy 让 B2 前进。

    设计目标：
      1) 永远返回格式合法的 {"action": List, "giveup": bool}，绝不抛异常导致评测 0 分；
      2) policy.pt 可用时用 RL 步态前进（最有希望拿 >=0.01 解锁 L1）；
      3) policy.pt 缺失/加载失败时，回退到"原地保持/零动作"，至少保证流程能跑通。
    """

    _TRAIN_TO_ENV_LEG = [0.25, 0.5, 0.5] * 4
    _ENV_TO_TRAIN_LEG = [4.0, 2.0, 2.0] * 4
    _LEG_ACTION_DIM = 12
    _DEFAULT_ACTION_DIM = 20
    _ACTION_SMOOTH_ALPHA = float(os.environ.get("ATEC_SMOOTH_ALPHA", "0.3"))

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = None
        self._mode = MODE

        if self._mode == "act":
            try:
                from solution_act import AlgSolution as _ActSolution  # noqa: WPS433
                self._delegate = _ActSolution()
                return
            except Exception as err:  # noqa: BLE001
                print(f"[solution] WARN: ACT delegate unavailable ({err}); "
                      f"falling back to locomotion (rl) mode.")
                self._mode = "rl"
        self._delegate = None

        policy_path = os.path.join(_HERE, "policy.pt")
        try:
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy.eval()
        except Exception as err:  # noqa: BLE001
            print(f"[solution] WARN: failed to load policy.pt ({err}); "
                  f"falling back to hold action.")
            self.policy = None

        self.leg_joint_indices = list(range(self._LEG_ACTION_DIM))
        self._train_to_env = torch.tensor(
            self._TRAIN_TO_ENV_LEG, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self._env_to_train = torch.tensor(
            self._ENV_TO_TRAIN_LEG, device=self.device, dtype=torch.float32
        ).view(1, -1)

        vx = float(os.environ.get("ATEC_VEL_X", "1.5"))
        vy = float(os.environ.get("ATEC_VEL_Y", "0.0"))
        vyaw = float(os.environ.get("ATEC_VEL_YAW", "0.0"))
        self._vel_cmd = torch.tensor(
            [vx, vy, vyaw], device=self.device, dtype=torch.float32
        ).view(1, 3)

        self._prev_action = None
        self._step_count = 0

    # ------------------------------------------------------------------ #
    # 每集复位 (server.py /reset -> agent.reset(**form_data))
    # ------------------------------------------------------------------ #
    def reset(self, **kwargs):
        try:
            delegate = getattr(self, "_delegate", None)
            if delegate is not None and hasattr(delegate, "reset"):
                delegate.reset(**kwargs)
            self._prev_action = None
            self._step_count = 0
        except Exception as err:  # noqa: BLE001
            print(f"[solution] WARN: reset() ignored error ({err})")
        return None

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def predicts(self, obs, current_score):
        # 委托模式（Task E / ACT）
        if self._delegate is not None:
            return self._delegate.predicts(obs, current_score)

        try:
            proprio = obs["proprio"]
            if not isinstance(proprio, torch.Tensor):
                proprio = torch.as_tensor(proprio, dtype=torch.float32)
            proprio = proprio.to(self.device).float()
            action_dim = (int(proprio.shape[-1]) - 12) // 3
            if action_dim < self._LEG_ACTION_DIM:
                # 不足以容纳 12 条腿：可能是非 B2 类机器人（如 Piper action_dim=8）。
                # 此时 locomotion policy 不适用，安全回退到 action_dim 维零动作；
                # 仅当 proprio 解析出非正维度才用 _DEFAULT_ACTION_DIM 兜底，
                # 绝不再硬编码 20 维（否则会与 Piper 期望的 8 维冲突而崩）。
                safe_dim = action_dim if action_dim > 0 else self._DEFAULT_ACTION_DIM
                return {"action": self._zero_action(proprio, safe_dim),
                        "giveup": False}

            if self.policy is None:
                # 安全回退：返回全零动作（关节保持默认姿态，至少跑通流程）。
                return {"action": self._zero_action(proprio, action_dim), "giveup": False}

            policy_obs = self._build_policy_obs(proprio, action_dim)
            with torch.inference_mode():
                action_train = self.policy(policy_obs)
            action_train = self._as_2d_tensor(action_train)

            action_env = self._leg_action_to_full(action_train, action_dim)

            if self._prev_action is not None and self._ACTION_SMOOTH_ALPHA > 0:
                alpha = self._ACTION_SMOOTH_ALPHA
                action_env = alpha * action_env + (1 - alpha) * self._prev_action
            self._prev_action = action_env.clone()
            self._step_count += 1

            return {"action": action_env.cpu().numpy().tolist(), "giveup": False}
        except Exception as err:  # noqa: BLE001 - 评测期绝不抛出
            print(f"[solution] WARN: predicts fallback to zero action ({err})")
            try:
                proprio = obs["proprio"]
                action_dim = (int(proprio.shape[-1]) - 12) // 3
                if action_dim <= 0:
                    action_dim = self._DEFAULT_ACTION_DIM
                return {"action": self._zero_action(proprio, action_dim), "giveup": False}
            except Exception:  # 最后兜底：返回单环境的全零默认动作（绝不返回空 list）
                return {"action": [[0.0] * self._DEFAULT_ACTION_DIM], "giveup": False}

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _zero_action(self, proprio, action_dim):
        num_envs = int(proprio.shape[0]) if hasattr(proprio, "shape") else 1
        return [[0.0] * action_dim for _ in range(num_envs)]

    @staticmethod
    def _as_2d_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    def _build_policy_obs(self, proprio, action_dim):
        """从 env 的 proprio 切出 leg 观测，拼成 locomotion policy 的输入。

        policy 输入顺序（沿用 solution_rl.py / robot_lab 约定）：
          [ base_ang_vel*0.25, projected_gravity, velocity_commands,
            joint_pos_leg, joint_vel_leg*0.05, prev_actions_leg(train scale) ]
        """
        idx = 3                                    # 跳过 base_lin_vel(3)
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        idx += 3                                    # 跳过 env 自带 velocity_commands(3)
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        prev_actions_all = proprio[:, idx:idx + action_dim]

        leg = self.leg_joint_indices
        joint_pos_leg = joint_pos_all[:, leg]
        joint_vel_leg = joint_vel_all[:, leg]
        prev_actions_leg = prev_actions_all[:, leg] * self._env_to_train

        vel_cmd = self._vel_cmd
        if proprio.shape[0] > 1:
            vel_cmd = vel_cmd.repeat(proprio.shape[0], 1)

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                vel_cmd,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                prev_actions_leg,
            ],
            dim=-1,
        )

    def _leg_action_to_full(self, action_train, action_dim):
        """把 policy 输出的 12D 腿部动作映射回当前 env 的全身动作维度。

        非腿部关节（机械臂/夹爪）填 0，保持默认姿态。
        """
        if action_train.shape[-1] != self._LEG_ACTION_DIM:
            # policy 维度与预期不符：直接零动作兜底，避免越界。
            return torch.zeros((action_train.shape[0], action_dim), device=self.device)
        num_envs = action_train.shape[0]
        leg_env = action_train * self._train_to_env
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_env
        return action_env


# 本地快速自检（不依赖 Isaac，只验证接口契约与容错回退）。
# 运行：python demo/solution.py
if __name__ == "__main__":
    print(f"[selfcheck] MODE={MODE}, cuda={torch.cuda.is_available()}")
    # 伪造一个 b2_piper(20DoF) 的 proprio：12 + 3*20 = 72 维
    fake_dim = 20
    fake_proprio = torch.zeros((1, 12 + 3 * fake_dim))
    sol = AlgSolution()
    out = sol.predicts({"proprio": fake_proprio, "extero": None, "image": {}}, 0.0)
    act = out["action"]
    n = len(act[0]) if act and isinstance(act[0], list) else len(act)
    assert isinstance(out, dict) and "action" in out and "giveup" in out, "返回格式错误"
    print(f"[selfcheck] OK -> giveup={out['giveup']}, action_dim={n} (expect {fake_dim})")
