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

        # --- Task D box-push 导航状态 ---
        # 允许通过环境变量预设任务类型，跳过检测（避免前 N 步走错路线）
        _force_task = os.environ.get("ATEC_TASK", "").upper()
        self._task_detected = _force_task if _force_task in ("A", "D") else None
        self._first_score = None         # 第一次拿到非零分的分数
        self._first_score_step = None    # 第一次拿到非零分的步数
        self._cumulative_vel_y = 0.0     # 累计 y 方向位移估计
        self._cumulative_vel_x = 0.0     # 累计 x 方向位移估计
        self._est_x = -3.0              # 估计的 x 位置（机器人初始 x=-3）
        self._est_y = 0.0               # 估计的 y 位置（机器人初始 y=0）
        self._box_pushed = False         # 箱子已推到位（分数 >= 14）
        self._box_pushed_step = None     # 推到位的步数
        self._prev_score = 0.0           # 上一步的分数

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
            # Reset Task D state (preserve forced task if set via env var)
            _force_task = os.environ.get("ATEC_TASK", "").upper()
            self._task_detected = _force_task if _force_task in ("A", "D") else None
            self._first_score = None
            self._first_score_step = None
            self._cumulative_vel_y = 0.0
            self._cumulative_vel_x = 0.0
            self._est_x = -3.0
            self._est_y = 0.0
            self._box_pushed = False
            self._box_pushed_step = None
            self._prev_score = 0.0
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
                safe_dim = action_dim if action_dim > 0 else self._DEFAULT_ACTION_DIM
                return {"action": self._zero_action(proprio, safe_dim),
                        "giveup": False}

            if self.policy is None:
                return {"action": self._zero_action(proprio, action_dim), "giveup": False}

            # --- Task D 导航逻辑 ---
            self._update_task_detection(current_score)
            self._update_position_estimate(proprio)
            self._update_vel_cmd_for_task()

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
            except Exception:
                return {"action": [[0.0] * self._DEFAULT_ACTION_DIM], "giveup": False}

    # ------------------------------------------------------------------ #
    # Task D 导航
    # ------------------------------------------------------------------ #
    def _update_task_detection(self, current_score):
        """通过 current_score 跳变模式区分 Task A vs Task D，并检测 box-push 完成。

        Task D 特征：得分大幅跳变（14.0 箱子 reward 或 2.0 机器人 reward）。
        Task A 特征：分数持续增长（RewardA 是连续的距离 reward）。

        策略：默认先按 Task D 导航（后退+侧移+推箱子）。如果在 step 200 之前
        拿到小分（< 1.5 且多次增长），判定为 Task A 切回直走。
        """
        # 检测箱子推到位（分数跳变 >= 10，即 14 分的 box reward）
        if not self._box_pushed and current_score - self._prev_score >= 10.0:
            self._box_pushed = True
            self._box_pushed_step = self._step_count
            self._task_detected = "D"
            print(f"[solution] Box pushed! score jumped from {self._prev_score:.1f} to "
                  f"{current_score:.1f} at step {self._step_count}")

        self._prev_score = current_score

        if self._task_detected is not None:
            return  # 已确定

        # 首次收到非零分
        if current_score > 0.005 and self._first_score is None:
            self._first_score = current_score
            self._first_score_step = self._step_count
            # 立即判定：任何小分（< 1.5）= Task A，大分（>= 1.5）= Task D
            if current_score < 1.5:
                self._task_detected = "A"
                print(f"[solution] Task detected: A (first_score={current_score:.3f} "
                      f"at step {self._step_count})")
                return
            else:
                self._task_detected = "D"
                print(f"[solution] Task detected: D (first_score={current_score:.3f} "
                      f"at step {self._step_count})")
                return

        # 如果到了 step 400 还没拿分，默认当 Task D（Task A 通常很快就有分）
        if self._step_count >= 400 and self._first_score is None:
            self._task_detected = "D"
            print(f"[solution] Task assumed: D (no score by step {self._step_count})")

    def _update_position_estimate(self, proprio):
        """用 base_lin_vel (proprio[:, 0:3]) 做简单积分估计位置。

        base_lin_vel 是机器人坐标系的线速度。因为我们主要直走/斜走，
        yaw 变化不大，近似为世界坐标系的速度即可。
        dt ≈ 0.02s（50Hz 控制频率）。
        """
        try:
            base_lin_vel = proprio[0, 0:3].cpu().tolist()  # [vx, vy, vz]
            dt = 0.02  # 控制周期
            self._est_x += base_lin_vel[0] * dt
            self._est_y += base_lin_vel[1] * dt
        except Exception:
            pass

    def _update_vel_cmd_for_task(self):
        """根据检测到的任务和当前阶段更新速度命令。

        Task A: 保持 vx=1.5 直走不变。
        Task D: 分阶段导航推箱子。

        Task D 导航策略（v3 — 先后退再侧移再推）：
        - 箱子初始位置 (-3, 1.6, 0.5)，机器人初始 (-3, 0, 0.8)
        - 箱子 0.8(x) x 1.0(y) x 0.6(z)，摩擦高(0.9/0.8)，8kg
        - 问题：机器人和箱子初始 x 相同(-3)，侧移时会漂移到 x > -3，
          导致前进时已经超过箱子，无法从后面推
        - 解决：先后退到 x < -3.5，再侧移对齐 y=1.6，再全速前进推箱子
        - 目标：RewardBoxXInRange +14 + RewardCrossX(x>-1.4) +2 = 16 分
        """
        if self._task_detected == "A":
            # Task A: 用环境变量或默认 1.5 直走
            _vx_a = float(os.environ.get("ATEC_VEL_X", "1.5"))
            self._vel_cmd = torch.tensor(
                [_vx_a, 0.0, 0.0], device=self.device, dtype=torch.float32
            ).view(1, 3)
            return

        # Task D 或未确定（默认按 Task D 导航）
        step = self._step_count
        est_y = self._est_y
        est_x = self._est_x

        BOX_Y = 1.6    # 箱子中心 y
        BOX_X = -3.0    # 箱子初始 x 中心
        # 箱子 y 范围约 [1.1, 2.1]（宽度 1.0），x 范围约 [-3.4, -2.6]（宽度 0.8）
        # 机器人 B2 body 约 0.6m(x) x 0.4m(y)

        # 箱子已推到位后：绕过箱子继续前进拿 RewardCrossX(x>-1.4) +2 分
        if self._box_pushed:
            steps_since_push = step - (self._box_pushed_step or step)
            if steps_since_push < 150 and est_y > 0.5:
                # Phase 3a: 向 -y 方向侧移，绕过箱子
                # 箱子在 y≈1.6 附近，宽 1.0m，y 范围 [1.1, 2.1]
                # 需要移到 y < 0.5 才能清晰绕过
                # 注意：侧移太快会摔！保持中等速度
                vx = 0.5   # 中速前进保持稳定
                vy = -0.6  # 稳定侧移（训练范围内），不要太快以免摔倒
                vyaw = 0.0
            else:
                # Phase 3b: 已经绕过箱子，全速前进过 x=-1.4
                vx = 1.5
                vy = 0.0
                vyaw = 0.0
        elif step < 40:
            # Phase 0: 后退+侧移同时进行
            # 需要退到箱子后面 (x < -3.7) 并开始靠近 y=1.6
            vx = -0.8
            vy = 0.8   # 同时大幅侧移
            vyaw = 0.0
        elif est_y < 1.4:
            # Phase 1: 继续侧移到精确对齐箱子中心 y=1.6
            # 保持在箱子后方
            vx = -0.3 if est_x > BOX_X - 0.7 else 0.0
            vy = 0.8
            vyaw = 0.0
        elif est_y > 2.2:
            # 太偏了，修正回来
            vx = 0.0
            vy = -0.5
            vyaw = 0.0
        else:
            # Phase 2: 已对齐 y ∈ [1.4, 2.2]，前进推箱子
            # 用中等速度推，太快会跳过箱子或失稳
            # 箱子需要推 1.6m (从 x=-3 到 x=-1.4)
            vx = 1.0   # 中速推，给箱子施加稳定持续的力
            # P 控制器保持 y 对齐箱子中心
            y_err = BOX_Y - est_y
            vy = max(-0.3, min(0.3, y_err * 2.0))  # 增大增益保持对齐
            vyaw = 0.0

        self._vel_cmd = torch.tensor(
            [vx, vy, vyaw], device=self.device, dtype=torch.float32
        ).view(1, 3)

        # 每 50 步打印一次状态
        if step % 50 == 0:
            phase = "bypass" if self._box_pushed else "push"
            print(f"[solution] TaskD step={step} est_pos=({est_x:.2f}, {est_y:.2f}) "
                  f"vel_cmd=({vx:.1f}, {vy:.2f}) phase={phase} "
                  f"task={self._task_detected} score={self._prev_score:.1f}")

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

    # Test 1: 基本接口
    out = sol.predicts({"proprio": fake_proprio, "extero": None, "image": {}}, 0.0)
    act = out["action"]
    n = len(act[0]) if act and isinstance(act[0], list) else len(act)
    assert isinstance(out, dict) and "action" in out and "giveup" in out, "返回格式错误"
    print(f"[selfcheck] OK -> giveup={out['giveup']}, action_dim={n} (expect {fake_dim})")

    # Test 2: Task D 导航逻辑（模拟 200 步）
    print("[selfcheck] Testing Task D navigation phases...")
    sol2 = AlgSolution()
    for step in range(200):
        # 模拟 proprio 带 base_lin_vel
        fp = torch.zeros((1, 12 + 3 * fake_dim))
        fp[0, 0] = 0.5   # vx
        fp[0, 1] = 0.8   # vy
        score = 0.0
        if step > 150:
            score = 2.0  # 模拟 Task D 跳分
        out2 = sol2.predicts({"proprio": fp, "extero": None, "image": {}}, score)
        assert "action" in out2, f"step {step}: 返回格式错误"
    print(f"[selfcheck] Task D nav OK: task={sol2._task_detected}, "
          f"est_pos=({sol2._est_x:.2f}, {sol2._est_y:.2f}), "
          f"final vel_cmd={sol2._vel_cmd.tolist()}")

    # Test 3: Task A 检测（小分连续增长）
    print("[selfcheck] Testing Task A detection...")
    sol3 = AlgSolution()
    for step in range(100):
        fp = torch.zeros((1, 12 + 3 * fake_dim))
        score = step * 0.01  # Task A 连续增长
        sol3.predicts({"proprio": fp, "extero": None, "image": {}}, score)
    assert sol3._task_detected == "A", f"Expected task A, got {sol3._task_detected}"
    print(f"[selfcheck] Task A detection OK: task={sol3._task_detected}")

    print("[selfcheck] ALL TESTS PASSED")
