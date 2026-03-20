import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pickle
from pathlib import Path
import mujoco

gym.register_envs(gymnasium_robotics)

# =========================
# 配置
# =========================
TASK_NAME = "slide cabinet"
SAVE_PATH = "./evaluation_v2/new_demos/scripted/slide_cabinet_scripted.pkl"
RENDER = True
MAX_EPISODE_STEPS = 280

# sites
EE_SITE_NAME = "end_effector"
TARGET_SITE_NAME = "slide_site"

# =========================
# 控制参数
# =========================
DAMPING = 1e-4

# 预抓取：先到把手上方一点
PREGRASP_OFFSET = np.array([0.0, 0.0, 0.02], dtype=np.float32)

PREGRASP_THRESHOLD = 0.10
ALIGN_THRESHOLD = 0.05
GRIPPER_CLOSE_THRESHOLD = 0.06
HANDLE_CENTER_THRESHOLD = 0.05

# 速度
APPROACH_SPEED = 0.18
ALIGN_SPEED = 0.08
PULL_SPEED = 0.16

# 拉动方向：你之前成功过的方向先保留
PULL_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)

# 夹爪动作
GRIPPER_OPEN_ACTION = +1.0
GRIPPER_CLOSE_ACTION = -1.0

# 如果连续若干步 task_error 几乎不下降，就认为“拉不动了”
NO_PROGRESS_EPS = 1e-4
NO_PROGRESS_LIMIT = 15


# =========================
# 环境
# =========================
def make_env():
    env = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=[TASK_NAME],
        render_mode="rgb_array",
    )
    return env


# =========================
# MuJoCo 工具函数
# =========================
def get_site_id(env, site_name: str):
    site_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found.")
    return site_id


def get_site_pos(env, site_name: str):
    site_id = get_site_id(env, site_name)
    return env.unwrapped.data.site_xpos[site_id].copy()


def get_site_rotmat(env, site_name: str):
    """
    返回 site 旋转矩阵，shape=(3,3)
    MuJoCo site_xmat 是展平后的 9 维
    """
    site_id = get_site_id(env, site_name)
    return env.unwrapped.data.site_xmat[site_id].reshape(3, 3).copy()


def get_handle_rel_in_ee_frame(env, ee_site_name: str, target_site_name: str):
    """
    把 handle 在 EE 局部坐标系中的位置算出来：
        handle_rel = R_ee^T (handle_pos - ee_pos)
    这个量很重要，用来判断把手是否真的在夹爪中心附近
    """
    ee_pos = get_site_pos(env, ee_site_name)
    ee_R = get_site_rotmat(env, ee_site_name)
    handle_pos = get_site_pos(env, target_site_name)

    handle_rel = ee_R.T @ (handle_pos - ee_pos)
    return handle_rel.astype(np.float32)


def compute_arm_qvel_from_ee_velocity(env, ee_v_world, site_name, damping=DAMPING):
    """
    ee_v_world: shape=(3,), 末端世界坐标系线速度
    返回 7 维 arm joint velocity
    """
    model = env.unwrapped.model
    data = env.unwrapped.data

    site_id = get_site_id(env, site_name)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    # 前7维对应 Panda arm
    J = jacp[:, :7]   # (3, 7)

    JJt = J @ J.T
    qdot = J.T @ np.linalg.solve(JJt + damping * np.eye(3), ee_v_world)

    return qdot.astype(np.float32)


# =========================
# 任务/观测辅助
# =========================
def get_task_error(obs, task_name):
    ag = np.asarray(obs["achieved_goal"][task_name], dtype=np.float64).reshape(-1)
    dg = np.asarray(obs["desired_goal"][task_name], dtype=np.float64).reshape(-1)
    return np.linalg.norm(ag - dg)


def get_gripper_width_from_obs(obs):
    """
    近似夹爪开口宽度
    observation[7], observation[8] 通常对应左右 finger 位置
    """
    return float(obs["observation"][7] + obs["observation"][8])


def unit_vector(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


# =========================
# 脚本策略（优化版）
# =========================
def scripted_policy(env, obs, phase, state):
    """
    phase:
      0 = pre_grasp   先到把手前方/上方一点
      1 = align       贴近把手
      2 = close       闭合夹爪
      3 = pull        保持闭合并拉动

    重要说明：
    这版仍然是“位置控制”主导，不是完整 6D grasp pose 控制。
    但加入了 handle_rel_ee 这个诊断量，用来判断把手是否在夹爪中心附近。
    """
    ee_pos = get_site_pos(env, EE_SITE_NAME)
    handle_pos = get_site_pos(env, TARGET_SITE_NAME)
    handle_rel_ee = get_handle_rel_in_ee_frame(env, EE_SITE_NAME, TARGET_SITE_NAME)

    gripper_width = get_gripper_width_from_obs(obs)
    task_error = get_task_error(obs, TASK_NAME)

    action = np.zeros(9, dtype=np.float32)

    # 世界系下的接近误差
    approach_error = np.linalg.norm(ee_pos - handle_pos)

    # EE局部系下，把手是否在夹爪中心附近
    handle_center_error = np.linalg.norm(handle_rel_ee)

    # ---------- Phase 0: pre_grasp ----------
    if phase == 0:
        pregrasp_pos = handle_pos + PREGRASP_OFFSET
        pos_error = np.linalg.norm(ee_pos - pregrasp_pos)

        direction = unit_vector(pregrasp_pos - ee_pos)
        ee_v = direction * APPROACH_SPEED
        qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, EE_SITE_NAME)

        action[:7] = np.clip(qdot_arm, -1.0, 1.0)
        action[7] = GRIPPER_OPEN_ACTION
        action[8] = GRIPPER_OPEN_ACTION

        next_phase = 1 if pos_error < PREGRASP_THRESHOLD else 0

    # ---------- Phase 1: align ----------
    elif phase == 1:
        direction = unit_vector(handle_pos - ee_pos)
        ee_v = direction * ALIGN_SPEED
        qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, EE_SITE_NAME)

        action[:7] = np.clip(qdot_arm, -1.0, 1.0)
        action[7] = GRIPPER_OPEN_ACTION
        action[8] = GRIPPER_OPEN_ACTION

        # 进入 close 的条件：
        # 1) 世界坐标距离足够近
        # 2) handle 在 EE 坐标系里也足够靠近中心
        if approach_error < ALIGN_THRESHOLD and handle_center_error < HANDLE_CENTER_THRESHOLD:
            next_phase = 2
        else:
            next_phase = 1

    # ---------- Phase 2: close gripper ----------
    elif phase == 2:
        # arm 基本不动，先尝试夹住
        action[:7] = 0.0
        action[7] = GRIPPER_CLOSE_ACTION
        action[8] = GRIPPER_CLOSE_ACTION

        # 只有在夹爪闭得差不多、且把手仍在中心附近，才进入 pull
        if gripper_width < GRIPPER_CLOSE_THRESHOLD and handle_center_error < HANDLE_CENTER_THRESHOLD:
            next_phase = 3
        else:
            next_phase = 2

    # ---------- Phase 3: pull ----------
    else:
        ee_v = unit_vector(PULL_DIRECTION) * PULL_SPEED
        qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, EE_SITE_NAME)

        action[:7] = np.clip(qdot_arm, -1.0, 1.0)
        action[7] = GRIPPER_CLOSE_ACTION
        action[8] = GRIPPER_CLOSE_ACTION

        next_phase = 3

    info = {
        "phase": phase,
        "ee_pos": ee_pos,
        "handle_pos": handle_pos,
        "handle_rel_ee": handle_rel_ee,
        "gripper_width": gripper_width,
        "task_error": task_error,
        "approach_error": approach_error,
        "handle_center_error": handle_center_error,
        "pregrasp_error": np.linalg.norm(ee_pos - (handle_pos + PREGRASP_OFFSET)),
    }

    return action, next_phase, info


# =========================
# 执行一条 scripted episode
# =========================
def collect_one_episode(env):
    obs, info = env.reset()

    observations = []
    actions = []
    rewards = []
    images = []

    achieved_goals = []
    step_task_completions = []
    episode_task_completions = []
    tasks_to_complete = []

    desired_goal = obs["desired_goal"]

    terminated = False
    truncated = False

    phase = 0
    state = {"no_progress_counter": 0}

    task_errors = []
    approach_errors = []
    phase_history = []
    gripper_widths = []
    handle_rel_ee_history = []
    handle_center_errors = []

    for step in range(MAX_EPISODE_STEPS):
        if RENDER:
            img = env.render()
            images.append(np.asarray(img, dtype=np.uint8))

        observations.append(np.asarray(obs["observation"], dtype=np.float64))
        achieved_goals.append(obs["achieved_goal"])

        old_task_error = get_task_error(obs, TASK_NAME)

        action, phase, policy_info = scripted_policy(env, obs, phase, state)
        next_obs, reward, terminated, truncated, info = env.step(action)

        new_task_error = get_task_error(next_obs, TASK_NAME)
        progress = old_task_error - new_task_error

        # pull 阶段监控“长时间没进展”
        if phase == 3:
            if progress < NO_PROGRESS_EPS:
                state["no_progress_counter"] += 1
            else:
                state["no_progress_counter"] = 0

            if state["no_progress_counter"] >= NO_PROGRESS_LIMIT:
                terminated = True

        actions.append(action)
        rewards.append(float(reward))
        step_task_completions.append(info.get("step_task_completions", []))
        episode_task_completions.append(info.get("episode_task_completions", []))
        tasks_to_complete.append(info.get("tasks_to_complete", []))

        task_errors.append(new_task_error)
        approach_errors.append(policy_info["approach_error"])
        phase_history.append(policy_info["phase"])
        gripper_widths.append(policy_info["gripper_width"])
        handle_rel_ee_history.append(policy_info["handle_rel_ee"])
        handle_center_errors.append(policy_info["handle_center_error"])

        print(
            f"step={step:03d} | "
            f"phase={policy_info['phase']} -> {phase} | "
            f"approach_error={policy_info['approach_error']:.4f} | "
            f"handle_center_error={policy_info['handle_center_error']:.4f} | "
            f"gripper_width={policy_info['gripper_width']:.4f} | "
            f"task_error={new_task_error:.4f} | "
            f"reward={reward:.2f} | "
            f"completed={episode_task_completions[-1]}"
        )

        obs = next_obs

        if terminated or truncated:
            break

    final_completed = episode_task_completions[-1] if len(episode_task_completions) > 0 else []
    success = (TASK_NAME in final_completed) or bool(terminated)

    traj = {
        "task_name": TASK_NAME,
        "display_camera": "default",
        "saved_cameras": ["default"],

        "observations": np.asarray(observations),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "images": np.asarray(images) if len(images) > 0 else None,

        "desired_goal": desired_goal,
        "achieved_goal": achieved_goals,

        "step_task_completions": step_task_completions,
        "episode_task_completions": episode_task_completions,
        "tasks_to_complete": tasks_to_complete,

        "task_errors": np.asarray(task_errors),
        "approach_errors": np.asarray(approach_errors),
        "phase_history": np.asarray(phase_history),
        "gripper_widths": np.asarray(gripper_widths),
        "handle_rel_ee_history": np.asarray(handle_rel_ee_history),
        "handle_center_errors": np.asarray(handle_center_errors),

        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(success),
        "length": len(actions),
    }

    return traj


# =========================
# 主函数
# =========================
def main():
    env = make_env()
    traj = collect_one_episode(env)
    env.close()

    print("\n" + "=" * 80)
    print("Scripted episode finished")
    print("success:", traj["success"])
    print("length:", traj["length"])
    print("terminated:", traj["terminated"])
    print("truncated:", traj["truncated"])
    print("final completed:",
          traj["episode_task_completions"][-1] if traj["length"] > 0 else [])
    print("final task_error:",
          traj["task_errors"][-1] if traj["length"] > 0 else None)
    print("final handle_center_error:",
          traj["handle_center_errors"][-1] if traj["length"] > 0 else None)
    print("=" * 80)

    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    demos = [traj]
    with open(save_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"saved to {save_path}")


if __name__ == "__main__":
    main()