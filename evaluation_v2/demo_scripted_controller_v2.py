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

# 更严格的成功标准（你自己的，不用环境默认 success）
STOP_DELTA_EPS = 8e-4
STOP_DELTA_STEPS = 5

# 准备位：先到把手前上方一点，减少 phase 0 直接顶开柜门
REACH_OFFSET = np.array([-0.030, 0.0, 0.010], dtype=np.float32)

# 阈值
REACH_THRESHOLD = 0.075
GRASP_APPROACH_THRESHOLD = 0.100
HANDLE_CENTER_THRESHOLD = 0.100
GRIPPER_CLOSE_THRESHOLD = 0.060

# 速度
REACH_SPEED = 0.30
GRASP_SPEED = 0.10
PULL_SPEED = 0.14

# 拉动方向
PULL_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)

# gripper
GRIPPER_OPEN_ACTION = +1.0
GRIPPER_CLOSE_ACTION = -1.0

# 抓取稳定判定
GRASP_SETTLE_STEPS = 3

# pull 阶段若长期没进展则停止
NO_PROGRESS_EPS = 1e-4
NO_PROGRESS_LIMIT = 25


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
    site_id = mujoco.mj_name2id(
        env.unwrapped.model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )
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
# 脚本策略
# =========================
def scripted_policy(env, obs, phase, state):
    """
    2-phase scripted policy

    phase:
      0 = approach_and_close   靠近把手的同时闭合夹爪
      1 = closed_push          保持闭合并推动柜门直到打开
    """
    ee_pos = get_site_pos(env, EE_SITE_NAME)
    handle_pos = get_site_pos(env, TARGET_SITE_NAME)
    handle_rel_ee = get_handle_rel_in_ee_frame(env, EE_SITE_NAME, TARGET_SITE_NAME)

    gripper_width = get_gripper_width_from_obs(obs)
    task_error = get_task_error(obs, TASK_NAME)

    action = np.zeros(9, dtype=np.float32)

    handle_center_error = np.linalg.norm(handle_rel_ee)

    # 接近目标：把手稍前/稍上方一点
    approach_target = handle_pos + REACH_OFFSET
    approach_delta = approach_target - ee_pos
    approach_error = np.linalg.norm(approach_delta)

    grasp_error = np.linalg.norm(ee_pos - handle_pos)

    # -------------------------
    # Phase 0: approach_and_close
    # -------------------------
    if phase == 0:
       state["phase0_steps"] += 1
    
       ee_v = 4.0 * approach_delta
    
       # 远时快，近时慢
       if approach_error > 0.15:
           ee_v = np.clip(ee_v, -0.30, 0.30)
       elif approach_error > 0.08:
           ee_v = np.clip(ee_v, -0.16, 0.16)
       else:
           ee_v = np.clip(ee_v, -0.08, 0.08)
    
       qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, EE_SITE_NAME)
    
       action[:7] = np.clip(qdot_arm, -1.0, 1.0)
    
       # 靠近时持续闭合
       action[7] = GRIPPER_CLOSE_ACTION
       action[8] = GRIPPER_CLOSE_ACTION
    
       # 只能靠几何接近程度切换，不能靠 gripper_width
       close_enough = (
           approach_error < REACH_THRESHOLD
           and grasp_error < GRASP_APPROACH_THRESHOLD
       )
       center_ready = (handle_center_error < HANDLE_CENTER_THRESHOLD)
       timeout_ready = (state["phase0_steps"] >= 80)
    
       if close_enough or center_ready or timeout_ready:
           next_phase = 1
           state["phase0_steps"] = 0
       else:
           next_phase = 0

    # -------------------------
    # Phase 1: closed_push
    # -------------------------
    else:
        # 如果离把手太远，说明 phase 0 切早了，退回去
        if grasp_error > 0.25:
            next_phase = 0
            action[:7] = 0.0
            action[7] = GRIPPER_CLOSE_ACTION
            action[8] = GRIPPER_CLOSE_ACTION
        else:
            ee_v = unit_vector(PULL_DIRECTION) * PULL_SPEED
            qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, EE_SITE_NAME)
    
            action[:7] = np.clip(qdot_arm, -1.0, 1.0)
            action[7] = GRIPPER_CLOSE_ACTION
            action[8] = GRIPPER_CLOSE_ACTION
    
            next_phase = 1

    info = {
        "phase": phase,
        "next_phase": next_phase,
        "ee_pos": ee_pos,
        "handle_pos": handle_pos,
        "handle_rel_ee": handle_rel_ee,
        "gripper_width": gripper_width,
        "task_error": task_error,
        "approach_error": approach_error,
        "grasp_error": grasp_error,
        "handle_center_error": handle_center_error,
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
    state = {
        "phase0_steps": 0,
        "stop_counter": 0,
        "ee_R_ref": get_site_rotmat(env, EE_SITE_NAME),
    }

    task_errors = []
    phase_history = []
    gripper_widths = []
    handle_rel_ee_history = []
    handle_center_errors = []
    grasp_errors = []
    approach_errors = []
    drawer_deltas = []
    rot_err_norms = []

    done_by_stop_delta = False
    ever_env_terminated = False

    for step in range(MAX_EPISODE_STEPS):
        if RENDER:
            img = env.render()
            images.append(np.asarray(img, dtype=np.uint8))

        observations.append(np.asarray(obs["observation"], dtype=np.float64))
        achieved_goals.append(obs["achieved_goal"])

        current_phase = phase
        action, next_phase, policy_info = scripted_policy(env, obs, current_phase, state)

        next_obs, reward, env_terminated, env_truncated, info = env.step(action)

        if env_terminated:
            ever_env_terminated = True

        new_task_error = get_task_error(next_obs, TASK_NAME)

        # =========================
        # 核心：计算柜门状态变化量
        # =========================
        old_ag = np.asarray(obs["achieved_goal"][TASK_NAME], dtype=np.float64).reshape(-1)
        new_ag = np.asarray(next_obs["achieved_goal"][TASK_NAME], dtype=np.float64).reshape(-1)
        drawer_delta = np.linalg.norm(new_ag - old_ag)

        # =========================
        # 基于 drawer_delta 的停止准则
        # 只在 pushing 阶段检查
        # =========================
        if current_phase == 1:
            if drawer_delta < STOP_DELTA_EPS:
                state["stop_counter"] += 1
            else:
                state["stop_counter"] = 0

            if state["stop_counter"] >= STOP_DELTA_STEPS:
                done_by_stop_delta = True
        else:
            state["stop_counter"] = 0

        # 只用我们自己的标准停止
        terminated = bool(done_by_stop_delta)
        truncated = bool(env_truncated)

        actions.append(action)
        rewards.append(float(reward))
        step_task_completions.append(info.get("step_task_completions", []))
        episode_task_completions.append(info.get("episode_task_completions", []))
        tasks_to_complete.append(info.get("tasks_to_complete", []))

        task_errors.append(new_task_error)
        phase_history.append(current_phase)
        gripper_widths.append(policy_info["gripper_width"])
        handle_rel_ee_history.append(policy_info["handle_rel_ee"])
        handle_center_errors.append(policy_info["handle_center_error"])
        grasp_errors.append(policy_info["grasp_error"])
        approach_errors.append(policy_info["approach_error"])
        drawer_deltas.append(drawer_delta)
        rot_err_norms.append(policy_info.get("rot_err_norm", 0.0))

        print(
            f"step={step:03d} | "
            f"phase={current_phase} -> {next_phase} | "
            f"approach_error={policy_info['approach_error']:.4f} | "
            f"grasp_error={policy_info['grasp_error']:.4f} | "
            f"handle_center_error={policy_info['handle_center_error']:.4f} | "
            f"rot_err={policy_info.get('rot_err_norm', 0.0):.4f} | "
            f"gripper_width={policy_info['gripper_width']:.4f} | "
            f"task_error={new_task_error:.4f} | "
            f"drawer_delta={drawer_delta:.6f} | "
            f"stop_cnt={state['stop_counter']} | "
            f"env_done={env_terminated} | "
            f"stop_done={done_by_stop_delta} | "
            f"reward={reward:.2f} | "
            f"completed={episode_task_completions[-1]}"
        )

        phase = next_phase
        obs = next_obs

        if terminated or truncated:
            break

    final_completed = episode_task_completions[-1] if len(episode_task_completions) > 0 else []
    final_task_error = task_errors[-1] if len(task_errors) > 0 else np.inf

    env_success = (TASK_NAME in final_completed)

    # 现在 success 的含义：
    # pushing阶段柜门连续多步几乎不再变化，因此认为“推到头了”
    success = bool(done_by_stop_delta)

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
        "grasp_errors": np.asarray(grasp_errors),
        "phase_history": np.asarray(phase_history),
        "gripper_widths": np.asarray(gripper_widths),
        "handle_rel_ee_history": np.asarray(handle_rel_ee_history),
        "handle_center_errors": np.asarray(handle_center_errors),
        "drawer_deltas": np.asarray(drawer_deltas),
        "rot_err_norms": np.asarray(rot_err_norms),

        "terminated": bool(terminated),
        "truncated": bool(truncated),

        "env_success": bool(env_success),
        "success": bool(success),

        "done_by_stop_delta": bool(done_by_stop_delta),
        "ever_env_terminated": bool(ever_env_terminated),
        "stop_counter_final": int(state["stop_counter"]),

        "final_task_error": float(final_task_error),
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
    print("env_success:", traj["env_success"])
    print("delta_stop_success:", traj["success"])
    print("ever_env_terminated:", traj["ever_env_terminated"])
    print("done_by_stop_delta:", traj["done_by_stop_delta"])
    print("stop_counter_final:", traj["stop_counter_final"])
    print("length:", traj["length"])
    print("terminated:", traj["terminated"])
    print("truncated:", traj["truncated"])
    print("final completed:",
          traj["episode_task_completions"][-1] if traj["length"] > 0 else [])
    print("final task_error:", traj["final_task_error"])
    print("final drawer_delta:",
          traj["drawer_deltas"][-1] if traj["length"] > 0 else None)
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