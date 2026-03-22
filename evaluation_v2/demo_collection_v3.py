import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pickle
from pathlib import Path
import pygame
import mujoco

gym.register_envs(gymnasium_robotics)

# =========================
# 可调参数
# =========================
TASK_NAME = "slide cabinet"   # 建议先试 "slide cabinet" 或 "light switch"
DISPLAY_CAMERA = "default"
SAVE_CAMERAS = ["default"]    # 第一阶段先只保存一个
SAVE_PATH = f"./evaluation_v2/new_demos/{SAVE_CAMERAS[0]}/{TASK_NAME}_ee_gamepad.pkl"

CONTROL_HZ = 20
MAX_EPISODE_STEPS = 280
DEADZONE = 0.15

# EE teleop 参数
EE_SITE_NAME = "end_effector"
EE_VEL_SCALE = 0.20
DAMPING = 1e-4

# gripper
GRIP_SCALE = 1.0


# =========================
# 环境
# =========================
def make_env(task_name: str):
    env = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=[task_name],
        render_mode="rgb_array",
    )
    return env


# =========================
# 相机渲染
# =========================
def render_with_camera(env, camera_name=None):
    """
    当前 FrankaKitchen-v1 下，default 视角是最稳的。
    如果 camera_name 不是 default，这里先退回 env.render()。
    后续若确认环境支持其他 camera，可再扩展。
    """
    return env.render()


# =========================
# 手柄初始化
# =========================
def init_joystick():
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("没有检测到手柄，请先连接 Xbox 手柄。")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Detected joystick: {joystick.get_name()}")
    return joystick


def apply_deadzone(v, deadzone=DEADZONE):
    return 0.0 if abs(v) < deadzone else float(v)


# =========================
# Jacobian / EE teleop
# =========================
def get_site_id(env, site_name: str):
    site_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found.")
    return site_id


def compute_arm_qvel_from_ee_velocity(env, ee_v_world, site_name, damping=DAMPING):
    """
    ee_v_world: shape=(3,), 末端世界坐标系线速度
    返回: 7维 arm joint velocity
    """
    model = env.unwrapped.model
    data = env.unwrapped.data

    site_id = get_site_id(env, site_name)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    # 只取前7个关节，对应 Panda arm
    J = jacp[:, :7]   # (3, 7)

    # 阻尼伪逆
    JJt = J @ J.T
    qdot = J.T @ np.linalg.solve(JJt + damping * np.eye(3), ee_v_world)

    return qdot.astype(np.float32)


def get_ee_gamepad_action(env, joystick):
    """
    最简单 EE teleop:
    - 左摇杆: x / y 平移
    - 右摇杆上下: z 平移
    - LT / RT: gripper 开合
    - A: 保存当前 episode
    - B: 丢弃当前 episode
    - START: 退出程序
    - LB: 慢速
    - RB: 快速
    """
    action = np.zeros(9, dtype=np.float32)

    # 摇杆轴
    lx = apply_deadzone(joystick.get_axis(0))
    ly = apply_deadzone(joystick.get_axis(1))
    rx = apply_deadzone(joystick.get_axis(2))
    ry = apply_deadzone(joystick.get_axis(3))

    # trigger
    try:
        lt_raw = joystick.get_axis(4)
        rt_raw = joystick.get_axis(5)

        # 常见兼容写法：若原始范围 [-1,1]，映射到 [0,1]
        lt = (lt_raw + 1) / 2 if -1.1 <= lt_raw <= 1.1 else lt_raw
        rt = (rt_raw + 1) / 2 if -1.1 <= rt_raw <= 1.1 else rt_raw
    except Exception:
        lt, rt = 0.0, 0.0

    # 按键
    btn_a = joystick.get_button(0)
    btn_b = joystick.get_button(1)
    btn_lb = joystick.get_button(4)
    btn_rb = joystick.get_button(5)
    btn_start = joystick.get_button(7)

    # 速度倍率
    vel_scale = EE_VEL_SCALE
    if btn_lb:
        vel_scale *= 0.5
    if btn_rb:
        vel_scale *= 2.0

    # 末端平移速度（世界坐标系）
    # 如果方向不符合你的直觉，只要改符号即可
    ee_v = np.array([
        lx,      # x
        -ly,     # y
        -ry,     # z
    ], dtype=np.float32) * vel_scale

    try:
        qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, EE_SITE_NAME)
    except Exception as e:
        print("EE mapping error:", e)
        qdot_arm = np.zeros(7, dtype=np.float32)

    action[:7] = np.clip(qdot_arm, -1.0, 1.0)

    # gripper: RT 关闭, LT 打开
    grip = (rt - lt) * GRIP_SCALE
    action[7] = grip
    action[8] = grip

    action = np.clip(action, -1.0, 1.0)

    buttons = {
        "save": bool(btn_a),
        "discard": bool(btn_b),
        "quit": bool(btn_start),
        "slow": bool(btn_lb),
        "fast": bool(btn_rb),
    }
    return action, buttons


# =========================
# 画面显示
# =========================
def render_to_pygame(screen, img, font, step_count, task_name, info_text=""):
    surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    screen.blit(surface, (0, 0))

    overlay_lines = [
        f"Task: {task_name} | Step: {step_count}/{MAX_EPISODE_STEPS}",
        f"Camera: {DISPLAY_CAMERA}",
        f"EE site: {EE_SITE_NAME}",
        "Controls:",
        "Left stick  = move x/y",
        "Right stick = move z (up/down)",
        "LT=open gripper | RT=close gripper",
        "LB=slow | RB=fast",
        "A=save & reset | B=discard & reset | START=quit",
    ]
    if info_text:
        overlay_lines.append(info_text)

    y = 8
    for line in overlay_lines:
        text = font.render(line, True, (255, 255, 255))
        bg = pygame.Surface((text.get_width() + 8, text.get_height() + 4))
        bg.set_alpha(160)
        bg.fill((0, 0, 0))
        screen.blit(bg, (8, y))
        screen.blit(text, (12, y + 2))
        y += text.get_height() + 8


# =========================
# 采一条 episode
# =========================
def collect_one_episode(env, screen, font, clock, joystick, task_name: str):
    obs, info = env.reset()

    observations = []
    actions = []
    rewards = []

    # 单视角保存
    if len(SAVE_CAMERAS) == 1:
        images = []
    else:
        images = {cam: [] for cam in SAVE_CAMERAS}

    achieved_goals = []
    step_task_completions = []
    episode_task_completions = []
    tasks_to_complete = []

    desired_goal = obs["desired_goal"]

    terminated = False
    truncated = False
    manual_discard = False
    step_count = 0

    prev_a = False
    prev_b = False
    prev_start = False

    while not (terminated or truncated):
        pygame.event.pump()

        display_img = render_with_camera(env, DISPLAY_CAMERA)
        action, buttons = get_ee_gamepad_action(env, joystick)

        # 边沿触发
        if buttons["quit"] and not prev_start:
            return None, True

        if buttons["save"] and not prev_a:
            terminated = True

        if buttons["discard"] and not prev_b:
            manual_discard = True
            terminated = True

        prev_a = buttons["save"]
        prev_b = buttons["discard"]
        prev_start = buttons["quit"]

        # 保存当前 observation / image / achieved_goal / action
        observations.append(np.asarray(obs["observation"], dtype=np.float64))
        achieved_goals.append(obs["achieved_goal"])
        actions.append(action)

        if len(SAVE_CAMERAS) == 1:
            img = render_with_camera(env, SAVE_CAMERAS[0])
            images.append(np.asarray(img, dtype=np.uint8))
        else:
            for cam in SAVE_CAMERAS:
                img = render_with_camera(env, cam)
                images[cam].append(np.asarray(img, dtype=np.uint8))

        next_obs, reward, terminated_env, truncated_env, info = env.step(action)

        rewards.append(float(reward))
        step_task_completions.append(info.get("step_task_completions", []))
        episode_task_completions.append(info.get("episode_task_completions", []))
        tasks_to_complete.append(info.get("tasks_to_complete", []))

        terminated = terminated or terminated_env
        truncated = truncated or truncated_env

        obs = next_obs
        step_count += 1

        info_text = f"Reward: {reward:.2f} | Completed: {episode_task_completions[-1] if episode_task_completions else []}"
        render_to_pygame(screen, display_img, font, step_count, task_name, info_text)
        pygame.display.flip()
        clock.tick(CONTROL_HZ)

        if step_count >= MAX_EPISODE_STEPS:
            truncated = True

    if manual_discard:
        print("[discarded] episode manually discarded")
        return "DISCARDED", False

    final_completed = episode_task_completions[-1] if len(episode_task_completions) > 0 else []
    success = (task_name in final_completed) or bool(terminated_env if "terminated_env" in locals() else terminated)

    traj = {
        "task_name": task_name,
        "display_camera": DISPLAY_CAMERA,
        "saved_cameras": SAVE_CAMERAS,

        "observations": np.asarray(observations),  # (T, 59)
        "actions": np.asarray(actions),            # (T, 9)
        "rewards": np.asarray(rewards),            # (T,)

        "images": (
            np.asarray(images) if len(SAVE_CAMERAS) == 1
            else {cam: np.asarray(v) for cam, v in images.items()}
        ),

        "desired_goal": desired_goal,
        "achieved_goal": achieved_goals,

        "step_task_completions": step_task_completions,
        "episode_task_completions": episode_task_completions,
        "tasks_to_complete": tasks_to_complete,

        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(success),
        "length": len(actions),
    }

    return traj, False


# =========================
# 主循环
# =========================
def main():
    env = make_env(TASK_NAME)

    pygame.init()
    joystick = init_joystick()

    obs, _ = env.reset()
    first_img = render_with_camera(env, DISPLAY_CAMERA)

    h, w = first_img.shape[:2]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("FrankaKitchen EE Teleop")
    font = pygame.font.SysFont("Menlo", 18)
    clock = pygame.time.Clock()

    save_path = Path(SAVE_PATH)
    if save_path.exists():
        with open(save_path, "rb") as f:
            demos = pickle.load(f)
        print(f"Loaded existing demos: {len(demos)} from {save_path}")
    else:
        demos = []

    episode_idx = len(demos)
    quit_program = False

    print("=" * 80)
    print("EE gamepad teleop started")
    print(f"Task: {TASK_NAME}")
    print(f"Display camera: {DISPLAY_CAMERA}")
    print(f"Save cameras: {SAVE_CAMERAS}")
    print(f"Using EE site: {EE_SITE_NAME}")
    print("Left stick = move x/y")
    print("Right stick up/down = move z")
    print("LT=open gripper | RT=close gripper")
    print("LB=slow | RB=fast")
    print("A=save, B=discard, START=quit")
    print("=" * 80)

    while not quit_program:
        episode_idx += 1
        print(f"\n[episode {episode_idx}] start")

        traj, quit_program = collect_one_episode(
            env, screen, font, clock, joystick, task_name=TASK_NAME
        )

        if quit_program:
            print("Quit requested.")
            break

        if traj == "DISCARDED":
            continue

        demos.append(traj)
        print(f"[episode {episode_idx}] saved")
        print("  success:", traj["success"])
        print("  length:", traj["length"])
        print("  terminated:", traj["terminated"])
        print("  truncated:", traj["truncated"])
        print("  final completed:",
              traj["episode_task_completions"][-1] if traj["length"] > 0 else [])

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(demos, f)
        print(f"  autosaved {len(demos)} demos -> {save_path}")

    env.close()
    pygame.quit()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"\nFinal saved {len(demos)} demos to {save_path}")


if __name__ == "__main__":
    main()