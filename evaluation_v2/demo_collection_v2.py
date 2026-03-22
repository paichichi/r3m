import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pickle
from pathlib import Path
import pygame

gym.register_envs(gymnasium_robotics)

# =========================
# 可调参数
# =========================
TASK_NAME = "microwave"
SAVE_PATH = f"./evaluation_v2/new_demos/{TASK_NAME}_gamepad.pkl"

DISPLAY_CAMERA = "default"
SAVE_CAMERAS = "default" 

CONTROL_HZ = 20
ACTION_MAG = 0.6
FAST_ACTION_MAG = 1.0
MAX_EPISODE_STEPS = 280
DEADZONE = 0.15


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
    尽量显式指定 camera 渲染。
    某些版本支持 mujoco_renderer.render(camera_name=...)
    如果失败就退回 env.render()
    """
    if camera_name is None:
        return env.render()

    # 优先尝试 unwrapped renderer
    try:
        return env.unwrapped.mujoco_renderer.render(
            render_mode="rgb_array",
            camera_name=camera_name
        )
    except Exception:
        pass

    # 退回默认 render
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
# Xbox 手柄 -> 9维 action
# =========================
def get_gamepad_action(joystick, mode_idx):
    """
    mode 0:
      left stick  -> joint1, joint2
      right stick -> joint3, joint4

    mode 1:
      left stick  -> joint5, joint6
      right stick x -> joint7
      right stick y -> unused

    triggers:
      LT/RT -> gripper

    备注：
    pygame 下不同平台 triggers 轴编号可能不同，
    这里给一个常见映射，后面如果不对你再校准。
    """
    action = np.zeros(9, dtype=np.float32)

    # 常见 Xbox 映射（macOS 下可能略有不同）
    lx = apply_deadzone(joystick.get_axis(0))
    ly = apply_deadzone(joystick.get_axis(1))
    rx = apply_deadzone(joystick.get_axis(2))
    ry = apply_deadzone(joystick.get_axis(3))

    # triggers 常见做法：LT axis=4, RT axis=5
    # 有些系统上范围是 [-1,1]，这里做兼容处理
    try:
        lt_raw = joystick.get_axis(4)
        rt_raw = joystick.get_axis(5)
        lt = (lt_raw + 1) / 2 if lt_raw < 0.99 else lt_raw
        rt = (rt_raw + 1) / 2 if rt_raw < 0.99 else rt_raw
    except Exception:
        lt, rt = 0.0, 0.0

    # 按键：A/B/LB/RB/START
    # 常见 Xbox button 编号（可能随系统略变）
    btn_a = joystick.get_button(0)
    btn_b = joystick.get_button(1)
    btn_lb = joystick.get_button(4)
    btn_rb = joystick.get_button(5)
    btn_start = joystick.get_button(7)

    mag = FAST_ACTION_MAG if joystick.get_button(9) else ACTION_MAG  # L3 之类可选

    if mode_idx == 0:
        action[0] = mag * lx
        action[1] = -mag * ly
        action[2] = mag * rx
        action[3] = -mag * ry

    elif mode_idx == 1:
        action[4] = mag * lx
        action[5] = -mag * ly
        action[6] = mag * rx

    # gripper：RT 关闭，LT 打开（可按你习惯调）
    grip = rt - lt
    action[7] = grip
    action[8] = grip

    action = np.clip(action, -1.0, 1.0)

    buttons = {
        "save": bool(btn_a),
        "discard": bool(btn_b),
        "mode_left": bool(btn_lb),
        "mode_right": bool(btn_rb),
        "quit": bool(btn_start),
    }
    return action, buttons


# =========================
# 画面显示
# =========================
def render_to_pygame(screen, img, font, step_count, task_name, mode_idx, info_text=""):
    surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    screen.blit(surface, (0, 0))

    overlay_lines = [
        f"task: {task_name}",
        f"step: {step_count}",
        f"camera: {DISPLAY_CAMERA}",
        f"mode: {mode_idx}  (LB/RB switch)",
        "A=save, B=discard, START=quit",
        "LT=open gripper, RT=close gripper",
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

    # 如果你要单视角，images 就是数组列表
    # 如果多视角，建议存 dict of list
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
    manual_save = False
    manual_discard = False
    step_count = 0

    mode_idx = 0
    prev_lb = False
    prev_rb = False
    prev_a = False
    prev_b = False
    prev_start = False

    while not (terminated or truncated):
        pygame.event.pump()

        # 显示视角
        display_img = render_with_camera(env, DISPLAY_CAMERA)

        action, buttons = get_gamepad_action(joystick, mode_idx)

        # 边沿触发：避免长按重复
        if buttons["quit"] and not prev_start:
            return None, True

        if buttons["save"] and not prev_a:
            manual_save = True
            terminated = True

        if buttons["discard"] and not prev_b:
            manual_discard = True
            terminated = True

        if buttons["mode_left"] and not prev_lb:
            mode_idx = (mode_idx - 1) % 2

        if buttons["mode_right"] and not prev_rb:
            mode_idx = (mode_idx + 1) % 2

        prev_lb = buttons["mode_left"]
        prev_rb = buttons["mode_right"]
        prev_a = buttons["save"]
        prev_b = buttons["discard"]
        prev_start = buttons["quit"]

        # 存 observation / image / achieved_goal / action
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

        # step
        next_obs, reward, terminated_env, truncated_env, info = env.step(action)

        rewards.append(float(reward))
        step_task_completions.append(info.get("step_task_completions", []))
        episode_task_completions.append(info.get("episode_task_completions", []))
        tasks_to_complete.append(info.get("tasks_to_complete", []))

        terminated = terminated or terminated_env
        truncated = truncated or truncated_env

        obs = next_obs
        step_count += 1

        info_text = f"reward={reward:.2f}, completed={episode_task_completions[-1] if episode_task_completions else []}"
        render_to_pygame(screen, display_img, font, step_count, task_name, mode_idx, info_text)
        pygame.display.flip()
        clock.tick(CONTROL_HZ)

        if step_count >= MAX_EPISODE_STEPS:
            truncated = True

    if manual_discard:
        print("[discarded] episode manually discarded")
        return "DISCARDED", False

    final_completed = episode_task_completions[-1] if len(episode_task_completions) > 0 else []
    success = (task_name in final_completed) or bool(terminated and task_name in final_completed)

    traj = {
        "task_name": task_name,
        "display_camera": DISPLAY_CAMERA,
        "saved_cameras": SAVE_CAMERAS,

        "observations": np.asarray(observations),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),

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
    pygame.display.set_caption("FrankaKitchen Gamepad Teleop Collector")
    font = pygame.font.SysFont("Menlo", 18)
    clock = pygame.time.Clock()

    # 自动续写已有 demo
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
    print("Gamepad teleop started")
    print(f"Task: {TASK_NAME}")
    print(f"Display camera: {DISPLAY_CAMERA}")
    print(f"Save cameras: {SAVE_CAMERAS}")
    print("A=save, B=discard, LB/RB=switch mode, START=quit")
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