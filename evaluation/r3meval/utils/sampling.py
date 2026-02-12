# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import numpy as np
from evaluation.r3meval.utils.gym_env import GymEnv
from evaluation.r3meval.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)
import gc
from collections import namedtuple

# from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
#                             ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)


# Single core rollout to sample trajectories
# =======================================================
# def do_rollout(
#         num_traj,
#         env,
#         policy,
#         eval_mode = False,
#         horizon = 1e6,
#         base_seed = None,
#         env_kwargs=None,
# ):
#     """
#     :param num_traj:    number of trajectories (int)
#     :param env:         environment (env class, str with env_name, or factory function)
#     :param policy:      policy to use for action selection
#     :param eval_mode:   use evaluation mode for action computation (bool)
#     :param horizon:     max horizon length for rollout (<= env.horizon)
#     :param base_seed:   base seed for rollouts (int)
#     :param env_kwargs:  dictionary with parameters, will be passed to env generator
#     :return:
#     """
#     # get the correct env behavior
#     print("Evaluating")
#     if type(env) == str:
#         ## MetaWorld specific stuff
#         if "v2" in env:
#             env_name = env
#             env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
#             env._freeze_rand_vec =False
#             env.horizon = 500
#             env.spec = namedtuple('spec', ['id', 'max_episode_steps', 'observation_dim', 'action_dim'])
#             env.spec.id = env_name
#             env.spec.observation_dim = int(env.observation_space.shape[0])
#             env.spec.action_dim = int(env.action_space.shape[0])
#             env.spec.max_episode_steps = 500
#         else:
#             env = GymEnv(env)
#     elif isinstance(env, GymEnv):
#         env = env
#     elif callable(env):
#         env = env(**env_kwargs)
#     else:
#         # print("Unsupported environment format")
#         # raise AttributeError
#         ## Support passing in one env for everything
#         env = env
#
#     if base_seed is not None:
#         try:
#             env.set_seed(base_seed)
#         except:
#             env.seed(base_seed)
#         np.random.seed(base_seed)
#     else:
#         np.random.seed()
#     # horizon = min(horizon, env.horizon)
#     paths = []
#
#     ep = 0
#     while ep < num_traj:
#         # seeding
#         if base_seed is not None:
#             seed = base_seed + ep
#             try:
#                 env.set_seed(seed)
#             except:
#                 env.seed(seed)
#             np.random.seed(seed)
#
#         observations=[]
#         actions=[]
#         rewards=[]
#         agent_infos = []
#         env_infos = []
#
#         o = env.reset()
#         done = False
#         t = 0
#         ims = []
#         try:
#             ims.append(env.env.env.get_image())
#         except:
#             ## For state based learning
#             pass
#
#         ## MetaWorld vs. Adroit/Kitchen syntax
#         try:
#             init_state = env.__getstate__()
#         except:
#             init_state = env.get_env_state()
#
#         while t < horizon and done != True:
#             a, agent_info = policy.get_action(o)
#             if eval_mode:
#                 a = agent_info['evaluation']
#
#             next_o, r, done, env_info_step = env.step(a)
#             env_info = env_info_step #if env_info_base == {} else env_info_base
#             observations.append(o)
#             actions.append(a)
#             rewards.append(r)
#             try:
#                 ims.append(env.env.env.get_image())
#             except:
#                 pass
#             agent_infos.append(agent_info)
#             env_infos.append(env_info)
#             o = next_o
#             t += 1
#
#         path = dict(
#             observations=np.array(observations),
#             actions=np.array(actions),
#             rewards=np.array(rewards),
#             agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
#             env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
#             terminated=done,
#             init_state = init_state,
#             images=ims
#         )
#
#         paths.append(path)
#         ep += 1
#
#     del(env)
#     gc.collect()
#     return paths
def unwrap_like_gym(env, max_depth=10):
    """
    Try to peel wrappers: .unwrapped (gym) or repeated .env (mjrl GymEnv often uses .env)
    Return the deepest env object.
    """
    cur = env
    for _ in range(max_depth):
        if hasattr(cur, "unwrapped"):
            # gym.Env has unwrapped property
            cur = cur.unwrapped
            continue
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        break
    return cur

def iter_env_candidates(env):
    """
    Yield possible places where _get_obs / get_obs may exist.
    Order matters: prefer the deepest first.
    """
    seen = set()
    cands = []

    # common wrapper chain: env, env.env, env.env.env, ...
    cur = env
    for _ in range(6):
        if cur is None: break
        cands.append(cur)
        cur = getattr(cur, "env", None)

    # gym unwrapped
    try:
        cands.append(env.unwrapped)
    except Exception:
        pass

    # deepest
    cands.append(unwrap_like_gym(env))

    for c in cands:
        if c is None:
            continue
        if id(c) in seen:
            continue
        seen.add(id(c))
        yield c

def slice_env_state(env_state_seq, t=0):
    """
    env_state_seq: dict of arrays, e.g. {'qpos':(T,29), 'qvel':(T,29), ...}
    return: dict of single-frame arrays, e.g. {'qpos':(29,), 'qvel':(29,), ...}
    """
    assert isinstance(env_state_seq, dict), type(env_state_seq)
    st = {}
    for k, v in env_state_seq.items():
        v = np.asarray(v)
        # v could be (T, ...) or scalar/() depending on key
        if v.shape == ():  # numpy scalar
            st[k] = v
        else:
            assert v.shape[0] > t, (k, v.shape, t)
            st[k] = v[t]
    return st

def env_has_fn(obj, name):
    return hasattr(obj, name) and callable(getattr(obj, name))

def set_env_to_demo_init(env, demo_path, t0=0, verbose=True):
    """
    Try to set env to the demo initial physical state.
    Priority:
      1) env.unwrapped.set_env_state(state_dict)  (most complete)
      2) env.unwrapped.set_state(qpos, qvel) or env.unwrapped.set_state(qpos, qvel) like signature
    Returns: True/False indicating whether we believe the state set succeeded.
    """
    # u = env.unwrapped
    u = unwrap_like_gym(env)

    if "env_infos" not in demo_path or "env_state" not in demo_path["env_infos"]:
        if verbose: print("[WARN] demo has no env_infos.env_state, cannot align init state")
        return False

    env_state_seq = demo_path["env_infos"]["env_state"]
    st0 = slice_env_state(env_state_seq, t=t0)

    if env_has_fn(u, "set_env_state"):
        try:
            u.set_env_state(st0)
            if verbose: print("[OK] set_env_state(st0) success")
            return True
        except Exception as e:
            if verbose: print("[FAIL] set_env_state:", repr(e))

    qpos = st0.get("qpos", None)
    qvel = st0.get("qvel", None)
    if qpos is None or qvel is None:
        if verbose: print("[FAIL] no qpos/qvel in st0 keys:", st0.keys())
        return False

    if env_has_fn(u, "set_state"):
        try:
            u.set_state(qpos, qvel)
            if verbose: print("[OK] set_state(qpos,qvel) success")
            return True
        except Exception as e:
            if verbose: print("[FAIL] set_state:", repr(e))

    if verbose: print("[FAIL] no usable state setter found")
    return False

def get_solved_flag(info):
    """
    Kitchen mj_envs often returns info with keys like:
      info['solved'] : bool
      info['rwd_sparse'], info['rwd_dense'], info['done']
    But in your saved paths, solved is inside env_infos arrays.
    During step(), we read from info dict returned by env.step().
    """
    if not isinstance(info, dict):
        return False
    if "solved" in info:
        return bool(info["solved"])
    # sometimes nested:
    if "env_infos" in info and isinstance(info["env_infos"], dict) and "solved" in info["env_infos"]:
        return bool(info["env_infos"]["solved"])
    return False

def make_demo_reset_fn(demos, t0=0, clip_action=True):
    def _reset(env, ep=0, base_seed=None):
        # 让采样可复现
        if base_seed is not None:
            rng = np.random.RandomState(base_seed + ep)
            idx = rng.randint(len(demos))
        else:
            idx = np.random.randint(len(demos))

        demo = demos[idx]

        o = env.reset()

        ok = set_env_to_demo_init(env, demo, t0=t0, verbose=False)
        if not ok:
            return o  # 失败就用普通 reset 的 obs

        # state 对齐后，重新取一次 obs（尽量不再 reset）
        for obj in iter_env_candidates(env):
            for getter in ("get_obs", "_get_obs", "get_observation", "_get_observation"):
                if hasattr(obj, getter) and callable(getattr(obj, getter)):
                    try:
                        return getattr(obj, getter)()
                    except Exception:
                        pass

        return o

    return _reset

def do_rollout(
        num_traj,
        env,
        policy,
        eval_mode=False,
        horizon=1e6,
        base_seed=None,
        env_kwargs=None,
        reset_fn=None,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """
    # get the correct env behavior
    print("Evaluating")
    if type(env) == str:
        ## MetaWorld specific stuff
        if "v2" in env:
            env_name = env
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
            env._freeze_rand_vec = False
            env.horizon = 500
            env.spec = namedtuple('spec', ['id', 'max_episode_steps', 'observation_dim', 'action_dim'])
            env.spec.id = env_name
            env.spec.observation_dim = int(env.observation_space.shape[0])
            env.spec.action_dim = int(env.action_space.shape[0])
            env.spec.max_episode_steps = 500
        else:
            env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        # print("Unsupported environment format")
        # raise AttributeError
        ## Support passing in one env for everything
        env = env

    if base_seed is not None:
        try:
            env.set_seed(base_seed)
        except:
            env.seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    # horizon = min(horizon, env.horizon)
    paths = []

    ep = 0
    while ep < num_traj:
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            try:
                env.set_seed(seed)
            except:
                env.seed(seed)
            np.random.seed(seed)

        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        # o = env.reset()
        if reset_fn is None:
            o = env.reset()
        else:
            o = reset_fn(env, ep=ep, base_seed=base_seed)  # 返回 obs

        done = False
        t = 0
        ims = []
        try:
            ims.append(env.env.env.get_image())
        except:
            ## For state based learning
            pass

        ## MetaWorld vs. Adroit/Kitchen syntax
        try:
            init_state = env.__getstate__()
        except:
            init_state = env.get_env_state()

        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info['evaluation']

            next_o, r, done, env_info_step = env.step(a)
            env_info = env_info_step  # if env_info_base == {} else env_info_base
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            try:
                ims.append(env.env.env.get_image())
            except:
                pass
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done,
            init_state=init_state,
            images=ims
        )

        paths.append(path)
        ep += 1

    del (env)
    gc.collect()
    return paths

def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        reset_fn=None,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs, reset_fn=reset_fn)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs, reset_fn=reset_fn)
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )

    return paths

def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
    
    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)

    pool.close()
    pool.terminate()
    pool.join()  
    return results
