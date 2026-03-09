# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Wrapper around a gym env that provides convenience functions
"""

# import gym
import gymnasium as gym
import numpy as np


class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon


class GymEnv(object):
    def __init__(self, env, env_kwargs=None,
                 obs_mask=None, act_repeat=1,
                 *args, **kwargs):

        # get the correct env behavior
        if type(env) == str:
            env = gym.make(env)
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        self.env = env
        self.env_id = env.unwrapped.spec.id
        self.act_repeat = act_repeat

        try:
            self._horizon = env.spec.max_episode_steps
        except AttributeError:
            self._horizon = env.spec._horizon

        assert self._horizon % act_repeat == 0
        self._horizon = self._horizon // self.act_repeat

        try:
            self._action_dim = self.env.action_space.shape[0]
        except AttributeError:
            self._action_dim = self.env.unwrapped.action_dim

        try:
            self._observation_dim = self.env.observation_space.shape[0]
        except AttributeError:
            self._observation_dim = self.env.unwrapped.obs_dim

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

        # obs mask
        self.obs_mask = np.ones(self._observation_dim) if obs_mask is None else obs_mask

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    # def reset(self, seed=None):
    #     try:
    #         self.env._elapsed_steps = 0
    #         return self.env.unwrapped.reset_model(seed=seed)
    #     except:
    #         if seed is not None:
    #             self.set_seed(seed)
    #         return self.env.reset()
    def reset(self, seed=None):
        # gymnasium: reset returns (obs, info)
        if seed is not None:
            out = self.env.reset(seed=seed)
        else:
            out = self.env.reset()

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs = out  # compatible with old env

        try:
            return self.obs_mask * obs
        except Exception:
            return obs

    def reset_model(self, seed=None):
        # overloading for legacy code
        return self.reset(seed)

    # def step(self, action):
    #     action = action.clip(self.action_space.low, self.action_space.high)
    #     if self.act_repeat == 1:
    #         obs, cum_reward, done, ifo = self.env.step(action)
    #     else:
    #         cum_reward = 0.0
    #         for i in range(self.act_repeat):
    #             obs, reward, done, ifo = self.env.step(action)
    #             cum_reward += reward
    #             if done: break
    #     return self.obs_mask * obs, cum_reward, done, ifo
    def step(self, action):
        action = action.clip(self.action_space.low, self.action_space.high)

        if self.act_repeat == 1:
            out = self.env.step(action)

            if isinstance(out, tuple) and len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = out

            cum_reward = reward

        else:
            cum_reward = 0.0
            done = False
            info = {}
            obs = None

            for _ in range(self.act_repeat):
                out = self.env.step(action)

                if isinstance(out, tuple) and len(out) == 5:
                    obs, reward, terminated, truncated, info = out
                    d = bool(terminated or truncated)
                else:
                    obs, reward, d, info = out

                cum_reward += reward
                done = done or d
                if done:
                    break

        try:
            return self.obs_mask * obs, cum_reward, done, info
        except Exception:
            return obs, cum_reward, done, info

    def render(self):
        try:
            self.env.unwrapped.mujoco_render_frames = True
            self.env.unwrapped.mj_render()
        except:
            self.env.render()

    def set_seed(self, seed=123):
        """
        Gymnasium-compatible seeding.
        Prefer reset(seed=seed) over legacy seed() / _seed().
        """
        try:
            self.env.reset(seed=seed)
            return
        except TypeError:
            pass
        except Exception:
            pass

        inner = getattr(self.env, "env", None)
        if inner is not None:
            try:
                inner.reset(seed=seed)
                return
            except TypeError:
                pass
            except Exception:
                pass

        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        if hasattr(self.env, "_seed"):
            return self.env._seed(seed)

        if inner is not None:
            if hasattr(inner, "seed"):
                return inner.seed(seed)
            if hasattr(inner, "_seed"):
                return inner._seed(seed)

        raise AttributeError("No compatible seeding method found for wrapped env.")

    def seed(self, seed=123):
        # legacy alias for older code that calls env.seed(...)
        return self.set_seed(seed)

    def get_obs(self):
        try:
            return self.obs_mask * self.env.get_obs()
        except:
            return self.obs_mask * self.env._get_obs()

    def get_env_infos(self):
        try:
            return self.env.unwrapped.get_env_infos()
        except:
            return {}

    # ===========================================
    # Trajectory optimization related
    # Envs should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.unwrapped.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.unwrapped.set_env_state(state_dict)
        except:
            self.env.unwrapped.__setstate__(state_dict)
            # raise NotImplementedError

    def real_env_step(self, bool_val):
        try:
            self.env.unwrapped.real_step = bool_val
        except:
            raise NotImplementedError

    # ===========================================

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        try:
            self.env.unwrapped.visualize_policy(policy, horizon, num_episodes, mode)
        except:
            for ep in range(num_episodes):
                o = self.reset()
                d = False
                t = 0
                score = 0.0
                while t < horizon and d is False:
                    a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                    o, r, d, _ = self.step(a)
                    score = score + r
                    self.render()
                    t = t+1
                print("Episode score = %f" % score)

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_env_state=None,
                        terminate_at_done=True,
                        seed=123):

        self.set_seed(seed)
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        for ep in range(num_episodes):
            self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs()
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]
