# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType, WrapperObsType
from gymnasium.spaces import Box, Dict
import omegaconf
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle
from torchvision.utils import save_image
import hydra


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def _get_embedding(embedding_name='resnet34', load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == 'resnet34':
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet50':
        model = models.resnet50(pretrained=prt, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim


class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """

    def __init__(self, env, embedding_name=None, device='cuda', load_path="", proprio=0, camera_name=None,
                 env_name=None):
        gym.ObservationWrapper.__init__(self, env)

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False

        # 先决定最终 device（CPU 模式就彻底 CPU）
        if device == 'cuda' and torch.cuda.is_available():
            print('Using CUDA.')
            self.device = torch.device('cuda')
        else:
            print('Not using CUDA.')
            self.device = torch.device('cpu')

        if load_path == "clip":
            import clip
            model, cliptransforms = clip.load("RN50", device=str(self.device))
            embedding = ClipEnc(model)
            embedding_dim = 1024
            self.transforms = cliptransforms

        elif (load_path == "random") or (load_path == ""):
            embedding, embedding_dim = _get_embedding(embedding_name=embedding_name, load_path=load_path)
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        elif load_path == "r3m":
            from r3m import load_r3m_reproduce
            rep = load_r3m_reproduce("r3m")

            # 关键：拆 DataParallel
            if isinstance(rep, torch.nn.DataParallel):
                rep = rep.module

            embedding = rep
            embedding_dim = rep.outdim
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ])

        else:
            raise NameError("Invalid Model")

        print("embedding type:", type(embedding))

        embedding.eval()
        embedding.to(self.device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim + self.proprio,))


    # def observation(self, observation):
    #
    #     ### INPUT SHOULD BE [0,255]
    #     if self.embedding is not None:
    #         inp = self.transforms(Image.fromarray(observation.astype(np.uint8))).reshape(-1, 3, 224, 224)
    #         if "r3m" in self.load_path:
    #             ## R3M Expects input to be 0-255, preprocess makes 0-1
    #             inp *= 255.0
    #         inp = inp.to(self.device)
    #         with torch.no_grad():
    #             emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
    #
    #         ## IF proprioception add it to end of embedding
    #         if self.proprio:
    #             try:
    #                 proprio = self.env.unwrapped.get_obs()[:self.proprio]
    #             except:
    #                 proprio = self.env.unwrapped._get_obs()[:self.proprio]
    #             emb = np.concatenate([emb, proprio])
    #
    #         return emb
    #     else:
    #         return observation

    def observation(self, observation):
        """
        observation can be:
          1) np.ndarray image: H x W x 3 uint8
          2) dict: {"image": img, "state": state_vec}
        """
        if self.embedding is None:
            return observation

        state = None
        if isinstance(observation, dict):
            img = observation["image"]
            state = observation.get("state", None)
        else:
            img = observation

        inp = self.transforms(Image.fromarray(img.astype(np.uint8))).reshape(1, 3, 224, 224)

        if "r3m" in self.load_path:
            inp *= 255.0

        inp = inp.to(self.device)

        with torch.no_grad():
            emb = self.embedding(inp).view(-1, self.embedding_dim).cpu().numpy()

        emb = emb[0]

        if self.proprio and state is not None:
            state = np.asarray(state).reshape(-1)
            proprio = state[:self.proprio]
            emb = np.concatenate([emb, proprio], axis=0)

        return emb



    def encode_batch(self, obs, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp = []
        for o in obs:
            i = self.transforms(Image.fromarray(o.astype(np.uint8))).reshape(-1, 3, 224, 224)
            if "r3m" in self.load_path:
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                i *= 255.0
            inp.append(i)
        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb =  self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None:
            return self.observation(self.env.observation(None))
        else:
            # returns the state based observations
            return self.env.unwrapped.get_obs()
          
    def start_finetuning(self):
        self.start_finetune = True


class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

    def get_image(self):
        if self.camera_name == "default":
            print("Camera not supported")
            assert(False)
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                            device_id=self.device_id)
        else:
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                              camera_name=self.camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()

class GymnasiumPixelObs(gym.ObservationWrapper):
    """
        Pixel observation wrapper for Gymnasium environments that support render_mode="rgb_array".
        Returns image observations as HxWx3 uint8 in [0,255].
    """
    def __init__(self, env, width=256, height=256, camera_name=None, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        super().__init__(env)
        self.width = int(width)
        self.height = int(height)
        self.camera_name = camera_name
        self.depth = depth
        # self.device_id = device_id
        # 原始 state space
        state_space = env.observation_space
        print("state_space type:", type(state_space))
        print("state_space:", state_space)
        # 新的 observation space: dict(image, state)
        self.observation_space = Dict({
            "image": Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            ),
            "state": state_space,
        })

    def get_image(self):
        img = self.env.render()
        if img is None:
            raise RuntimeError(
                "env.render() returned None. Make sure render_mode='rgb_array' when creating the env."
            )

        img = np.asarray(img)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        return img

    def observation(self, observation):
        #state_space type: <class 'gymnasium.spaces.dict.Dict'>
        if isinstance(observation, dict) and "observation" in observation:
            state_vec = observation["observation"]
        else:
            state_vec = observation

        return {
            "image": self.get_image(),
            "state": state_vec,
        }