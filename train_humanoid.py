import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import os
import time
import tempfile
import shutil
from sklearn.decomposition import PCA
import zipfile


# 自定义回调函数
class CustomCallback(BaseCallback):
    def __init__(self, test_env, verbose=1, test_freq=10000):
        super().__init__(verbose)
        self.test_env = test_env
        self.test_freq = test_freq
        self.start_time = time.time()

    def _on_step(self):
        if self.num_timesteps % 1000 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
            mean_reward = np.mean([info.get("reward", 0) for info in self.locals["infos"]])
            print(f"步数: {self.num_timesteps}, FPS: {fps:.2f}, 平均奖励: {mean_reward:.2f}")
            if "train" in self.model.logger.name_to_value:
                actor_loss = self.model.logger.name_to_value.get('train/actor_loss', 0)
                critic_loss = self.model.logger.name_to_value.get('train/critic_loss', 0)
                ent_coef = self.model.logger.name_to_value.get('train/ent_coef', 0)
                print(f"actor_loss: {actor_loss}")
                print(f"critic_loss: {critic_loss}")
                print(f"ent_coef: {ent_coef}")
                if actor_loss > 1e10 or critic_loss == float('inf'):
                    print("警告：损失值过大，停止训练")
                    return False
        if self.num_timesteps % self.test_freq == 0:
            self._test_model()
        return True

    def _test_model(self):
        obs = self.test_env.reset()
        total_reward = 0
        for _ in range(1000):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, dones, infos = self.test_env.step(action)
            total_reward += reward[0]
            if dones[0]:
                obs = self.test_env.reset()
        print(f"测试完成，总奖励: {total_reward:.2f}")


# 观测降维包装器
class ObservationProjectionWrapper(gym.Wrapper):
    def __init__(self, env, pca_model=None, target_dim=None):
        super().__init__(env)
        self.original_dim = env.observation_space.shape[0]
        self.target_dim = target_dim if target_dim is not None else self.original_dim
        self.pca = pca_model if pca_model else PCA(n_components=min(self.original_dim, self.target_dim))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.target_dim,), dtype=np.float64)

    def _collect_observations(self, env, num_samples):
        obs_samples = []
        obs, _ = env.reset()
        for _ in range(num_samples):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            obs_samples.append(obs)
            if terminated or truncated:
                obs, _ = env.reset()
        obs_samples = np.array(obs_samples)
        print(f"收集的观测样本形状: {obs_samples.shape}")
        return obs_samples

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._project_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._project_observation(obs), reward, terminated, truncated, info

    def _project_observation(self, obs):
        pca_obs = self.pca.transform(obs.reshape(1, -1)).flatten()
        if self.target_dim > len(pca_obs):
            padded_obs = np.zeros(self.target_dim, dtype=np.float64)
            padded_obs[:len(pca_obs)] = pca_obs
            return padded_obs
        return pca_obs[:self.target_dim]


# 奖励整形包装器
class RewardShapingWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        forward_velocity = info.get("x_velocity", 0)
        z_height = info.get("z_position", 0)
        angular_velocity = np.linalg.norm(info.get("angular_velocity", np.zeros(3)))
        reward += forward_velocity * 0.01
        reward += max(0, z_height - 0.5) * 0.02
        reward -= angular_velocity * 0.001
        info["reward"] = reward
        return obs, reward, terminated, truncated, info


# 创建训练环境（无视频录制）
def make_env(rank, pca_model=None, target_dim=None):
    def _init():
        env = gym.make("Humanoid-v5", max_episode_steps=1000)
        env = ObservationProjectionWrapper(env, pca_model=pca_model, target_dim=target_dim)
        env = RewardShapingWrapper(env)
        return env

    return _init


# 创建测试环境（带视频录制）
def make_test_env(pca_model=None, target_dim=None, video_folder=None):
    env = gym.make("Humanoid-v5", max_episode_steps=1000, render_mode="rgb_array")
    env = ObservationProjectionWrapper(env, pca_model=pca_model, target_dim=target_dim)
    env = RewardShapingWrapper(env)
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    return env


# 测试函数（带视频录制）
def test_model(model, video_folder, num_steps=1000):
    env = make_test_env(pca_model=pca_model, target_dim=target_dim, video_folder=video_folder)
    obs, _ = env.reset()
    total_reward = 0
    steps_completed = 0
    while steps_completed < num_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps_completed += 1
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    print(f"测试完成（{num_steps} 步），总奖励: {total_reward:.2f}")


# 自定义加载函数，强制加载模型权重
def custom_load_model(path, env, device, custom_objects=None):
    try:
        # 尝试标准加载
        model = SAC.load(path, env=env, device=device, custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"标准加载失败: {e}")
        print("尝试强制加载模型权重...")
        # 初始化一个新模型
        model = SAC("MlpPolicy", env, learning_rate=1e-5, ent_coef=0.2, device=device)
        # 解压 ZIP 文件到临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # 加载 policy.pth
            policy_path = os.path.join(temp_dir, "policy.pth")
            if os.path.exists(policy_path):
                loaded_params = torch.load(policy_path, map_location=device, weights_only=True)
                model.policy.load_state_dict(loaded_params, strict=False)
                print("成功加载策略网络权重")
                return model
            else:
                print("未找到 policy.pth 文件")
                return None
        except Exception as load_err:
            print(f"强制加载失败: {load_err}")
            return None
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    num_envs = min(16, os.cpu_count())
    temp_env = gym.make("Humanoid-v5", max_episode_steps=1000)
    print(f"观测空间形状: {temp_env.observation_space.shape}")
    n_features = temp_env.observation_space.shape[0]

    # 收集观测样本并拟合 PCA，目标维度设为 376 以匹配预训练模型
    wrapper = ObservationProjectionWrapper(temp_env)
    obs_samples = wrapper._collect_observations(temp_env, num_samples=10000)
    target_dim = 376  # 匹配预训练模型的观测空间维度
    pca_model = PCA(n_components=min(n_features, target_dim))
    pca_model.fit(obs_samples)
    temp_env.close()

    train_env = SubprocVecEnv([make_env(i, pca_model=pca_model, target_dim=target_dim) for i in range(num_envs)])
    test_env = SubprocVecEnv([make_env(0, pca_model=pca_model, target_dim=target_dim)])

    # 验证环境观测空间
    print("当前训练环境观测空间:", train_env.observation_space)
    print("当前训练环境动作空间:", train_env.action_space)

    pretrained_model_path = "E:\\mujoco\\sac-Humanoid-v3.zip"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 检查预训练模型内容
    if os.path.exists(pretrained_model_path):
        print(f"检查预训练模型文件: {pretrained_model_path}")
        with zipfile.ZipFile(pretrained_model_path, 'r') as zip_ref:
            print("文件内容:", zip_ref.namelist())
            try:
                temp_model = SAC.load(pretrained_model_path, print_system_info=True)
                print("预训练模型观测空间:", temp_model.observation_space)
                print("预训练模型动作空间:", temp_model.action_space)
            except Exception as e:
                print(f"无法解析模型内容: {e}")

    custom_objects = {"learning_rate": 1e-5, "ent_coef": 0.2}

    # 尝试加载预训练模型，使用自定义加载函数
    model = None
    if os.path.exists(pretrained_model_path):
        print(f"从预训练模型加载: {pretrained_model_path}")
        model = custom_load_model(pretrained_model_path, train_env, device, custom_objects)
        if model is not None:
            print("模型加载成功")
            model.save("E:\\mujoco\\sac_humanoid_v5_converted.zip")
            print("模型已转换为兼容格式: E:\\mujoco\\sac_humanoid_v5_converted.zip")
        else:
            print("所有加载尝试均失败")

    if model is None:
        print("未找到预训练模型或加载失败，从头开始训练")
        model = SAC("MlpPolicy", train_env, learning_rate=1e-5, ent_coef=0.2, device=device)

    # 训练前测试预训练模型并录制视频
    print("开始训练前测试...")
    test_model(model, video_folder="E:\\mujoco\\ceshi_videos", num_steps=1000)

    # 强制设置参数
    model.learning_rate = 1e-5
    model.ent_coef = 0.2
    model.max_grad_norm = 0.1
    model.train_freq = TrainFreq(frequency=num_envs, unit=TrainFrequencyUnit.STEP)
    model.gradient_steps = num_envs
    model.batch_size = 256

    # 解冻部分参数
    for param in model.actor.parameters():
        param.requires_grad = False
    for param in model.actor.latent_pi.parameters():
        param.requires_grad = True
    model.actor.mu.weight.requires_grad = True
    model.actor.mu.bias.requires_grad = True

    # 验证参数
    print(f"当前学习率: {model.learning_rate}")
    print(f"当前熵系数: {model.ent_coef}")
    print(f"最大梯度范数: {model.max_grad_norm}")

    # 设置回调
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="E:\\mujoco\\checkpoints",
                                             name_prefix="sac_humanoid_v5")
    callback = CallbackList([CustomCallback(test_env=test_env, test_freq=10000), checkpoint_callback])

    # 开始训练（5,000,000 步）
    try:
        model.learn(total_timesteps=5000000, reset_num_timesteps=False, callback=callback, log_interval=1,
                    tb_log_name="sac_humanoid_v5")
        model.save("E:\\mujoco\\sac_humanoid_v5_finetuned")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")

    # 训练后测试最终模型并录制视频
    print("开始训练后测试...")
    test_model(model, video_folder="E:\\mujoco\\videos", num_steps=1000)

    # 清理环境
    train_env.close()
    test_env.close()