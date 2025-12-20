import time
import gymnasium as gym
from torch.optim import Adam
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple,Generator,Any
import matplotlib
from gymnasium import Env

if not os.path.exists('results'):
    os.makedirs('results')

matplotlib.use("Agg")

PLOT_PATH = "results/ppo_learning_graph"

class PPO:
    def __init__(self, policy_class, env:Env, **hyperparameters):
        self._init_hyperparameters(hyperparameters)

        self.best_mean_rew:float = -99999
        self.actor_state_dict_from_best = None
        self.critic_state_dict_from_best = None

        self.avg_rew_hist:np.ndarray = np.array([])
        self.avg_actor_losses_hist:np.ndarray = np.array([])

        self.env:Env = env
        self.continues_env:bool = isinstance(env.action_space, gym.spaces.Box)
        self.obs_dim:int = env.observation_space.shape[0]
        self.act_dim:int = env.action_space.shape[0] if self.continues_env else env.action_space.n

        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)
        self.actor_optim:torch.optim.optimizer.Optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim:torch.optim.optimizer.Optimizer = Adam(self.critic.parameters(), lr=self.lr)

        if self.continues_env:
            self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
            self.cov_mat = torch.diag(self.cov_var)

        self.logger:dict = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': []
        }

    def learn(self, total_timesteps:int = 1_000_000) -> Tuple[dict,dict]:
        print("start training PPO")
        t_so_far:int = 0
        i_so_far:int = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            V, _ ,_= self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            procent_of_finish = (t_so_far+1) / total_timesteps
            new_lr = self.lr * (1.0-procent_of_finish)
            new_lr = max(new_lr, 0.0)
            self.actor_optim.param_groups[0]["lr"] = new_lr
            self.critic_optim.param_groups[0]["lr"] = new_lr

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_prob,ent = self.evaluate(batch_obs, batch_acts)

                ratio = torch.exp(curr_log_prob - batch_log_probs)

                surr1 = A_k * ratio
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A_k
                ent = -ent.mean()

                actor_loss = (-torch.min(surr1, surr2)).mean() + self.exploration * ent
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.parameters_max_change)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.parameters_max_change)
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())

            self._log_summary()

            #if i_so_far % self.save_freq == 0:
            #    torch.save(self.actor.state_dict(), 'results/ppo_actor.pth')
            #    torch.save(self.critic.state_dict(), 'results/ppo_critic.pth')

        return self.actor.state_dict(),self.critic.state_dict()

    def rollout(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_rtgs = []

        t = 0
        ep_rews = []
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False
            for ep_t in range(self.max_timestep_per_episode):
                t += 1
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0):
                    self.env.render()
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)

                done = terminated | truncated

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        self.logger['batch_lens'] = batch_lens
        self.logger['batch_rews'] = batch_rews

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews) -> torch.Tensor:
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discount_factor = 0
            for rew in reversed(ep_rews):
                discount_factor = rew + self.gamma * discount_factor
                batch_rtgs.insert(0, discount_factor)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, obs) -> Tuple[np.ndarray, torch.Tensor]:
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat) if self.continues_env else Categorical(logits=mean)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat) if self.continues_env else Categorical(logits=mean)
        log_prob = dist.log_prob(batch_acts)
        ent = dist.entropy()
        return v, log_prob,ent

    def show_results(self) -> Generator[int,Any,Any]:
        print("start show")
        iteration_counter = 0
        while True:
            iteration_counter += 1
            done = False
            obs, _ = self.env.reset()

            epiode_reward = 0

            while not done and epiode_reward<10_000:

                action = self.actor(obs).detach().numpy() if self.continues_env else self.actor(obs).argmax().item()
                obs, rew, terminated, truncated, _ = self.env.step(action)

                epiode_reward += rew
                done = terminated #| truncated
            yield epiode_reward

    def _init_hyperparameters(self, hyperparameters) -> None:
        self.lr:float = 0.0005
        self.save_freq:int = 10
        self.max_timestep_per_episode:int = 1600
        self.timesteps_per_batch:int = 4800
        self.n_updates_per_iteration:int = 4
        self.clip:float = 0.2
        self.render:bool = False
        self.render_every_i:int = 10
        self.gamma:float = 0.95
        self.seed:int|None = None
        self.exploration:float = 0.2
        self.parameters_max_change:float = 2.5

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed is not None:
            assert (type(self.seed) == int)
            torch.manual_seed(self.seed)
            print("seed set")

    def _log_summary(self) -> None:
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_reward) for ep_reward in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        self.avg_rew_hist = np.append(self.avg_rew_hist, avg_ep_rews)
        self.avg_actor_losses_hist = np.append(self.avg_actor_losses_hist, -avg_actor_loss)

        if avg_ep_rews > self.best_mean_rew:
            self.best_mean_rew = avg_ep_rews
            torch.save(self.actor.state_dict(), 'results/ppo_actor.pth')
            torch.save(self.critic.state_dict(), 'results/ppo_critic.pth')

            avg_ep_lens = str(round(avg_ep_lens, 2))
            avg_ep_rews = str(round(avg_ep_rews, 2))
            avg_actor_loss = str(round(avg_actor_loss, 5))

            print(flush=True)
            print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
            print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
            print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            print(f"Average Loss: {avg_actor_loss}", flush=True)
            print(f"Timesteps So Far: {t_so_far}", flush=True)
            print(f"Iteration took: {delta_t} secs", flush=True)
            print("Model State Dict save")
            print(f"------------------------------------------------------", flush=True)
            print(flush=True)

        self.plot_graph(self.avg_rew_hist, self.avg_actor_losses_hist)

        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def help(self) -> None:
        print(f"self.init intailaize some parameters ")
        print(f"self.learn start learning model and save results in ppo_actor and pp_critic")
        print(f"self.rollout collect training data")
        print(f"self.compute_rtgs compute rtgs in formula f(n) = rew[n] + gamma * f(n-1)")
        print(f"self.get_action get action from actor and transform in to numpy frendly form")
        print(f"self.evaluate get value from value net and log prob from current policy")
        print(f"self.show_results start infinity loop showing game that policy play")
        print(f"self._init_hyperparameters create parameters for learning")
        print(f"self._log_summary print all information from training every iteration")
        print(f"self.plot graph plto graph at {PLOT_PATH}")

    def plot_graph(self, hist_avg_rew, hist_avg_act_loss) -> None:
        fig = plt.figure(figsize=(8,5))
        plt.subplot(121)
        plt.ylabel("avg_actor_loss")
        plt.xlabel("epoch")
        plt.plot(hist_avg_act_loss)

        plt.subplot(122)
        plt.ylabel("avg_rew")
        plt.xlabel("epoch")
        plt.grid(axis="y")
        plt.plot(hist_avg_rew, color='orange')

        plt.subplots_adjust(wspace=1.0)
        fig.savefig(PLOT_PATH)
        plt.close(fig)

    def __getattr__(self, item:str):
        raise AttributeError(
            f"Item '{item}' does not exist. If you want help, call the help method."
        )

    def load(self, path_to_actor:str|None=None, path_to_critic:str|None=None)->None:
        try:
            if path_to_actor:
                self.actor.load_state_dict(torch.load(path_to_actor))
            if path_to_critic:
                self.critic.load_state_dict(torch.load(path_to_critic))
        except FileNotFoundError :
            raise "path isn\'t exist"
#podtlenem