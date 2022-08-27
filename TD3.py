import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, env, player_id):
		super(Actor, self).__init__()
		self.max_action = env.max_action
		self.bullet_num = env.bullet_num
		self.state_dim = env.state_dim
		self.output_dim = env.player_action_dim[player_id]

		self.l1 = nn.Linear(self.state_dim, 80)
		self.l2 = nn.Linear(80, 128)
		self.l3 = nn.Linear(128, 128)
		self.l4 = nn.Linear(128, self.output_dim)

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		a = self.max_action * torch.tanh(self.l4(a))

		return a


class Critic(nn.Module):
	def __init__(self, env, player_id):
		super(Critic, self).__init__()
		self.state_dim = env.state_dim
		self.action_dim = env.player_action_dim[player_id]
		self.max_action = env.max_action
		self.bullet_num = env.bullet_num

		# Q1 architecture
		self.l1 = nn.Linear(self.state_dim + self.action_dim, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, 1)

		# Q2 architecture
		self.l4 = nn.Linear(self.state_dim + self.action_dim, 128)
		self.l5 = nn.Linear(128, 128)
		self.l6 = nn.Linear(128, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# 这里是魔改了 TD3，把 simple actor 弄了进去
class TD3(object):
	def __init__(self, env, args, player_id, simple_actor):
		self.player_id = player_id

		self.actor = Actor(env, player_id).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(env, player_id).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.imitation_mode = False
		self.simple_actor = simple_actor

		self.max_action = args.max_action
		self.discount = args.discount
		self.n_step = args.bullet_time
		self.tau = args.tau
		self.policy_noise = args.policy_noise
		self.noise_clip = args.noise_clip
		self.rand_rate = args.rand_rate

		self.high_freq = args.high_freq
		self.low_freq = args.low_freq
		self.stuck_freq = args.stuck_freq
		self.dec_rate = args.actor_dec_rate
		self.dec_step = args.actor_dec_step

		self.policy_freq = args.high_freq
		self.time_step = 0
		self.total_it = 0
		self.actor_loss = None
		self.critic_loss = None

	def select_action(self, state, test=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		if self.imitation_mode:
			action = self.simple_actor(state).cpu().data.numpy().flatten()
		else:
			action = self.actor(state).cpu().data.numpy().flatten()
			if not test:
				if self.player_id == 1 and np.random.rand() < self.rand_rate:
					action[2] = np.random.rand() * 2 - 1
				self.time_step += 1
				if self.time_step % self.dec_step == 0:
					self.rand_rate *= self.dec_rate

		return action

	def train(self, replay_buffer, batch_size=128):
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

			if self.imitation_mode:
				next_action = self.simple_actor(next_state)
			else:
				next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = not_done * target_Q


			for i in range(self.n_step):
				target_Q = self.discount * target_Q + reward[:, self.n_step - i - 1].unsqueeze(1)


		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		self.critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		self.critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			if self.imitation_mode:
				self.actor_loss = F.mse_loss(self.actor(state), self.simple_actor(state))
			else:
				self.actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			self.actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			self.actor_loss = 0

	def switch_stuck_freq(self):
		self.policy_freq = self.stuck_freq

	def switch_low_freq(self):
		self.policy_freq = self.low_freq

	def switch_high_freq(self):
		self.policy_freq = self.high_freq

	def switch_imitation(self):
		self.imitation_mode = True

	def switch_normal(self):
		self.imitation_mode = False

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		print("model " + filename + " is saved!")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		print("model " + filename + " is loaded!")

	def save_actor(self, filename):
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		print("model actor " + filename + " is saved!")

	def load_actor(self, filename):
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		print("model actor" + filename + " is loaded!")
