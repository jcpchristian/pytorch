import gym
import torch
import torch.nn as nn
from torch import optim
import torchvision
torchvision.disable_beta_transforms_warning()
import random
from collections import deque, namedtuple
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VARIABLES
MAX_MEMORY = 20_000  # max memory of agent
BATCH_SIZE = 128  # batch memory size of agent
GAMMA = 0.99  # future reward discount factor
EPSILON_START = 1.0  # starting probability of agent choosing random action
EPSILON_END = 0.05  # ending probability of agent choosing random action
EPSILON_DECAY = 10000  # epsilon rate of decay
TAU = 0.005  # rate at which target network trains
lr = 0.001
episodes = 500
render_mode: str = 'rgb_array'

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward'))


# CREATE ENV AND ACTION DICTIONARY
env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
action = [i for i in range(env.action_space.n)]
meanings = env.unwrapped.get_action_meanings()
actions = {action: meanings for [action, meanings] in zip(action, meanings)}


# MODEL CLASS
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4, )
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Linear(64*22*16, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=-3, end_dim=-1)  # account for frames being either 3d tensors and batched 4d tensors
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x


# AGENT CLASS
class Agent:
    def __init__(self):
        self.model = Model().to(device)
        self.steps_done = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.gamma = GAMMA
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Qtrainer(self.model, self.criterion, self.optimizer)
        self.eps_threshold = EPSILON_START

    def get_action(self, state):
        rand = random.random()
        self.eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if rand < self.eps_threshold:
            return torch.tensor(env.action_space.sample())  # return random action
        else:
            state = torch.FloatTensor(state).moveaxis(2, 0)
            with torch.no_grad():
                return torch.argmax(self.model(state.to(device)))

    def remember_transition(self, state, next_state, action, reward):
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        self.memory.append((state, next_state, action, reward))

    def get_memory_sample(self, BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)


class Qtrainer:
    def __init__(self, model, criterion, optimizer):
        self.gamma = GAMMA
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train_step(self):
        if len(agent.memory) < BATCH_SIZE:
            return

        # BATCHES
        transitions = agent.get_memory_sample(BATCH_SIZE)
        transitions_batch = Transition(*zip(*transitions))
        state_batch = torch.cat(transitions_batch.state)
        next_state_batch = torch.cat(transitions_batch.next_state)
        action_batch = torch.cat(transitions_batch.action)
        reward_batch = torch.cat(transitions_batch.reward)

        # Q AND QNEW FROM AGENT AND TRAINER RESPECTIVELY
        agent_actions = agent.model(state_batch).gather(1, action_batch)
        qtrainer_actions = Qtrainer.model(next_state_batch).max(1).values * agent.gamma + reward_batch
        qtrainer_actions = qtrainer_actions.unsqueeze(1)

        # LOSS / OPTIMIZER STEPS
        loss = agent.criterion(agent_actions, qtrainer_actions)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()


agent = Agent()
Qtrainer = Qtrainer(agent.model, agent.criterion, agent.optimizer)
state, _ = env.reset()
score = 0
all_scores = []
for episode in range(1, episodes + 1):
    done = False
    while not done:
        action = agent.get_action(state).to(device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state_tensor = torch.FloatTensor(state).moveaxis(2,0).to(device)
        state = next_state
        score += reward
        next_state = torch.FloatTensor(next_state).moveaxis(2,0).to(device)
        reward = torch.tensor([reward]).to(device)
        done = terminated or truncated
        agent.remember_transition(state_tensor, next_state, action.view(1, 1), reward)
        Qtrainer.train_step()

        if done:
            state, _ = env.reset()
            print(f'Episode: {episode} | Score: {score} | Epsilon {agent.eps_threshold}')
            all_scores.append(score)
            score = 0

    # QTRAINER SOFT UPDATE
    target_net_state_dict = Qtrainer.model.state_dict()
    agent_state_dict = agent.model.state_dict()
    for key in agent_state_dict:
        target_net_state_dict[key] = agent_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    Qtrainer.model.load_state_dict(target_net_state_dict)

torch.save(agent.model, 'agent.pt')
print(f'Finished with best score{max(all_scores)}')
