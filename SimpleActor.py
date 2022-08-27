import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip(x):
    x = torch.max(x, torch.tensor(-1.0).to(device))
    x = torch.min(x, torch.tensor(1.0).to(device))
    return x


def genaction1(state):
    # player1的simple actor
    p1_place = state[:, 0:2]
    p2_place = state[:, 2:4]
    rel = p2_place - p1_place
    move = torch.where(rel > 0, 1.0, -1.0)

    # 指向player2方向
    angle = torch.atan2(rel[:, 1], rel[:, 0]) / math.pi
    angle = clip(angle + (np.random.rand() - 0.5) / 10).unsqueeze(1)
    action = torch.cat([move, angle], dim=1)
    return action


def genaction2(state):
    action = torch.rand((state.shape[0], 2))
    action = torch.where(action < 0.5, -1.0, 1.0)
    return action
