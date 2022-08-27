import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import imageio
import math
import random
import copy


class PKBG(object):
    def __init__(self, arg):
        self.size = arg.size
        self.half_size = self.size / 2
        self.bullet_num = arg.bullet_num        # 子弹个数
        self.bullet_speed = arg.bullet_speed    # 子弹速度
        self.bullet_time = arg.bullet_time      # 子弹冷却时间
        self.player1_speed = arg.player1_speed  # 玩家1速度
        self.player2_speed = arg.player2_speed  # 玩家2速度
        self.player_size = arg.player_size      # 玩家大小
        self.ran = random.Random(arg.random_seed)
        self.spot_range = self.player_size * arg.spot_rate

        self.max_action = arg.max_action
        self.player_action_dim = [0, 3, 2]      # ?, p1, p2
        self.state_dim = 2 + 2 + 1 + self.bullet_num * 5
        # cur_state_dim : 1 + 2 + 2 + 1 + self.bullet_num * 5
        # shot_avail, p1_loc, p2_loc, player_size, bullet_(shot, pos, vel)[]

        self.on_render = False
        self.save_path = ""
        self.time_step = 0
        self.shot_avail = False
        self.bullet_count = self.bullet_num
        self.player1_loc = None
        self.player2_loc = None
        self.bullet_pos = np.zeros((self.bullet_num, 2), dtype=np.float32)
        self.bullet_vel = np.zeros((self.bullet_num, 2), dtype=np.float32)
        self.bullet_shot = np.zeros(self.bullet_num, dtype=np.float32)      # 子弹还在不在场上，为了加入状态，设为浮点型

    def reset(self, on_render, save_path):
        self.on_render = on_render
        self.save_path = save_path
        self.time_step = 0
        self.shot_avail = True
        self.player1_loc = np.array([self.ran.random() * self.size, self.ran.random() * self.size])
        self.player2_loc = np.array([self.ran.random() * self.size, self.ran.random() * self.size])

        for i in range(self.bullet_num):
            if self.bullet_exist(i):
                self.bullet_mute(i)

        if self.on_render:
            self.render()

        return self.getstate()

    def step(self, action1, action2):
        reward1 = -0.1      # 尽快打到人
        reward2 = 0.1       # 只要活下来就有奖励
        done = False

        # 计算子弹下一时刻的位置
        for i in range(self.bullet_num):
            self.bullet_pos[i] += self.bullet_vel[i]

        # 打出子弹
        angle_1 = action1[2] * math.pi
        if self.shot_avail:
            for i in range(self.bullet_num):
                if not self.bullet_exist(i):
                    self.bullet_gene(i, angle_1)
                    break

        # 计算player下一时刻位置
        self.player1_loc += action1[0:2] * self.player1_speed
        self.player2_loc += action2[0:2] * self.player2_speed

        # 判断出界
        if self.outofrange(self.player1_loc):
            reward1 -= 4
            self.player1_loc = self.fixed(self.player1_loc)
        if self.outofrange(self.player2_loc):
            reward2 -= 2
            self.player2_loc = self.fixed(self.player2_loc)

        # 将出界子弹收回
        for i in range(self.bullet_num):
            if self.bullet_exist(i):
                if self.outofrange(self.bullet_pos[i]):
                    self.bullet_mute(i)

        # 判断player2是否中弹
        for i in range(self.bullet_num):
            if self.bullet_exist(i):
                if self.inter(self.bullet_pos[i], self.player2_loc, self.player_size):
                    self.bullet_mute(i)
                    reward1 += 15
                    reward2 -= 10
                    done = True

        # 密集reward
        if not done:
            r1, r2 = self.getreward()
            reward1 += 0.5 * r1
            reward2 += 0.5 * r2

        # 更新环境状态
        self.time_step += 1
        self.shot_avail = not bool(self.time_step % self.bullet_time) & bool(self.bullet_count)
        if self.on_render:
            self.render()

        next_state = self.getstate()
        info = []
        return next_state, reward1, reward2, done, info

    def getstate(self):
        # 主要是做归一化工作
        state = np.concatenate(([float(self.shot_avail) * self.half_size],
                                self.player1_loc - self.half_size,
                                self.player2_loc - self.half_size,
                                [self.player_size]), axis=0)
        for i in range(self.bullet_num):
            state = np.concatenate((state, [self.bullet_shot[i] * self.half_size],
                                    self.bullet_pos[i] - self.half_size,
                                    self.bullet_vel[i]), axis=0)
        state = state / self.half_size
        return state

    def getreward(self):
        reward1 = 0.0
        reward2 = 0.0
        full = 0.05         # 和子弹数目，地图大小，视野范围，玩家大小有关
        k = self.spot_range - self.player_size
        a = full / (k ** 2)
        for i in range(self.bullet_num):
            if self.bullet_exist(i):
                if self.inter(self.bullet_pos[i], self.player2_loc, self.spot_range):
                    v = np.linalg.norm(self.bullet_pos[i] - self.player2_loc, 2) - self.player_size
                    r = a * ((v - k) ** 2)
                    reward1 += r
                    reward2 -= r
        return reward1, reward2

    # 生成这一时刻的图像
    def render(self):
        fig = plt.figure(figsize=(6, 6))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        circle = mpatches.Circle(self.player1_loc, self.player_size, color='r')
        axes.add_patch(circle)
        circle = mpatches.Circle(self.player2_loc, self.player_size, color='g')
        axes.add_patch(circle)
        for i in range(self.bullet_num):
            if self.bullet_exist(i):
                circle = mpatches.Circle(self.bullet_pos[i], 0.04, color='k')
                axes.add_patch(circle)
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        plt.savefig(self.save_path + "_" + str(self.time_step) + ".png")
        # plt.show()
        plt.close(fig)

    # 生成 gif
    def build_gif(self):
        if not self.on_render:
            return
        img_seq = list()
        gif_file = self.save_path + ".gif"
        for i in range(self.time_step + 1):
            img_file = self.save_path + "_" + str(i) + ".png"
            img = imageio.imread(img_file)
            img_seq.append(img)
            os.remove(img_file)
        imageio.mimsave(gif_file, img_seq, "GIF", duration=0.35)

    # 去掉子弹，为了归一化后坐标为0，设在场景中间
    def bullet_mute(self, index):
        self.bullet_pos[index] = [self.half_size, self.half_size]
        self.bullet_vel[index] = [0.0, 0.0]
        self.bullet_shot[index] = 0.0  # False
        self.bullet_count += 1

    # 生成子弹
    def bullet_gene(self, index, angle_1):
        self.bullet_pos[index] = [self.player1_loc[0], self.player1_loc[1]]
        self.bullet_vel[index] = np.array([math.cos(angle_1), math.sin(angle_1)]) * self.bullet_speed
        self.bullet_shot[index] = 1.0   # True
        self.bullet_count -= 1

    # 是否出界
    def outofrange(self, pos) -> bool:
        return pos[0] < 0 or pos[0] > self.size or pos[1] < 0 or pos[1] > self.size

    # 判断点是否在圆内
    def inter(self, pos1, pos2, dis) -> bool:
        return np.linalg.norm(pos1 - pos2, 2) <= dis

    # 判断子弹是否可用
    def bullet_exist(self, index) -> bool:
        return self.bullet_shot[index] > 0.0

    # 调整 player 位置
    def fixed(self, pos):
        pos[0] = min(max(pos[0], 0), self.size)
        pos[1] = min(max(pos[1], 0), self.size)
        return pos
