import numpy as np
from options import Options
import torch
import os

import utils
import TD3
import Env
import SimpleActor


def game_info(t1, t2, t3, r1, r2, dn):
    print(f"Episode Num: {t1}-{t2} Total step: {t3 + 1} P1 reward: {r1:.3f} P2 reward: {r2:.3f} Captured:{dn}")


# 将数据装进 replay buffer
def fetch_pack(pack, args, replay1, replay2, p1_avl, p2_avl):
    l = len(pack)
    for i in range(l):
        state = pack[i][0]
        shot_avl = state[0] > 0.001
        state = state[1:]
        action1 = pack[i][1]
        action2 = pack[i][2]
        r1 = np.zeros(args.bullet_time)
        r2 = np.zeros(args.bullet_time)
        next_state = None
        done_bool = None
        for j in range(args.bullet_time):
            if i + j < l:
                r1[j] = pack[i + j][3]
                r2[j] = pack[i + j][4]
                next_state = pack[i + j][5][1:]
                done_bool = pack[i + j][6]
        if p1_avl and shot_avl:
            replay1.add(state, action1, r1, next_state, done_bool)
        if p2_avl:
            replay2.add(state, action2, r2, next_state, done_bool)


def test(name, env, policy1, policy2, pic_root):
    p1_mode = False
    p2_mode = False
    p1_mode, policy1.imitation_mode = policy1.imitation_mode, p1_mode
    p2_mode, policy2.imitation_mode = policy2.imitation_mode, p2_mode
    state = env.reset(True, pic_root + "/" + name)
    for time_step in range(args.max_timesteps):
        action1 = policy1.select_action(state[1:], test=True)
        action2 = policy2.select_action(state[1:], test=True)
        next_state, reward1, reward2, done, _ = env.step(action1, action2)
        if done:
            break
        state = next_state
    env.build_gif()  # visualize
    p1_mode, policy1.imitation_mode = policy1.imitation_mode, p1_mode
    p2_mode, policy2.imitation_mode = policy2.imitation_mode, p2_mode


# 利用simple actor pretrain一下两个网络
def both_simple(policy1, replay1, replay2, env, args, pic_root, save_root):
    print("Begin both simple actor pretraining...")

    stride = args.render_stride
    t_tot = args.bisimple_episode
    tt_tot = args.bisimple_game_episode
    policy1.switch_imitation()
    policy2.switch_imitation()
    policy1.switch_high_freq()
    policy2.switch_high_freq()

    for t in range(t_tot):
        for tt in range(tt_tot):
            state = env.reset(t % stride == 0 and tt >= tt_tot - 1, pic_root + f"/Both simple {t}_{tt}")
            pack = list()

            for time_step in range(args.max_timesteps):
                action1 = policy1.select_action(state[1:])
                action2 = policy2.select_action(state[1:])
                next_state, reward1, reward2, done, _ = env.step(action1, action2)
                done_bool = float(done)
                pack.append((state, action1, action2, reward1, reward2, next_state, done_bool))
                if t > 0:
                    policy1.train(replay1, args.batch_size)
                if done:
                    break
                state = next_state

            env.build_gif()  # visualize
            fetch_pack(pack, args, replay1, replay2, True, True)
            print(f"Case {t}-{tt} is done!")

            if (t + 1) % stride == 0 and tt + 1 == tt_tot:
                test(f"test Both {t}_{tt}", env, policy1, policy2, pic_root)

    if args.save_model:
        policy1.save(save_root + "/imitation1")

    print("Both simple actor pretraining is done!")
    print("")


# p1 模仿学习，p2 强化学习
def p1_simple(policy1, policy2, replay1, replay2, env, args, pic_root, save_root):
    print("Begin p1 simple actor pretraining...")

    stride = args.render_stride
    t_tot = args.p1simple_episode
    tt_tot = args.p1simple_game_episode
    policy1.switch_imitation()
    policy2.switch_normal()
    policy1.switch_high_freq()
    policy2.switch_low_freq()

    for t in range(t_tot):
        for tt in range(tt_tot):
            state = env.reset(t % stride == 0 and tt >= tt_tot - 1, pic_root + f"/p1 simple {t}_{tt}")
            pack = list()

            p1_reward = 0
            p2_reward = 0
            for time_step in range(args.max_timesteps):
                action1 = policy1.select_action(state[1:])
                action2 = policy2.select_action(state[1:])
                next_state, reward1, reward2, done, _ = env.step(action1, action2)
                done_bool = float(done)
                pack.append((state, action1, action2, reward1, reward2, next_state, done_bool))
                if time_step % 4 == 0:
                    policy1.train(replay1, args.batch_size)
                policy2.train(replay2, args.batch_size)
                p1_reward += reward1
                p2_reward += reward2
                if done or time_step == args.max_timesteps - 1:
                    game_info(t, tt, time_step, p1_reward, p2_reward, done)
                    break
                state = next_state

            env.build_gif()  # visualize
            fetch_pack(pack, args, replay1, replay2, True, True)
            print(f"Case {t}-{tt} is done!")

            if (t + 1) % stride == 0 and tt + 1 == tt_tot:
                test(f"test p1 simple {t}_{tt}", env, policy1, policy2, pic_root)

        if args.save_model and (t + 1) % 10 == 0:
            policy1.save(save_root + f"/pressed1_iter{t + 1}")
            policy2.save(save_root + f"/pressed2_iter{t + 1}")

    print("P1 simple actor pretraining is done!")
    print("")


# 正式开始训练
def train(policy1, policy2, replay1, replay2, env, args, pic_root, save_root):
    # train
    print("Begin training...")

    stride = args.render_stride
    t_tot = args.train_episode
    tt_tot = args.game_episode
    policy1.switch_normal()
    policy2.switch_normal()
    if args.train_mode == 1:
        policy1.switch_high_freq()
        policy2.switch_high_freq()

    for t in range(t_tot):
        if args.train_mode == 2:
            if t <= 55:
                policy1.switch_stuck_freq()
                policy2.switch_stuck_freq()
            elif t <= 155:
                policy1.switch_low_freq()
                policy2.switch_low_freq()
            else:
                policy1.switch_high_freq()
                policy2.switch_high_freq()
        player_id = t // 10 % 2 + 1
        on_train = args.train_mode == 1 or (args.train_mode == 2 and t > 9)
        for tt in range(tt_tot):
            state = env.reset(t % stride == 0 and tt >= tt_tot - 1, pic_root + f"/train {t}_{tt}")
            pack = list()

            p1_reward = 0
            p2_reward = 0
            for time_step in range(args.max_timesteps):
                action1 = policy1.select_action(state[1:])
                action2 = policy2.select_action(state[1:])
                next_state, reward1, reward2, done, _ = env.step(action1, action2)
                done_bool = float(done)
                pack.append((state, action1, action2, reward1, reward2, next_state, done_bool))
                if on_train:
                    if player_id == 1:
                        policy1.train(replay1, args.batch_size)
                    else:
                        policy2.train(replay2, args.batch_size)
                p1_reward += reward1
                p2_reward += reward2
                if done or time_step == args.max_timesteps - 1:
                    game_info(t, tt, time_step, p1_reward, p2_reward, done)
                    break
                state = next_state

            env.build_gif()  # visualize
            fetch_pack(pack, args, replay1, replay2, player_id == 1, player_id == 2)
            print(f"Case {t}-{tt} is done!")

        if args.save_model and (t + 1) % 10 == 0:
            policy1.save(save_root + f"/player1_iter{t + 1}")
            policy2.save(save_root + f"/player2_iter{t + 1}")

    print("Training is done!")


if __name__ == "__main__":
    args = Options().parse()

    # root = "obs://pkgb-rl/pkgb-rl/code"
    root = "."
    load_root = root + "/load"
    save_root = root + "/save"
    pic_root = root + "/pic"
    test_root = root + "/test"

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_ids))

    if args.save_model and not os.path.exists(save_root):
        os.makedirs(save_root)

    if not os.path.exists(pic_root):
        os.makedirs(pic_root)

    if not os.path.exists(test_root):
        os.makedirs(test_root)

    env = Env.PKBG(args)

    policy1 = TD3.TD3(env, args, 1, SimpleActor.genaction1)
    policy2 = TD3.TD3(env, args, 2, SimpleActor.genaction2)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # train_mode=0 时表示 test
    # train_mode=1 时表示从头开始 train
    # train_mode=2 时表示从 imitation 之后开始 train
    if args.train_mode:
        replay1 = utils.ReplayBuffer(env.state_dim, env.player_action_dim[1], env.bullet_time, 10000)
        replay2 = utils.ReplayBuffer(env.state_dim, env.player_action_dim[2], env.bullet_time, 10000)

        if args.model1_file:
            policy_file = load_root + "/" + args.model1_file
            if args.train_mode == 1:
                policy1.load(policy_file)
            else:
                policy1.load_actor(policy_file)
        else:
            both_simple(policy1, replay1, replay2, env, args, pic_root, save_root)

        if args.model2_file:
            policy_file = load_root + "/" + args.model2_file
            if args.train_mode == 1:
                policy2.load(policy_file)
            else:
                policy2.load_actor(policy_file)
        else:
            p1_simple(policy1, policy2, replay1, replay2, env, args, pic_root, save_root)

        train(policy1, policy2, replay1, replay2, env, args, pic_root, save_root)
    else:
        if args.model1_file:
            policy_file = args.model1_file
            policy1.load(load_root + "/" + policy_file)

        if args.model2_file:
            policy_file = args.model2_file
            policy2.load(load_root + "/" + policy_file)

        for i in range(100):
            test("test_" + str(i + 1), env, policy1, policy2, test_root)
            print("test " + str(i + 1) + " done!")
