import argparse


class Options:
    def __init__(self):
        self.initialized = False
        self.parser = None
        self.opt = None

    def initialize(self, parser):
        # environment
        parser.add_argument("--bullet_num", default=8, type=int, help="Number of bullets")
        parser.add_argument("--bullet_time", default=4, type=int, help="Cooling time")
        parser.add_argument("--bullet_speed", default=0.45, type=float, help="Speed of bullets")
        parser.add_argument("--player1_speed", default=0.15, type=float, help="Speed of player1")
        parser.add_argument("--player2_speed", default=0.25, type=float, help="Speed of player2")
        parser.add_argument("--spot_rate", default=4, type=float, help="Range of perception")
        parser.add_argument("--player_size", default=0.5, type=float, help="Size of players")
        parser.add_argument("--size", default=10.0, type=float, help="Size of the battlefield")
        parser.add_argument("--random_seed", default=19260817, type=int, help="Random seed")

        # train: Epoch * Games
        parser.add_argument("--train_mode", default=2, type=int, help="Testing, Training, Semi-training")
        parser.add_argument("--render_stride", default=5, type=int, help="Stride of rendering")
        parser.add_argument("--bisimple_episode", default=250, type=int, help="Both simple -- epoch")
        parser.add_argument("--bisimple_game_episode", default=20, type=int, help="Both simple -- games")
        parser.add_argument("--p1simple_episode", default=250, type=int, help="P1 simple -- epoch")
        parser.add_argument("--p1simple_game_episode", default=20, type=int, help="P1 simple -- games")
        parser.add_argument("--train_episode", default=500, type=int, help="Training -- epoch")
        parser.add_argument("--game_episode", default=20, type=int, help="Training -- games")
        parser.add_argument("--max_timesteps", default=100, type=int, help="Max steps in one game")
        parser.add_argument("--batch_size", default=256, type=int, help="Batch size")

        # model
        parser.add_argument("--max_action", default=1, type=int, help="Output clipping")
        parser.add_argument("--actor_dec_rate", default=0.95, type=float, help="Declining rate")
        parser.add_argument("--actor_dec_step", default=500, type=int, help="Declining step")
        parser.add_argument("--rand_rate", default=0.5, type=float, help="Initial random rate for exploring")
        parser.add_argument("--eval_freq", default=5e3, type=int, help="Evaluation frequency")
        parser.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
        parser.add_argument("--discount", default=0.99, type=float, help="Discount factor")
        parser.add_argument("--tau", default=0.005, type=float, help="Target network update rate")
        parser.add_argument("--policy_noise", default=0.2, type=float, help="Noise of target policy")
        parser.add_argument("--noise_clip", default=0.5, type=float, help="Range to clip target policy noise")
        parser.add_argument("--high_freq", default=2, type=int, help="High frequency of delayed policy updates")
        parser.add_argument("--low_freq", default=10, type=int, help="Low frequency of delayed policy updates")
        parser.add_argument("--stuck_freq", default=500, type=int, help="Stuck frequency of delayed policy updates")
        parser.add_argument("--winning_rate", default=0.9, type=float, help="Winning rate")
        parser.add_argument("--gpu_ids", default="3", type=str, help="CUDA device")

        # file
        parser.add_argument("--save_model", default=True, type=bool, help="Whether to save model")
        parser.add_argument("--model1_file", default="pressed1", type=str, help="Policy1 model")
        parser.add_argument("--model2_file", default="pressed2", type=str, help="Policy2 model")

        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            opt, _ = parser.parse_known_args()
            self.parser = parser
            self.opt = opt

        self.print_options(self.opt)

        return self.opt

