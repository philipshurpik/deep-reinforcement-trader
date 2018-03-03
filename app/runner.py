import os, datetime
import numpy as np
from .env import MarketEnv
from .ddpg.DDPGAgent import DDPGAgent
from .config import Config


class Runner:
    def __init__(self, model_name, verbose=0):
        save_folder = os.path.join(os.getcwd(), "artifacts", "ddpg_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + model_name)
        os.makedirs(save_folder)
        np.random.seed(Config.seed)
        self.env = MarketEnv(csv_name=Config.data["file_name"], window_size=Config.model["window_size"], verbose=verbose, save_folder=save_folder)
        self.env.seed(Config.seed)
        self.agent = DDPGAgent(self.env, model_config=Config, save_folder=save_folder)

    def train(self):
        self.run(episodes_number=100000)

    def test(self):
        self.run(train_mode=False, episodes_number=100)

    def run(self, episodes_number, train_mode=True):
        for episode_n in range(episodes_number):
            print("**************** Episode " + str(episode_n) + " ****************")
            if train_mode:
                self.train_episode(episode_n)
            if episode_n % 10 == 0:
                print("*** Test " + str(episode_n) + " ***")
                self.test_episode()

    def train_episode(self, episode_n):
        self.agent.train_episode()
        if episode_n > 100 and episode_n % 10 == 0:
            self.agent.save_weights(episode_n)
        self.env.print_summary(epsilon=self.agent.epsilon)

    def test_episode(self):
        self.agent.test_episode()
        self.env.print_summary(epsilon=self.agent.epsilon)
