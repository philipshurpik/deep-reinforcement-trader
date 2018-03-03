import numpy as np
import pandas as pd


class Stats(object):
    def __init__(self, filename, interval=100):
        self.filename = filename
        self.interval = interval
        self.rewards = []
        self.buy_and_holds = []
        self.draw_downs = []
        self.epsilons = []
        self.operations = []
        self.mean_rewards = []
        self.mean_buy_and_holds = []
        self.mean_draw_downs = []

    def add(self, reward, buy_and_hold, draw_dawn, epsilon, operations):
        self.rewards.append(reward)
        self.buy_and_holds.append(buy_and_hold)
        self.draw_downs.append(draw_dawn)
        self.epsilons.append(epsilon)
        self.operations.append(operations)
        #
        self.mean_rewards.append(np.mean(self.rewards[-self.interval:]))
        self.mean_buy_and_holds.append(np.mean(self.buy_and_holds[-self.interval:]))
        self.mean_draw_downs.append(np.mean(self.draw_downs[-self.interval:]))
        self.save()

    def save(self):
        df = pd.DataFrame(
            data=np.array([self.epsilons, self.operations, self.rewards, self.buy_and_holds, self.draw_downs,
                           self.mean_rewards, self.mean_buy_and_holds, self.mean_draw_downs]).T,
            columns=["Epsilon", "Operations", "Rewards", "Buy and hold", "Drawdowns", "Mean Rewards", "Mean Buy and Holds", "Mean Drawdowns"]
        )
        df.to_csv(self.filename, sep=";",)

    def print_latest(self):
        r, b, d = self.mean_rewards[-1], self.mean_buy_and_holds[-1], self.mean_draw_downs[-1]
        print("Stats\t| Avg. Reward: %.4f\t| Avg. Buy&Hold: %.4f| Avg. Drawdown: %.4f" % (r, b, d))
