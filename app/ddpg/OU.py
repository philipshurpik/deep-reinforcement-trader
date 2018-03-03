import numpy as np
np.random.seed(42)


class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
