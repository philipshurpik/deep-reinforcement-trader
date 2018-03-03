# Simple crypto trading using Reinforcement Learning

I used DDPG approach:
https://arxiv.org/pdf/1509.02971v5.pdf

DDPG implementation was adapted from this repo:
https://github.com/yanpanlau/DDPG-Keras-Torcs

Also while trying I tried simple Policy Gradient with deterministic actions but it not worked well

# The main problem:
It converges to 1 strange case - to make only 1 long position each run.
(or sometimes it can not converge making shorts)


# Ideas how to fix it:
* Using classical indicators from TA-lib - like moving average 
* Try different type type of activations:  
1) sigmoid - [0..1] - position size
2) softmax - [short, hold, long] for selecting action
* Try to use DDPG or PPO implementation from tensorforce/other libraries