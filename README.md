### Simple crypto trading using Reinforcement Learning

I used DDPG approach:
https://arxiv.org/pdf/1509.02971v5.pdf

DDPG implementation was adapted from this repo:
https://github.com/yanpanlau/DDPG-Keras-Torcs
+ Some ideas and loading datasets from PGPortfolio project

Also while trying I tried simple Policy Gradient with deterministic actions but it not worked well

### The main problem:
Currently it converges to 1 strange case - to make only long positions each test run - so actually converges to something like buy and hold strategy  
[[https://github.com/philipshurpik/repository/blob/master/stats/test_stats_graph.png|alt=test_stats_graph]]


### Ideas how to fix it:
* Using classical indicators from TA-lib - like moving average:  
maybe it will help our neural network to find different patterns 
* Try different type of activation - combine both:    
  sigmoid - [0..1] - position size  
  softmax - [short, hold, long] for selecting action  
* Try to use DDPG or PPO implementation from tensorforce/other libraries