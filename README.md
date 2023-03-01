# KN follow-up toy problem
Our agent is presented 9 transient forced photometry light curves from the Zwicky Transient Facility (ZTF) [public+private i-band survey and ToO during the first 1-2 days using 180/300s exposures for GW localizations >/<1000 sq deg], one of which is the KN and the rest are contaminants, chosen randomly from a list of supernovae and unassociated GRB afterglows. The agent observes the events on day 1 and on days 2 through 7 assigns one additional photometry with ZTF in g, r, or i using a deep 300s exposure to one of the events. The resulting forced photometry or upper limit is added in the next timestep. The agent gets a reward 1 if the follow-up is assigned to the KN, and 0 otherwise. 

The objective is to maximize the number of follow-up assigned to the true KN. The maximum achievable score is 6. A random agent will achieve an expected score of 6/9 = 0.67

### Pythia
Pythia uses linear value function approximation to learn the optimal Q∗ in SARSA(0). It currently performs 3x better than random.

### Human players
It is very hard to know what the optimal behavior policy (Pythia uses the Q-value and humans uses their brains) is for many problems. Therefore it is essential to compare against strong benchmarks to establish the utility of an approach. For our problem astronomers will provide the benchmark.

We will be playing several of the following games.

There are 9 forced photometry light curves from ZTF, one of which is a KN and the rest are a selection of GRB afterglows and supernovae of vanilla types. We start a game/episode (i.e. a collection of 9 light curves per GW trigger scenario) on day 2 after trigger: day 1's data has arrived and you need to choose for which event and with which filter (g, r, or i) you want to obtain a single additional 300s exposure at a random time during the next 24 hours. You will repeat the process on day 3, when data collected during day 2 by ZTF and the follow-up observation decided by you (marked as X) will be available. There are 6 timesteps in an episode, after which a new episode will begin with a new set of 9 light curves. Your goal is to maximize the number of follow-up allocated to the KN.

There will be two phases of this exercise.

First is training, where you will learn to play the game. The true KN will be highlighted. You can play as many episodes as you like to learn to make decisions that maximize your score. Your actions and play time will be recorded.

Next, you will be tested. In this phase, you will not know which the true KN is. You will play 100 episodes, for a total of 600 decisions (I know, sorry :disappointed: We need a statistically large sample). Once again, your responses and decision times will be recorded.

Note 1: you can only select 1 filter and 1 event at any timestep. If you check Finished with none or more than one option selected in each, nothing will happen. Fix the issue, uncheck Finished, and check it again to proceed.

Note 2: while you get a reward 1 regardless of which filter you obtain the observation in, you may want to think about how the information you get from that decision might help you in the next timestep.

Note 3: if, for some reason you want to pause training or testing, just close the plotting window. Your progress is after each episode. Next time you resume, you will start where you left off from. Please try to refrain from switching between training and testing steps. Please only start testing if you are confident with your training.

Note 4: Use flag --plots-window-size 8 to make the window smaller or larger. This may help with the funky window sizes depending on your system setup.

Note 5: please report bugs!

To train:
`python human.py --train --agent <your initials>`

Just close the plot window when you are done and want to take the test.

To test:
`python human.py --agent <your initials>`

Please share the files in your folder outdir/<your initials> with Ari along with a few sentences on what your strategy was. 

## Installation and Data files

Environment: coming soon...

Download [this](https://1drv.ms/u/s!At8xIP1B4oiJi-cSo5g2jJbBE_Bi5A?e=mBOVNp) folder and place it in the directory that you have checked out. Keep the name data.
Password: `proti_Pythia`
