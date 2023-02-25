# KN follow-up toy problem
Our agent is presented 9 transient forced photometry light curves from the Zwicky Transient Facility (ZTF) [public+private i-band survey and ToO during the first 1-2 days using 180/300s exposures for GW localizations >/<1000 sq deg], one of which is the KN and the rest are contaminants, chosen randomly from a list of supernovae and unassociated GRB afterglows. The agent observes the events on day 1 and on days 2 through 7 assigns one additional photometry with ZTF in g, r, or i using a deep 300s exposure to one of the events. The resulting forced photometry or upper limit is added in the next timestep. The agent gets a reward 1 if the follow-up is assigned to the KN, and 0 otherwise. 

The objective is to maximize the number of follow-up assigned to the true KN. The maximum achievable score is 6. A random agent will achieve an expected score of 6/9 = 0.67

### Pythia
Pythia uses linear value function approximation to learn the optimal Q∗ in SARSA(0). It currently performs 3x better than random.

### Human players
It is very hard to know what the optimal behavior policy (Pythia uses the Q-value and humans uses their brains) is for many problems. Therefore it is essential to compare against strong benchmarks to establish the utility of an approach. For our problem astronomers will provide the benchmark.

To play against Pythia use: `python human.py --plots --verbose --agent <yourname> --n-episodes 10`

You will see 9 forced photometry light curves in real-time and you will choose for which event and with filter do you want to obtain an additonal 300s exposure. You will not know which the true KN is. There will be 5 timestep per episode (one collection of 9 light curves per GW trigger scenario) and --n-episodes number of games to play.

Note: while you get a reward 1 regardless of which filter you obtain the observation in, you may want to think about how the information you get from that decision might help you in the next timestep.

## Installation and Data files

Environment: coming soon...

Download [this](https://1drv.ms/u/s!At8xIP1B4oiJi-cSo5g2jJbBE_Bi5A?e=mBOVNp) folder and place it in the directory that you have checked out. Keep the name data.
Password: `proti_Pythia`
