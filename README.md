# KN follow-up toy problem
Our agent is presented 9 transient light curves from the Zwicky Transient Facility (ZTF), one of which is the KN and the rest are contaminants, chosen randomly from a list of supernovae and unassociated GRB afterglows. The agent observes the events on day 1 and on days 2 through 6 assigns one additional photometry with ZTF in g, r, or i using a deep 300s exposure to one of the events. The resulting forced photometry or upper limit is added in the next timestep. The agent gets a reward 1 if the follow-up is assigned to the KN, and 0 otherwise. 

The objective is to maximize the number of follow-up assigned to the true KN. The maximum achievable score is 5. A random agent will achieve an expected score of 5/9 = 0.55

## Pythia
Pythia uses linear value function approximation to learn the optimal Q$^∗$ in SARSA(0). It currently performs 3x better than random.

## Human players
It is very hard to know what the optimal behavior policy (Pythia uses the Q-value and humans uses their brains) is for many problems. Therefore it is essential to compare against strong benchmarks to establish the utility of an approach. For our problem astronomers will provide the benchmark.

To play against Pythia use: `python human.py --plots --verbose --agent <yourname> --n-episodes 10`

You will see 9 forced photometry light curves in real-time and you will choose for which event and with filter do you want to obtain an additonal 300s exposure. You will not know which the true KN is.

Note: while you get a reward 1 regardless of which filter you obtain the observation in, you may want to think about how the information you get from that decision might help you in the next timestep.
