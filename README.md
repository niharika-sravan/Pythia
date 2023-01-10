# Pythia
Our agent is presented $N$ transient light curves from the Zwicky Transient Facility (ZTF), one of which is the KN and the rest are contaminants, chosen randomly from a list of supernovae and unassociated GRB afterglows. The agent observes the events on day 1 and on days 2 through 6 assigns one additional photometry with ZTF in g, r, or i using a deep 300s exposure to one of the events. The photometry or upper limit from the chosen observation is added in the next timestep. The agent gets a reward 1 if the follow-up is assigned to the KN, and 0 otherwise. 

The objective is to maximize the number of follow-up assigned to the true KN. The maximum achievable score is 5. A random agent will achieve an expected score of $5/N$. 

## AI
Pythia uses linear value function approximation to learn the optimal Q∗ in SARSA(0). It currently performs 3x better than random.

## Human players
It is very hard to know what the optimal behavior policy (Pythia uses the Q-value and humans uses their brains) is for many problems. Therefore it is essential to compare against strong benchmarks to establish the utility of the algorithm.

To play against Pythia use: `python human.py --plots --verbose --agent <yourname> --n-episodes 10`

Note: while you get a reward 1 regardless of which filter you obtain the observation in, you may want to think about how the information you get from that decision might help you in the next timestep.
