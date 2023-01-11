
import argparse
import os
import numpy as np
import pandas as pd
import time
import random

import defs
import pythia

import plot_utils


def get_parser():

    parser = argparse.ArgumentParser(
        description="Automated agent."
    )
    parser.add_argument(
        "--agent", type=str, default="Human", help="Name of the agent"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to simulate",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.,
        help="Randomness parameter",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=False,
        help="create summary plots",
    )
    parser.add_argument( 
        "--plots-window-size",
        type=int,
        default=10,
        help="Size of plotting window",
    ) 
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print out progress",
    )
    parser.add_argument(
        "--seed",
        metavar="seed",
        type=int,
        default=42,
        help="Sampling seed (default: 42)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path to the output directory",
        default="outdir",
    )
    return parser


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    n_episodes = args.n_episodes
    epsilon = args.epsilon
    agent = args.agent
    plots_window_size = args.plots_window_size

    outputDirectory = f'{args.outdir}/{agent}'
    if not os.path.isdir(outputDirectory):
        os.makedirs(outputDirectory)

    game_data = []

    data = defs.train_data('test')

    lcs = pd.DataFrame()
    k = 0
    while k < n_episodes:
        t = time.time()
        #select episode
        KN = random.choice(data.KN_lc['sim'].unique().tolist()) #names
        contaminants = random.sample(data.contaminant_lc['sim'].unique().tolist(), defs.N - 1)
        KN_idx = random.randrange(defs.N)
        contaminant_idx = [i for i in range(defs.N) if i != KN_idx]
        random.shuffle(contaminant_idx)
        KN_lc_ = data.KN_lc[data.KN_lc['sim'] == KN]
        contaminant_lcs_ = data.contaminant_lc[data.contaminant_lc['sim'].isin(contaminants)]
        if KN_lc_['passband'].nunique() < defs.n_phot:
            continue
        if (contaminant_lcs_.groupby(['sim']).apply(lambda x: x['passband'].nunique()) < defs.n_phot).any():
            continue

        k += 1
        R_tau = 0
        KN_lc = KN_lc_.copy()
        contaminant_lcs = contaminant_lcs_.copy()
        for timestep in range(1, defs.horizon-1): #in SARSA not useful if action prime does not exist
            state = pythia.State(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep)

            if args.plots:
                plotName = f'{outputDirectory}/{k}_{timestep}.png'
                state.KN_lc['position'] = KN_idx
                state.contaminant_lcs['position'] = state.contaminant_lcs['sim'].map(dict(zip(contaminants, contaminant_idx)))
                game_start = time.time()
                choice_data = plot_utils.plot_state(pd.concat([state.KN_lc, state.contaminant_lcs]), plotName, title=f'Episode {k}: Observation {timestep}', interact=True, plots_window_size=plots_window_size, number_of_transients=defs.N)
                #convert choice_data to action vector
                action = np.zeros((defs.n_filt * defs.N, 1))
                choice_idx = 3*(choice_data['position'])+choice_data['passband']
                action[choice_idx] = 1
                reward = pythia.get_reward(action, KN_idx)
                choice_data = {**choice_data, 'reward': reward, 'timestep': timestep, 'episode': k, 'time': time.time() - game_start}
                game_data.append(choice_data)

            state_prime, KN_lc, contaminant_lcs, obs = pythia.next_state(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep+1, action)
            R_tau += reward

            if args.verbose:
                print(k, timestep, reward)

        if args.verbose:
            print(k, R_tau, time.time()-t)
        KN_lc['episode'] = k
        KN_lc['position'] = KN_idx
        contaminant_lcs['episode'] = k
        contaminant_lcs['position'] = contaminant_lcs['sim'].map(dict(zip(contaminants, contaminant_idx)))
        lcs = pd.concat([lcs, KN_lc, contaminant_lcs])

    lcs.to_csv(os.path.join(outputDirectory, 'lcs.csv'), index=False)

    df = pd.DataFrame(game_data)
    df.to_csv(os.path.join(outputDirectory, 'game.csv'), index=False)

if __name__ == "__main__":
    main()
