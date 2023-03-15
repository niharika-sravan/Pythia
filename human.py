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
        description="KN follow-up agent."
    )
    parser.add_argument(
        "--train",
        help="The true event will be highlighted.",
        action="store_true"
    )
    parser.add_argument(
        "--label-all",
        help="All events will be labeled.",
        action="store_true"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="Human",
        help="Name of the agent"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to simulate",
    )
    parser.add_argument(
        "--plots-window-size",
        type=int,
        default=10,
        help="Size of plotting window",
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

    train = args.train
    label_all = args.label_all
    n_episodes = args.n_episodes
    agent = args.agent
    plots_window_size = args.plots_window_size

    outputDirectory = f'{args.outdir}/{agent}'
    if not os.path.isdir(outputDirectory):
        os.makedirs(outputDirectory)

    if train: phase = 'train'
    else: phase = 'test'
    data = defs.train_data(phase)

    if os.path.isfile(os.path.join(outputDirectory, 'lcs_'+phase+'.csv')):
      lcs = pd.read_csv(os.path.join(outputDirectory, 'lcs_'+phase+'.csv'))
      behav = pd.read_csv(os.path.join(outputDirectory, 'behav_'+phase+'.csv'))
      k = behav['episode'].max()
    else:
      lcs = pd.DataFrame()
      behav = pd.DataFrame()
      k = 0
    while (k < n_episodes or train):
        #select episode
        KN = random.choice(data.KN_lc['sim'].unique().tolist()) #names
        contaminants = random.sample(data.contaminant_lc['sim'].unique().tolist(), defs.N - 1)
        KN_idx = random.randrange(defs.N)
        contaminant_idx = [i for i in range(defs.N) if i != KN_idx]
        random.shuffle(contaminant_idx)
        KN_lc_ = data.KN_lc[data.KN_lc['sim'] == KN]
        contaminant_lcs_ = data.contaminant_lc[data.contaminant_lc['sim'].isin(contaminants)]
        if KN_lc_['passband'].nunique() < defs.n_phot: continue
        if (contaminant_lcs_.groupby(['sim']).apply(lambda x: x['passband'].nunique()) < defs.n_phot).any(): continue

        k += 1
        R_tau = 0
        KN_lc = KN_lc_.copy()
        contaminant_lcs = contaminant_lcs_.copy()
        for timestep in range(1, defs.horizon): #in SARSA not useful if action prime does not exist
            play_start = time.time()
            state = pythia.State(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep)
            state.KN_lc['position'] = KN_idx
            state.contaminant_lcs['position'] = state.contaminant_lcs['sim'].map(dict(zip(contaminants, contaminant_idx)))
            choice_data = plot_utils.plot_state(pd.concat([state.KN_lc, state.contaminant_lcs]), agent, phase, label_all,
                                                title=f'Episode {k}: Timestep {timestep}',
                                                plots_window_size=plots_window_size,
                                                number_of_transients=defs.N, KN_name = KN, KN_loc = KN_idx,
                                                contaminant_name = contaminants, contaminant_loc = contaminant_idx)
            #convert choice_data to action vector
            action = np.zeros((defs.n_filt * defs.N, 1))
            choice_idx = 3*(choice_data['position'])+choice_data['passband']
            action[choice_idx] = 1 #one hot encoding

            reward = pythia.get_reward(action, KN_idx)
            R_tau += reward
            state_prime, KN_lc, contaminant_lcs, obs = pythia.next_state(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep+1, action, agent=agent)
            info = [k, timestep, obs['sim'].item(), obs['passband'].item(), time.time() - play_start]
            behav = pd.concat([behav, pd.DataFrame([info], columns = ['episode', 'timestep', 'event_chosen', 'obs_passband', 't_decision'])])

        KN_lc['episode'] = k
        KN_lc['position'] = KN_idx
        contaminant_lcs['episode'] = k
        contaminant_lcs['position'] = contaminant_lcs['sim'].map(dict(zip(contaminants, contaminant_idx)))
        lcs = pd.concat([lcs, KN_lc, contaminant_lcs])
        lcs.to_csv(os.path.join(outputDirectory, 'lcs_'+phase+'.csv'), index=False)
        behav.to_csv(os.path.join(outputDirectory, 'behav_'+phase+'.csv'), index=False)

    dist_dict = dict(zip(data.KN_lc['sim'], data.KN_lc['luminosity_distance']))
    kws = behav['event_chosen'].str.split('_')
    behav['type'] = kws.str[0]#.map(typ_dict)
    lcs['luminosity_distance'] = lcs['sim'].map(dist_dict)
    eps_faint = lcs[lcs['luminosity_distance'] > 150.]['episode'].unique()
    behav_faint = behav[behav['episode'].isin(eps_faint)]
    print('Thank you for playing! Here are your scores')
    print('Agent score score_faint frac frac_faint')
    print(agent,
        round(behav[behav['type'].isin(['NSBH','BNS'])].shape[0]/behav.shape[0]*(defs.horizon-1), 2),
        round(behav_faint[behav_faint['type'].isin(['NSBH','BNS'])].shape[0]/behav_faint.shape[0]*(defs.horizon-1), 2),
        round(behav[behav['type'].isin(['NSBH','BNS'])]['episode'].nunique()/behav['episode'].max(), 2),
        round(behav_faint[behav_faint['type'].isin(['NSBH','BNS'])]['episode'].nunique()/len(eps_faint), 2))

if __name__ == "__main__":
    main()
