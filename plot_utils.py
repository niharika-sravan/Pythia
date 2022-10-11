
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons


def plot_state(lcs, plotName, xlim=[0,7], interact=False):

    pass_to_color = {1: 'g', 2: 'r', 3: 'k'}
    global choice_data
    choice_data = {'position': None,
                   'passband': None}

    if 'episode' not in lcs.columns:
        lcs['episode'] = 0

    lcs_episodes = lcs.groupby('episode')
    for episode_name, lcs_episode in lcs_episodes:

        lcs_episode.sort_values(by=['position'], inplace=True)
        lc_objs = lcs_episode.groupby('position')
        positions = list([str(x) for x in set(lcs_episode.position)])
    
        fig = plt.figure(figsize=(10, 10))
    
        ncols = 3
        nrows = int(np.ceil(len(positions) / ncols))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.1, hspace=0.25)
    
        for ii, (group_position, lc_obj) in enumerate(lc_objs):
            sim = lc_obj.sim.iloc[0] 

            loc_x, loc_y = np.divmod(ii, nrows)
            loc_x, loc_y = int(loc_x), int(loc_y)
            ax = fig.add_subplot(gs[loc_y, loc_x])
    
            lc_passbands = lc_obj.groupby('passband')
            for group_passband, lc_passband in lc_passbands:
                mjd = lc_passband.mjd.to_numpy()
                y, sigma_y = lc_passband.mag.to_numpy(), lc_passband.mag_err.to_numpy()
                t = mjd - lc_passband.tc.iloc[0]
                passband = lc_passband.passband.to_numpy()
                color = pass_to_color[passband[0]]
    
                idx = np.where(np.isfinite(sigma_y))[0]
                if len(idx) > 0:
                    ax.errorbar(
                        t[idx],
                        y[idx],
                        sigma_y[idx],
                        fmt="o",
                        color=color,
                        markersize=16,
                    )
    
                idx = np.where(~np.isfinite(sigma_y))[0]
                if len(idx) > 0:
                    ax.errorbar(
                        t[idx],
                        y[idx],
                        sigma_y[idx],
                        fmt="v",
                        color=color,
                        markersize=16,
                    )
    
                ax.set_title(group_position, fontsize=18)

                ax.set_xlim(xlim)
    
        fig.text(0.45, 0.05, "Time [days]", fontsize=30)
        fig.text( 
            0.01,
            0.5,
            "Apparent Magnitude",
            va="center",
            rotation="vertical",
            fontsize=30,
        ) 
    
        if interact:

            def func_passbands(label):
                global choice_data
                passband = passbands.index(label)
                choice_data['passband'] = passband

            passbands = ['g', 'r', 'i']
            passbands_label = [False] * len(passbands)
 
            ax_passbands = plt.axes([0.1, 0.85, 0.1, 0.1])
            passband_button = CheckButtons(ax_passbands, passbands, passbands_label)
            passband_button.on_clicked(func_passbands)

            def func_positions(label):
                global choice_data
                position = positions.index(label)
                choice_data['position'] = position

            positions_label = [False] * len(positions)

            ax_positions = plt.axes([0.8, 0.85, 0.15, 0.15])
            position_button = CheckButtons(ax_positions, positions, positions_label)
            position_button.on_clicked(func_positions)

            def func_close(label):
                plt.close()
                                                       
            positions_label = [False] * len(positions)
                                                          
            ax_close = plt.axes([0.45, 0.92, 0.1, 0.05])
            close_button = CheckButtons(ax_close, ['Finished'], [False]) 
            close_button.on_clicked(func_close)            

            plt.show()

        # plt.tight_layout()
        plt.savefig(plotName)
        plt.close()

        return choice_data
