
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons


def plot_state(lcs, plotName, number_of_transients=None, title=None, xlim=[0,7], interact=False, plots_window_size=10):

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
    
        fig = plt.figure(figsize=(plots_window_size, plots_window_size))
    
        ncols = 3
        if number_of_transients is None:
            number_of_transients = len(positions)

        nrows = int(np.ceil(number_of_transients / ncols))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.1, hspace=0.25)
  
        for ii in range(number_of_transients):
            for jj, (group_position, lc_obj) in enumerate(lc_objs):
                if ii == group_position:
                    transient_present=True
                    break
            if ii != group_position:
                transient_present=False

            loc_x, loc_y = np.divmod(ii, nrows)
            loc_x, loc_y = int(loc_x), int(loc_y)
            ax = fig.add_subplot(gs[loc_y, loc_x])
            ax.set_title(ii, fontsize=18)

            if not transient_present: 
                ax.text(0.05, 0.45, "No Photometry Yet", fontsize=12)
                continue

            lc_passbands = lc_obj.groupby('passband')
            for group_passband, lc_passband in lc_passbands:
                lc_surveys = lc_passband.groupby('survey')
                for group_survey, lc_survey in lc_surveys:
                    mjd = lc_survey.mjd.to_numpy()
                    y, sigma_y = lc_survey.mag.to_numpy(), lc_survey.mag_err.to_numpy()
                    t = mjd - lc_survey.tc.iloc[0]
                    passband = lc_survey.passband.to_numpy()
                    color = pass_to_color[passband[0]]
                    if group_survey == "Pythia":
                        det_marker = 'x'
                        upper_marker = '1'
                    else:
                        det_marker = 'o'
                        upper_marker = 'v'
    
                    idx = np.where(np.isfinite(sigma_y))[0]
                    if len(idx) > 0:
                        ax.errorbar(
                            t[idx],
                            y[idx],
                            sigma_y[idx],
                            fmt=det_marker,
                            color=color,
                            markersize=16,
                        )
        
                    idx = np.where(~np.isfinite(sigma_y))[0]
                    if len(idx) > 0:
                        ax.plot(
                            t[idx],
                            y[idx],
                            marker=upper_marker,
                            color=color,
                            markersize=16,
                        )
    
            ax.set_xlim(xlim)
            ax.invert_yaxis()
    
        fig.text(0.45, 0.05, "Time [days]", fontsize=30)
        fig.text( 
            0.01,
            0.5,
            "Apparent Magnitude",
            va="center",
            rotation="vertical",
            fontsize=30,
        ) 
        if title is not None:
            fig.text(0.35, 0.01, title, fontsize=30)
        
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
                choice_data['position'] = int(label)

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
