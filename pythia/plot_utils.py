import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

def plot_state(lcs, agent, phase, label_all, number_of_transients=None, title=None, xlim=[0,7], plots_window_size=10,
               KN_name = None, KN_loc = None, contaminant_name = None, contaminant_loc = None):

    pass_to_color = {1: 'g', 2: 'r', 3: 'k'}
    global choice_data
    choice_data = {'position': None,
                   'passband': None}

    lc_objs = lcs.groupby('position')

    fig = plt.figure(figsize=(plots_window_size*1.2, plots_window_size))

    ncols = 3
    nrows = int(np.ceil(number_of_transients / ncols))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.25, hspace=0.25, top=0.95, left=0.1)

    for ii in range(number_of_transients):
        for jj, (group_position, lc_obj) in enumerate(lc_objs):
            if ii == group_position:
                transient_present=True
                break
        if ii != group_position:
            transient_present=False

        loc_y, loc_x = np.divmod(ii, nrows)
        loc_y, loc_x = int(loc_x), int(loc_y)
        ax = fig.add_subplot(gs[loc_x, loc_y])

        title_text = '#'+str(ii)
        if (phase == 'train'):
            if ii == KN_loc:
                title_text = title_text+' '+KN_name.split('_')[0]
            elif label_all:
                title_text = title_text+' '+contaminant_name[contaminant_loc.index(ii)].split('_')[0]
        ax.set_title(title_text, fontsize=16)

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
                if group_survey == agent:
                    det_marker = 'x'
                    upper_marker = 'v'
                else:
                    det_marker = 'o'
                    upper_marker = 'v'

                idx = np.where(np.isfinite(sigma_y))[0]
                if len(idx) > 0:
                    ax.errorbar(
                        t[idx],
                        y[idx],
                        yerr=sigma_y[idx],
                        fmt=det_marker,
                        color=color,
                        markersize=8,
                    )

                idx = np.where(~np.isfinite(sigma_y))[0]
                if len(idx) > 0:
                    ax.scatter(
                        t[idx],
                        y[idx],
                        marker=upper_marker,
                        color=color,
                        s=25,
                    )

        ax.set_xlim(xlim)
        ax.invert_yaxis()

    fig.text(0.4, 0.05, "Time [days]", fontsize=30)
    fig.text(0.01, 0.5, "Apparent Magnitude", va="center", rotation="vertical", fontsize=30)
    if title is not None:
        fig.text(0.3, 0.01, title, fontsize=30)

    positions_label = [False] * number_of_transients
    ax_positions = plt.axes([0.92, 0.75, 0.2, 0.2])
    position_button = CheckButtons(ax_positions, list(range(number_of_transients)), positions_label)

    passbands = ['g', 'r', 'i']
    passbands_label = [False] * len(passbands)
    ax_passbands = plt.axes([0.92, 0.6, 0.1, 0.1])
    passband_button = CheckButtons(ax_passbands, passbands, passbands_label)

    def func_close(label):
      if sum(passband_button.get_status()) == 1 and sum(position_button.get_status()) == 1 and sum(close_button.get_status()):
        choice_data['passband'] = int(np.where(passband_button.get_status())[0])
        choice_data['position'] = int(np.where(position_button.get_status())[0])
        plt.close()

    ax_close = plt.axes([0.92, 0.45, 0.1, 0.1])
    close_button = CheckButtons(ax_close, ['Finished'], [False])
    close_button.on_clicked(func_close)

    plt.show()

    return choice_data
