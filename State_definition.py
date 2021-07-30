import os
import numpy as np
import pandas as pd
from scipy import signal
import mat73
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from Get_Tau_Transient import get_tau
from sklearn.cluster import MeanShift
from einops import rearrange
from sklearn.preprocessing import StandardScaler
import math
import umap
import numpy.matlib
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
import stimulus_analysis


def construct_state_variables(phasic_event_data, T, type):
    n_ch = len(phasic_event_data.channel.unique())
    n_tr = len(phasic_event_data.trial.unique())
    n_band = len(phasic_event_data.band.unique())
    state_vars = np.zeros([n_ch, n_tr, n_band, T])
    std_len = {'theta': 1./3., 'beta': 0.4/3, 'gamma1': 0.3/3, 'gamma2': 0.2/3}
    channels = phasic_event_data.channel.unique().astype(int)
    for channel in channels:
        events_channel = phasic_event_data[phasic_event_data.channel == channel]
        trials = events_channel.trial.unique().astype(int)
        for trial in trials:
            events_trial = events_channel[events_channel.trial == trial]
            bands = events_trial.band.unique()
            for band_no, band in enumerate(bands):
                times = events_trial[events_trial.band == band]['sample number'].values
                times = times.astype(int)
                spikes = np.zeros(T)
                filt = signal.gaussian(4000, std= std_len[band] * 625) # size should be frequency dependent
                if type == 'rate':
                    spikes[times] = 1
                    rate = np.convolve(spikes, filt, mode='same')
                    state_vars[channel, trial, band_no, :] = rate
                elif type == 'power':
                    pow_vals = events_trial[events_trial.band == band]['power'].values
                    spikes[times] = 10*np.log10(pow_vals)
                    rate = np.convolve(spikes, filt, mode='same')
                    state_vars[channel, trial, band_no, :] = rate
                elif type == 'bit-based':
                    len_vals = events_trial[events_trial.band == band]['length'].values
                    for t, l in zip(times, len_vals):
                        spikes[t-int(l/2):t+int(l/2)] = 1
                    state_vars[channel, trial, band_no, :] = spikes

    return state_vars


def state_estimation_clustering(reduced_state_mat,n_tr,T):
    # subsample channel info
    # cluster
    mshft = MeanShift(bin_seeding=True, min_bin_freq=20).fit(reduced_state_mat) # BW too small -> too many clusters
    state_estimates = mshft.predict(reduced_state_mat).reshape(n_tr, T)
    return state_estimates


def state_estimation_bitbased(state_mat):
    # must fix - one state across all channels
    [n_ch, n_tr, n_bands, T] = state_mat.shape
    state_estimates = np.zeros([n_ch, n_tr, T])
    for channel in range(n_ch):
        for trial in range(n_tr):
            for time in range(T):
                state_estimates[channel, trial, time] = int(str(np.squeeze(state_mat[channel, trial, :, time])),2)

    return state_estimates


def get_velocity_and_pupil_area(session_id, stim, probe, T_in_s):
    data_directory = "D:/ecephys__project_cache/"
    directory = os.path.join(data_directory + '/session_' + str(session_id))
    matlab_directory = os.path.join(directory + '/MATLAB_files')
    probe_directory = os.path.join(matlab_directory + '/' + probe)
    stim_table = pd.read_csv(probe_directory + '/' + stim + '_meta.csv', index_col=0)
    session = EcephysSession.from_nwb_path(os.path.join(directory, 'session_' + str(session_id) + '.nwb'), api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })
    presentation_ids = stim_table.index.values
    pupil_data = session.get_screen_gaze_data()

    speeds = np.zeros([len(presentation_ids), int(T_in_s * 60)])
    speed_times = np.zeros([len(presentation_ids), int(T_in_s * 60)])

    pupil_size = np.zeros([len(presentation_ids), int(T_in_s * 30)])
    pupil_times = np.zeros([len(presentation_ids), int(T_in_s * 30)])
    for trial, pres_id in enumerate(presentation_ids):
        pres_row = stim_table.loc[pres_id]

        mask = (session.running_speed['start_time'] >= pres_row['Start']) \
               & (session.running_speed['start_time'] < pres_row['End'])
        L = len(np.where(mask)[0])
        speeds[trial, :L] = session.running_speed[mask]['velocity']
        speed_times[trial, :L] = session.running_speed[mask]['start_time']
        speed_times[trial, :L] = speed_times[trial, :L] - speed_times[trial, 0]

        mask = (pupil_data.index.values >= pres_row['Start']) \
               & (pupil_data.index.values < pres_row['End'])
        L = len(np.where(mask)[0])  # let it be because the length could be shorter
        pupil_size[trial, :L] = pupil_data[mask].raw_pupil_area.values[:900]
        pupil_times[trial, :L] = pupil_data[mask].index.values[:900]
        pupil_times[trial, :L] = pupil_times[trial, :L] - pupil_times[trial, 0]
    return speeds, speed_times, pupil_size, pupil_times


def pop_average(session_id, probe, stim):
    data_directory = "D:/ecephys__project_cache/"
    directory = os.path.join(data_directory + '/session_' + str(session_id))
    matlab_directory = os.path.join(directory + '/MATLAB_files')
    probe_directory = os.path.join(matlab_directory + '/' + probe)
    stim_table = pd.read_csv(probe_directory + '/' + stim + '_meta.csv', index_col=0)
    session = EcephysSession.from_nwb_path(os.path.join(directory, 'session_' + str(session_id) + '.nwb'), api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })
    probe_id = session.units[session.units.probe_description == probe].probe_id
    session_units = session.units[session.units.probe_id == probe_id]
    cortex = ['VISp', 'VISl', 'VISli', 'VISrl', 'VISal', 'VISam', 'VISpm']
    session_units = session_units.rename(columns={"channel_local_index": "channel_id",
                                              "ecephys_structure_acronym": "ccf",
                                              'probe_vertical_position': "ypos"})

    session_units['unit_id'] = session_units.index
    cortical_units_ids = np.array([idx for idx, ccf in enumerate(session_units.ccf.values) if ccf in cortex])
    session_units_cortex = session_units.iloc[cortical_units_ids]
    #   spike counts
    duration = round(np.mean(stim_table.duration.values), 2)
    time_bin_edges = np.linspace(0, duration, int(duration * 1250) + 1)
    spike_counts = session.presentationwise_spike_counts(
        bin_edges=time_bin_edges,
        stimulus_presentation_ids=stim_table.index.values,
        unit_ids=session_units_cortex.index.valeus,
        binarize=True
    )
    # separate by layer


def state_definition(session, probe, stim, trial_duration, state_var_type='power',
                     state_est_type='clustering', phasic_event_data=None):
    if phasic_event_data is None:
        phasic_event_data = get_tau(session, probe, stim)
    print('Got LFP events')
    # KEEP IN MIND IF TAUS ARE DOWN-SAMPLED
    state_variable_matrix = construct_state_variables(phasic_event_data, trial_duration, state_var_type)
    print('Designed state variable matrix')

    # behavior
    running_speed, rs_times, pupil_data, pd_times = get_velocity_and_pupil_area(session, stim, probe,
                                                                                state_variable_matrix.shape[3] / 625)

    # state space visualization
    print('Visualizing state - space ...')
    # reduced_space
    win = [0.5, 0.25, 0.1, 0.03]
    Hz = [2, 4, 10, 30]
    nn = [5, 5, 10, 1000]
    state_reduced = state_variable_matrix[::2, :, :, :]
    [n_ch, n_tr, n_bands, T] = state_reduced.shape
    for n, bin in enumerate(win):
        mean_state = np.zeros([n_ch, n_tr, n_bands, Hz[n]*int(trial_duration/625)])
        binned_running_speed = np.zeros([n_tr, Hz[n]*int(trial_duration/625)])
        binned_pupil_size = np.zeros([n_tr, Hz[n] * int(trial_duration / 625)])
        for t in range(mean_state.shape[3]):
            bin_w = math.floor(bin*625)
            mean_state[:, :, :, t] = np.mean(state_reduced[:, :, :, 0 + t * bin_w: bin_w + t * bin_w], axis=3)

            for trial in range(n_tr):
                mask = np.where((rs_times[trial, :] >= 0 + t * bin) & (rs_times[trial, :] <= bin + t * bin))
                binned_running_speed[trial, t] = np.mean(running_speed[trial, mask], axis=1)

                mask = np.where((pd_times[trial, :] >= 0 + t * bin) & (pd_times[trial, :] <= bin + t * bin))
                binned_pupil_size[trial, t] = np.mean(pupil_data[trial, mask], axis=1)

        Tm = mean_state.shape[3]
        X = rearrange(mean_state, 'b c h w -> (c w) (b h)')
        times = np.matlib.repmat(np.linspace(0, trial_duration/625, Tm), 1, n_tr).reshape(-1)
        trials = np.matlib.repeat(np.linspace(0, n_tr, n_tr),Tm)
        scaled_X = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA(n_components=20)
        embedding_PCA = pca.fit_transform(scaled_X)

        reducer = umap.UMAP(n_neighbors=nn[n], metric='euclidean', n_components=2, min_dist=0.01)
        embedding_UMAP = reducer.fit_transform(embedding_PCA)
        dat_frame = pd.DataFrame()
        dat_frame['dim1'] = embedding_UMAP[:, 0]
        dat_frame['dim2'] = embedding_UMAP[:, 1]
        dat_frame['times'] = times
        dat_frame['trials'] = trials
        plt.figure()
        plt.scatter(dat_frame.dim1, dat_frame.dim2, c=dat_frame.times, cmap="BuPu", alpha=0.7)
        plt.title('UMAP projection over time')

        if state_est_type == 'clustering':
            state_estimates = state_estimation_clustering(embedding_UMAP, n_tr, Tm)
            plt.scatter(dat_frame.dim1, dat_frame.dim2, c=state_estimates[:], cmap="Spectral_r", alpha=0.7)
            plt.title('Bin Size: ' + str(bin) + 's, ' + 'UMAP projection over time')
            plt.figure()
            sns.heatmap(state_estimates)
            plt.title('States over trials')
            print('Bin Size: ' + str(bin) + 's, ' + 'State estimation complete')

        mean_state_shifted_axis = rearrange(mean_state, 'b c h w -> b h (c w)')
        top_ch, mid_ch, bottom_ch = 2, int(n_ch / 2), n_ch - 2

        plt.figure()
        plt.subplot(411)
        top_plot_mat = mean_state_shifted_axis[top_ch, :, :]
        sns.heatmap(top_plot_mat[:, :int(5*Hz[n]*trial_duration/625)])
        for l in range(1, 6):
            plt.plot(int(Hz[n]*trial_duration/625)*l*np.ones(5), np.arange(0, 5), 'w--')

        plt.subplot(412)
        mid_plot_mat = mean_state_shifted_axis[mid_ch, :, :]
        sns.heatmap(mid_plot_mat[:, :int(5*Hz[n]*trial_duration/625)])
        for l in range(1, 6):
            plt.plot(int(Hz[n]*trial_duration/625)*l*np.ones(5), np.arange(0, 5), 'w--')

        plt.subplot(413)
        bottom_plot_mat = mean_state_shifted_axis[bottom_ch, :, :]
        sns.heatmap(bottom_plot_mat[:, :int(5*Hz[n]*trial_duration/625)])
        for l in range(1, 6):
            plt.plot(int(Hz[n]*trial_duration/625)*l*np.ones(5), np.arange(0, 5), 'w--')

        plt.subplot(414)
        state_behavior = np.zeros([3, mean_state_shifted_axis.shape[2]])
        state_behavior[0, :] = state_estimates.reshape(-1)
        state_behavior[1, :] = binned_running_speed.reshape(-1)
        state_behavior[2, :] = binned_pupil_size.reshape(-1)/np.nanmax(binned_pupil_size.reshape(-1))
        sns.heatmap(state_behavior[:, :int(5 * Hz[n] * trial_duration / 625)])
        for l in range(1, 6):
            plt.plot(int(Hz[n]*trial_duration/625)*l*np.ones(5), np.arange(0, 5), 'w--')

    if state_est_type == 'bit-based':
        state_estimates = state_estimation_bitbased(state_variable_matrix)
        print('State estimation complete')

    # plots

    plt.figure()
    sns.heatmap(state_estimates)


state_definition(767871931, 'probeC', 'natural_movie_one_more_repeats', 30*625, 'rate', 'clustering')
# for checking its okay but switch to natural movie once the code is complete

