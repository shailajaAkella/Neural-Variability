import os
import h5py
import pandas as pd
import mat73
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


def get_tau(session, probe, stim, bands=['all']):
    data_dir1 = 'C:/Users/shailaja.akella/OneDrive - Allen Institute/Desktop/RA3'
    session_dir = data_dir1 + '/session_' + str(session) + '/MATLAB_files'
    probe_dir = session_dir + '/' + probe
    stim_file = (os.path.join(probe_dir, stim + '_data_Fs625.mat'))
    lfp_file = (os.path.join(probe_dir, stim + '_lfp_clean.mat'))
    assert(os.path.isfile(stim_file))
    data = h5py.File(stim_file, 'r')
    struct_data = data['data_mat']

    lfp_data = mat73.loadmat(lfp_file)
    lfp_channels = lfp_data['channels']

    data_dir2 = os.path.join('D:/ecephys__project_cache/session_' + str(session))
    session_dat = EcephysSession.from_nwb_path(os.path.join(data_dir2, 'session_' + str(session) + '.nwb'), api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })

    if 'all' in bands:
        bands = ['theta', 'beta', 'gamma1', 'gamma2']

    TAU = pd.DataFrame()
    for band in bands:
        for channel in range(len(list(struct_data[band]))):
            channel_data = data[struct_data[band][channel, 0]]
            layer = \
            session_dat.channels[session_dat.channels.index == lfp_channels[channel]].probe_vertical_position.values[0]
            trial_det = channel_data[band + '_det']
            n_tr = trial_det.shape[0]
            for trial in range(n_tr):
                if isinstance(data[trial_det[trial, 0]], h5py._hl.dataset.Dataset):
                    continue
                tau_tr = data[trial_det[trial, 0]]['tau'][0, 0]
                if isinstance(tau_tr, h5py.h5r.Reference):
                    num_taus = data[trial_det[trial, 0]]['tau'].shape[0]
                    for nt in range(num_taus):
                        tau_tr = data[data[trial_det[trial, 0]]['tau'][nt, 0]][0, 0]
                        pow_tr = data[data[trial_det[trial, 0]]['pow'][nt, 0]][0, 0]
                        len_tr = len(data[data[trial_det[trial, 0]]['PhEv'][nt, 0]][0, :])
                        TAU = TAU.append(
                            {'band': band, 'channel': int(channel),
                             'trial': int(trial), 'sample number': tau_tr, 'power': pow_tr,
                             'length': int(len_tr), 'ypos': layer}, ignore_index=True)
                else:
                    tau_tr = data[trial_det[trial, 0]]['tau'][0, 0]
                    pow_tr = data[trial_det[trial, 0]]['pow'][0, 0]
                    len_tr = len(data[trial_det[trial, 0]]['PhEv'][0, :])
                    TAU = TAU.append(
                        {'band': band, 'channel': int(channel),
                         'trial': int(trial), 'sample number': tau_tr, 'power': pow_tr,
                         'length': int(len_tr), 'ypos': layer}, ignore_index=True)

    return TAU

