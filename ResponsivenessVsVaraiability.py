import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stimulus_analysis


def get_neural_variability(rate):
    # calculate across trial NV
    # ref: Churchland, Mark M., et al. "Neural variability in premotor cortex provides
    # a signature of motor preparation." Journal of Neuroscience 26.14 (2006): 3697-3712.

    c = 1.57 # needs to determined based on rate calculation
    eps = eps_d = 0.1
    [n_tr, T] = rate.shape
    NV = np.zeros(T)

    for t in range(T):
        r_bar = np.mean(rate[:, t])
        NV[t] = c * (1 / (r_bar + c * eps_d)) * (eps + (np.sum((rate[:, t] - r_bar) ** 2)) / (n_tr - 1))
        # NV[t] = np.var(rate[:, t])/np.mean(rate[:, t])
    return NV


def get_FF(sua):
    return np.var(np.sum(sua, axis=0))/np.mean(np.sum(sua, axis=0))


def zscore(x):
    return (x - np.mean(x, axis=-1).reshape(-1, 1)) / np.std(x, axis=-1).reshape(-1, 1)


sns.set_theme()
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
data_directory = "D:/ecephys__project_cache/"
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

print('Getting all sessions...')
sessions = cache.get_session_table()

functional_connectivity_sessions = sessions[(sessions.session_type == 'functional_connectivity')].iloc[[1, 4]]

# for each mouse
dur = {'drifting_gratings_75_repeats': 2*1025, 'drifting_gratings_contrast': 0.5*1025}
PD = pd.DataFrame()
for session_id, row in functional_connectivity_sessions.iterrows():
    directory = os.path.join(data_directory + '/session_' + str(session_id))
    session = EcephysSession.from_nwb_path(os.path.join(directory, 'session_' + str(session_id) + '.nwb'), api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })
    # get keys
    print('Getting keys..')
    keys = []
    search = 'meta'
    list_of_files = os.listdir(directory + '/MATLAB_files/probeA')
    for file_name in list_of_files:
        if search in file_name:
            keys.append(file_name.split('_meta')[0])
    print(keys)

    keys = ['drifting_gratings_contrast']
    # for each area
    for key in keys:
        print('Data for stimulus: ' + str(key))
        matlab_directory = os.path.join(directory + '/MATLAB_files')

        unit_info = stimulus_analysis.StimulusAnalysis(session, stimulus_key=key, trial_duration=dur[key]/1025)
        responsiveness, _, _ = unit_info.responsiveness_vs_spontaneous

        print('For area corresponding to:')

        for probe_id, probe in session.probes.iterrows():
            print(' ' + probe.description)
            probe_directory = os.path.join(matlab_directory + '/' + probe.description)
            # get stimulus_table
            stim_table = pd.read_csv(probe_directory + '/' + key + '_meta.csv', index_col=0)

            # check duration
            duration = round(np.mean(stim_table.duration.values), 2)
            print('  Trial duration is ' + str(duration) + 's')

            # visual cortex acronyms
            cortex = ['VISp', 'VISl', 'VISli', 'VISrl', 'VISal', 'VISam', 'VISpm']

            # get units from the visual cortex
            session_units = session.units[session.units.probe_id == probe_id]
            session_units = session_units.rename(columns={"channel_local_index": "channel_id",
                                                          "ecephys_structure_acronym": "ccf",
                                                          'probe_vertical_position': "ypos"})
            session_units['unit_id'] = session_units.index
            cortical_units_ids = np.array([idx for idx, ccf in enumerate(session_units.ccf.values) if ccf in cortex])
            session_units_cortex = session_units.iloc[cortical_units_ids]

            data = responsiveness.loc[session_units_cortex.index.values]
            data['probe'] = probe.description
            for unit_id, row in session_units_cortex.iterrows():
                #   spike counts
                time_bin_edges = np.linspace(0, duration, int(duration * 1250) + 1)
                spike_counts = session.presentationwise_spike_counts(
                    bin_edges=time_bin_edges,
                    stimulus_presentation_ids=stim_table.index.values,
                    unit_ids=row.unit_id,
                    binarize=True
                )

                mask = stim_table.stimulus_condition_id == unit_info._get_preferred_condition(unit_id)

                SUA = np.squeeze(spike_counts.values)
                FF1 = get_FF(SUA[mask, :int(0.1*1250)])
                FF2 = get_FF(SUA[mask, int(0.1*1250):])
                filt = np.ones(int(1250 * 0.05))
                SUA = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=1, arr=SUA)

                layer = session_units_cortex.loc[unit_id].ypos

                T = SUA.shape[1]
                response = get_neural_variability(SUA[mask, :])
                time = np.linspace(0, duration, T)
                data.loc[unit_id, 'layer'] = layer
                data.loc[unit_id, 'ff1'] = FF1
                data.loc[unit_id, 'ff2'] = FF2
                data.loc[unit_id, 'mean_var1'] = np.mean(response[:int(0.1*1250)])
                data.loc[unit_id, 'mean_var2'] = np.mean(response[int(0.1*1250):])
                data.loc[unit_id, 'max_var1'] = np.max(response[:int(0.1*1250)])
                data.loc[unit_id, 'max_var2'] = np.max(response[int(0.1*1250):])

            PD = PD.append(data)

probes = ['probeC', 'probeD', 'probeF', 'probeE', 'probeB', 'probeA']
sns.lmplot(x='sig_fraction', y='ff1', col='probe', data=PD,
                    col_order=probes, col_wrap=3)
sns.lmplot(x='sig_fraction', y='ff2', col='probe', data=PD,
                    col_order=probes, col_wrap=3)
plt.show()

