import math
import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
import warnings
import pandas as pd
from scipy import stats
import mat73
import matplotlib.pyplot as plt
import seaborn as sns
from allensdk.brain_observatory.ecephys.visualization import raster_plot
from Get_Tau_Transient import get_tau
import stimulus_analysis
from scipy.spatial.distance import pdist, squareform


def GaussMat(X):
    # Gram matrix
    X = X.reshape(-1, 1)
    N = X.shape[0]
    sig = (1.06*np.nanstd(X[:]))*(N**(-1/5))
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-pairwise_sq_dists / sig ** 2)
    return K


def calculate_MI(X, Y):
    # ref: Giraldo, Luis Gonzalo Sanchez, Murali Rao, and Jose C. Principe.
    # "Measures of entropy from data using infinitely divisible kernels."
    # IEEE Transactions on Information Theory 61.1 (2014): 535-548.

    alpha = 1.01
    N = len(X)
    Kx = GaussMat(X)/N
    Ky = GaussMat(Y)/N
    Kxy = Kx*Ky*N # must be element wise multiplication

    Ly, _ = np.linalg.eig(Ky)
    absLy = np.abs(Ly)
    Hy = (1/(1-alpha)) * math.log2(np.sum(absLy**alpha))

    Lx, _ = np.linalg.eig(Kx)
    absLx = np.abs(Lx)
    Hx = (1 / (1 - alpha)) * math.log2(np.sum(absLx ** alpha))

    Lxy,_ = np.linalg.eig(Kxy)
    absLxy = np.abs(Lxy)
    Hxy = (1 / (1 - alpha)) * math.log2(np.sum(absLxy ** alpha))

    MI = Hx + Hy - Hxy

    return MI/np.sqrt(Hx*Hy) #normalization akin to pearson correlation.


warnings.filterwarnings("ignore")
sns.set_theme()
pd.set_option("display.max_columns", None)
data_directory = "D:/ecephys__project_cache/"
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

print('Getting all sessions...')
sessions = cache.get_session_table()

functional_connectivity_sessions = sessions[(sessions.session_type == 'functional_connectivity')].iloc[[1, 4]]

order = {
    'probeA': 5,
    'probeB': 4,
    'probeC': 0,
    'probeD': 1,
    'probeE': 3,
    'probeF': 2
}

dur = {'drifting_gratings_contrast': 625}
# for each mouse

for session_id, row in functional_connectivity_sessions.iterrows():
    directory = os.path.join(data_directory + '/session_' + str(session_id))
    session = EcephysSession.from_nwb_path(os.path.join(directory, 'session_' + str(session_id) + '.nwb'), api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })

    keys = ['drifting_gratings_contrast']
    for key in keys:
        matlab_directory = os.path.join(directory + '/MATLAB_files')

        print('  Data for stimulus: ' + str(key))

        # find maximally responsive units
        unit_info = stimulus_analysis.StimulusAnalysis(session, stimulus_key=key, trial_duration=dur[key] / 1025)
        responsiveness_all, pd_stim_all, pd_spont_all = unit_info.responsiveness_vs_spontaneous
        responsiveness_all.index.names = ['unit_id']

        # for each area
        print('For area corresponding to:')
        for probe_id, probe in session.probes.iterrows():
            print(' ' + probe.description)
            # get stimulus_table
            probe_directory = os.path.join(matlab_directory + '/' + probe.description)
            stim_table = pd.read_csv(probe_directory + '/' + key + '_meta.csv', index_col=0)

            duration = round(np.mean(stim_table.duration.values), 2)

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

            match = np.intersect1d(responsiveness_all.index.values, session_units_cortex.unit_id.values)
            responsiveness = responsiveness_all.loc[match]
            pd_stim = pd_stim_all.loc[match]
            pd_spont = pd_spont_all.loc[match]
            responsiveness['sig_fraction'] = np.round(responsiveness['sig_fraction'].values, 1)
            grouped_responsiveness = responsiveness.sort_values('unit_id').groupby(['sig_fraction'])

            mi = pd.DataFrame()
            # may be don't normalize?
            normalized_stim_mean_counts = pd_stim.divide(pd_spont.mean(axis=1), axis=0)
            normalized_stim_mean_counts = normalized_stim_mean_counts[
                ~normalized_stim_mean_counts.isin([np.nan, np.inf, -np.inf]).any(1)]
            contrast_data = stim_table.contrast.values
            orientation_data = stim_table.orientation.values
            for unit, counts in normalized_stim_mean_counts.iterrows():
                mi_contrast = calculate_MI(counts.values, contrast_data)
                mi_orientation = calculate_MI(counts.values, orientation_data)
                m_temp = pd.DataFrame()
                m_temp['MI'] = [mi_contrast, mi_orientation]
                m_temp['unit_id'] = unit
                m_temp['feature'] = ['contrast', 'orientation']
                m_temp['sig_fraction'] = responsiveness.loc[unit].sig_fraction
                mi = mi.append(m_temp)

            sns.lineplot(x='sig_fraction', y='MI', hue='feature', ci=None,
                         style='feature', markers=True, dashes=False)
            plt.title('For Probe' + probe.description)




