from six import string_types
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.ndimage as ndi
import warnings

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysNwbSessionApi

import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)


class StimulusAnalysis(object):
    def __init__(self, ecephys_session, is_ophys_session=False, trial_duration=None, **kwargs):
        """
        :param ecephys_session: an EcephySession object or path to ece nwb file.
        """
        # TODO: Create a set of a class methods.
        if isinstance(ecephys_session, EcephysSession):
            self._ecephys_session = ecephys_session
        elif isinstance(ecephys_session, string_types):
            nwb_version = kwargs.get('nwb_version', 2)
            self._ecephys_session = EcephysSession.from_nwb_path(path=ecephys_session, nwb_version=nwb_version)
        elif isinstance(ecephys_session, EcephysNwbSessionApi):
            # nwb_version = kwargs.get('nwb_version', 2)
            self._ecephys_session = EcephysSession(api=ecephys_session)
        else:
            raise TypeError(f"Don't know how to make a stimulus analysis object from a {type(ecephys_session)}")

        self._unit_ids = None
        self._unit_filter = kwargs.get('filter', None)
        self._params = kwargs.get('params', None)
        self._unit_count = None
        self._stim_table = None
        self._conditionwise_statistics = None
        self._presentationwise_statistics = None
        self._presentationwise_spikes = None
        self._conditionwise_psth = None
        self._stimulus_conditions = None
        self._responsiveness_vs_shuffle = None
        self._responsiveness_vs_spontaneous = None

        self._spikes = None
        self._stim_table_spontaneous = None
        self._stimulus_key = kwargs.get('stimulus_key', None)
        self._running_speed = None
        # self._sweep_events = None
        # self._mean_sweep_events = None
        #  self._sweep_p_values = None
        self._metrics = None

        # start and stop times of blocks for the relevant stimulus. Used by the overall_firing_rate functions that only
        # need to be calculated once, but not accessable to the user
        self._block_starts = None
        self._block_stops = None

        # self._module_name = None  # TODO: Remove, .name() should be hardcoded
        if is_ophys_session:
            self._default_bin_resolution = 0.033
            self._bin_offset = 0.066
            self._use_amplitudes = True
        else:
            self._default_bin_resolution = 0.002
            self._bin_offset = 0.0
            self._use_amplitudes = False

        self._psth_resolution = kwargs.get('psth_resolution',
                                           self._default_bin_resolution)

        # Duration a sponteous stimulus should last for before it gets included in the analysis.
        self._spontaneous_threshold = kwargs.get('spontaneous_threshold', 100.0)

        # Roughly the length of each stimulus duration, used for calculating spike statististics
        self._trial_duration = trial_duration

        # Keeps track of preferred stimulus_condition_id for each unit
        self._preferred_condition = {}

    @property
    def ecephys_session(self):
        return self._ecephys_session

    @property
    def unit_ids(self):
        """Returns a list of unit IDs for which to apply the analysis"""
        if self._unit_ids is None:
            units_df = self.ecephys_session.units
            if isinstance(self._unit_filter, (list, tuple, np.ndarray, pd.Series)):
                # If the user passes a list/array of ids
                units_df = units_df.loc[self._unit_filter]

                print(units_df.ecephys_structure_acronym.unique())

            elif isinstance(self._unit_filter, dict):
                if 'unit_id' in self._unit_filter.keys():
                    # If user wants to filter by the unit_id column which is actually the dataframe index
                    units_df = units_df.loc[self._unit_filter['unit_id']]

                else:
                    # Create a mask for all units that match the all of specified conditions.
                    mask = True
                    for col, val in self._unit_filter.items():
                        if isinstance(val, (list, np.ndarray)):
                            mask &= units_df[col].isin(val)
                        else:
                            mask &= units_df[col] == val
                    units_df = units_df[mask]

            if units_df is None or units_df.empty:
                # If not units are found don't proceed.
                raise Exception('Could not find units for ecephys session.')

            self._unit_ids = units_df.index.values

        return self._unit_ids

    @property
    def unit_count(self):
        """Get the number of units."""
        if not self._unit_count:
            self._unit_count = len(self.unit_ids)
        return self._unit_count

    @property
    def name(self):
        """ Return the stimulus name."""
        return self._module_name

    @property
    def trial_duration(self):
        if self._trial_duration is None or self._trial_duration < 0.0:
            # TODO: Should we calculate trial_duration from min(stim_table['duration']) if not set by user/subclass?
            raise TypeError(f'Invalid value {self._trial_duration} for parameter "trial_duration".')

        return self._trial_duration

    @property
    def spikes(self):
        """Returns a dictionary of unit_id -> spike-times."""
        # TODO: This may be unecessary since we already have the presentationwise_spike_times table.
        if self._spikes is None:
            self._spikes = self.ecephys_session.spike_times
            if len(self._spikes) > self.unit_count:
                # if a filter has been applied such that not all the cells are being used in the analysis
                self._spikes = {k: v for k, v in self._spikes.items() if k in self.unit_ids}

        return self._spikes

    @property
    def stim_table(self):
        # Stimulus table is already in EcephysSession object, just need to subselect presentations for this stimulus.
        if self._stim_table is None:
            if self._stimulus_key is None:
                stims_table = self.ecephys_session.stimulus_presentations
                self._stimulus_key = self._find_stimulus_key(stims_table)
                if self._stimulus_key is None:
                    raise Exception('Could not find approipate stimulus_name key for current stimulus type. Please '
                                    'specify using the stimulus_key parameter.')

            self._stim_table = self.ecephys_session.get_stimulus_table(
                [self._stimulus_key] if isinstance(self._stimulus_key, string_types) else self._stimulus_key
            )

            if self._stim_table.empty:
                raise Exception(f'Could not find stimulus data with "stimulus_key" {self._stimulus_key}')

            # TODO: Should we remove columns that are not relevant to the selected stimulus? If a feature for another
            #  has random junk it can mess up stimulus_conditions table.

        return self._stim_table

    def _find_stimulus_key(self, stim_table):
        """Tries to guess the correct stimulus_key based on the data.

        :param stim_table:
        :return:
        """
        known_keys_lc = [k.lower() for k in self.__class__.known_stimulus_keys()]
        for table_key in stim_table['stimulus_name'].unique():
            if table_key.lower() in known_keys_lc:
                return table_key

        else:
            return None

    @property
    def known_spontaneous_keys(self):
        return ['spontaneous', "spontaneous_activity"]

    @property
    def total_presentations(self):
        """ Total nmber of presentations / trials"""
        return len(self.stim_table)

    @property
    def metrics_names(self):
        return [c[0] for c in self.METRICS_COLUMNS]

    @property
    def metrics_dtypes(self):
        return [c[1] for c in self.METRICS_COLUMNS]

    @property
    def METRICS_COLUMNS(self):
        raise NotImplementedError

    @property
    def stim_table_spontaneous(self):
        """Returns a stimulus table with only 'spontaneous' stimulus selected."""
        # Used by sweep_p_events for creating null dist.
        # TODO: This may not be need anymore? Ask the scientists if sweep_p_events will be required in the future.
        if self._stim_table_spontaneous is None:
            spont_table = self.ecephys_session.get_stimulus_table(self.known_spontaneous_keys)
            spont_table = spont_table[spont_table['duration'] > self._spontaneous_threshold]

            spont_table_check = spont_table[(spont_table.start_time > self.stim_table.start_time.min()) &
                                      (spont_table.start_time < self.stim_table.start_time.max())]

            if len(spont_table_check) == 0:
                spont_table = spont_table.iloc[[0]]  # just use the first one

            self._stim_table_spontaneous = spont_table

        return self._stim_table_spontaneous

    @property
    def null_condition(self):
        raise NotImplementedError()

    @property
    def conditionwise_psth(self):
        """For every unit and stimulus-condition construction a PSTH table. ie. the spike-counts at a each time-interval
        during a stimulus, averaged over all trials of the same stim condition.

        Each PSTH will count and average spikes over a time-window as determined by class parameter 'trial_duration'
        which ideally be a similar value as the duration of each stimulus (in seconds). The length of each time-bin
        is determined by the class parameter 'psth_resolution' (in seconds).

        Returns
        -------
        conditionwise_psth xarray.DataArray
            An 3D table that contains the PSTH for every unit/condition, with the following coordinates
                - stimulus_condition_id
                - time_relative_to_stimulus_onset
                - unit_id
        """

        if self._conditionwise_psth is None:
            if self._psth_resolution > self.trial_duration:
                warnings.warn('parameter "psth_resolution" > "trial_duration", PSTH will not be properly created.')

            # get the spike-counts for every stimulus_presentation_id
            dataset = self.ecephys_session.presentationwise_spike_counts(
                bin_edges=np.arange(0 + self._bin_offset,
                                    self.trial_duration + self._bin_offset,
                                    self._psth_resolution),
                stimulus_presentation_ids=self.stim_table.index.values,
                unit_ids=self.unit_ids
            )

            # replace the stimulus_presentation_id (which will be unique for every single stim) with the corresponding
            # stimulus_condition_id (which will be shared among presenations with the same conditions.
            da = dataset.assign_coords(stimulus_presentation_id=self.stim_table['stimulus_condition_id'].values)
            da = da.rename({'stimulus_presentation_id': 'stimulus_condition_id'})

            # Average spike counts across each stimulus_condition_id.
            n_stimuli = len(da['stimulus_condition_id'])
            n_cond_ids = len(np.unique(da.coords['stimulus_condition_id'].values))
            if n_stimuli == n_cond_ids:
                # If every condition_id is unique then calling groupby().mean() is unnecessary and will raise an error.
                self._conditionwise_psth = da
            else:
                self._conditionwise_psth = da.groupby('stimulus_condition_id').mean(dim='stimulus_condition_id')

        return self._conditionwise_psth

    @property
    def conditionwise_statistics(self):
        """Create a table of spike statistics, averaged and indexed by every unit_id, stimulus_condition_id pair.

        Returns
        -------
        conditionwise_statistics: pd.DataFrame
            A dataframe indexed by unit_id and stimulus_condition containing spike_count, spike_mean, spike_sem,
            spike_std and stimulus_presentation_count information.
        """
        if self._conditionwise_statistics is None:
            self._conditionwise_statistics = self.ecephys_session.conditionwise_spike_statistics(
                self.stim_table.index.values,
                self.unit_ids
            )

        return self._conditionwise_statistics

    @property
    def presentationwise_spike_times(self):
        """Constructs a table containing all the relevant spike_times plus the stimulus_presentation_id and unit_id
        for the given spike.

        Returns
        -------
        presentationwise_spike_times : pd.DataFrame
            Indexed by spike_time, each spike containing the corresponding stimulus_presentation_id and unit_id

        """
        if self._presentationwise_spikes is None:
            self._presentationwise_spikes = self.ecephys_session.presentationwise_spike_times(
                stimulus_presentation_ids=self.stim_table.index.values,
                unit_ids=self.unit_ids
            )

        return self._presentationwise_spikes

    @property
    def presentationwise_statistics(self):
        """Returns a table of the spike-counts, stimulus-conditions and running speed for every stimulus_presentation_id
        , unit_id pair.

        Returns
        -------
        presentationwise_statistics: pd.DataFrame
            MultiIndex : unit_id, stimulus_presentation_id
            Columns : spike_count, stimulus_condition_id, running_speed

        """
        if self._presentationwise_statistics is None:
            # for each presentation_id and unit_id get the spike_counts across the entire duration. Since there is only
            # a single bin we can drop time_relative_to_stimulus_onset.
            df = self.ecephys_session.presentationwise_spike_counts(
                bin_edges=np.array([0.0 + self._bin_offset, self.trial_duration + self._bin_offset]),
                stimulus_presentation_ids=self.stim_table.index.values,
                unit_ids=self.unit_ids
            ).to_dataframe().reset_index(level='time_relative_to_stimulus_onset', drop=True)

            # left join table with stimulus_condition_id and mean running_speed joined on stimulus_presentation_id
            df = df.join(self.stim_table.loc[df.index.levels[0].values]['stimulus_condition_id'])
            self._presentationwise_statistics = df.join(self.running_speed)

        return self._presentationwise_statistics

    @property
    def stimulus_conditions(self):
        """Returns a table of relevant stimulus_conditions.

        Returns
        -------
        pd.DataFrame :
            Index : stimulus_condition_id
            Columns : stimulus parameter types

        """

        if self._stimulus_conditions is None:
            condition_list = self.stim_table['stimulus_condition_id'].unique()
            self._stimulus_conditions = self.ecephys_session.stimulus_conditions[
                self.ecephys_session.stimulus_conditions.index.isin(condition_list)
            ]

        return self._stimulus_conditions

    @property
    def running_speed(self):
        """Construct a dataframe with the averaged running speed for each stimulus_presenation_id

        Return
        -------
        running_speed: pd.DataFrame:
            For each stimulus_presenation_id (index) contains the averaged running velocity.

        """
        if self._running_speed is None:
            def get_velocity(presentation_id):
                """Helper function for getting avg. velocities for a given presenation_id"""
                pres_row = self.stim_table.loc[presentation_id]
                mask = (self.ecephys_session.running_speed['start_time'] >= pres_row['start_time']) \
                       & (self.ecephys_session.running_speed['start_time'] < pres_row['stop_time'])

                return self.ecephys_session.running_speed[mask]['velocity'].mean()

            self._running_speed = pd.DataFrame(index=self.stim_table.index.values,
                                               data={'running_speed':
                                                         [get_velocity(i) for i in self.stim_table.index.values]
                                                     }).rename_axis('stimulus_presentation_id')

        return self._running_speed

    @property
    def responsiveness_vs_spontaneous(self):
        """baseline taken from randomized 'spontaneous' trial data."""
        if self._responsiveness_vs_spontaneous is None:
            self._responsiveness_vs_spontaneous, pd_stim, pd_spont = self._calc_spont_p_values()

        return self._responsiveness_vs_spontaneous, pd_stim, pd_spont

    @property
    def responsiveness_vs_shuffle(self):
        """baseline taken from data with randomly shuffled start times."""
        if self._responsiveness_vs_shuffle is None:
            self._responsiveness_vs_shuffle = self._calc_shuffle_p_values()

        return self._responsiveness_vs_shuffle

    def _calc_spont_p_values(self, n_samples=1000):
        """ Calculates the fraction of responses to each unit's preferred condition
        that are above a 95% of similar intervals during spontaneous activity.

        Parameters
        ==========
        n_samples : number of spontaneous intervals to use


        Returns
        =======
        responsiveness_vs_spontaneous : pd.DataFrame
            index: units

        """

        n_baseline = 1000

        def baseline_shift_range(arr, amount):
            return np.apply_along_axis(lambda x: x + (np.random.rand(1)) * amount, 1, arr)

        bin_edges = np.arange(0 + self._bin_offset,
                              self._trial_duration + self._bin_offset,
                              self._default_bin_resolution)

        spont_start = self.stim_table_spontaneous.iloc[0].start_time
        spont_end = self.stim_table_spontaneous.iloc[0].stop_time
        spont_trial_inds = np.ones((n_samples,)) * self.stim_table_spontaneous.index.values[0]

        baseline_shift_callback = lambda arr: baseline_shift_range(arr, spont_end - spont_start)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            responses_spont = self.ecephys_session.presentationwise_spike_counts(bin_edges,
                                                                                 stimulus_presentation_ids=spont_trial_inds,
                                                                                 unit_ids=self.unit_ids,
                                                                                 time_domain_callback=baseline_shift_callback)

            responses = self.ecephys_session.presentationwise_spike_counts(bin_edges,
                                                                           stimulus_presentation_ids=self.stim_table.index.values,
                                                                           unit_ids=self.unit_ids,
                                                                           )

            resp_mean = responses.mean(dim='time_relative_to_stimulus_onset')
            resp_spont_mean = responses_spont.mean(dim='time_relative_to_stimulus_onset')

            def get_sig_fraction(unit_id):
                resp_for_unit = resp_mean.sel(unit_id=unit_id).data
                spont_for_unit = resp_spont_mean.sel(unit_id=unit_id).data

                mask = self.stim_table.stimulus_condition_id == self._get_preferred_condition(unit_id)

                sig_level_spont = np.quantile(spont_for_unit, 0.95)
                sig_fraction_spont = np.sum(resp_for_unit[mask] > sig_level_spont) / np.sum(mask)

                return sig_fraction_spont

            sig_fraction = [get_sig_fraction(unit_id) for unit_id in self.unit_ids]
            # bin = np.linspace(np.min([np.min(resp_mean), np.min(resp_spont_mean)]),
            #                   np.max([np.min(resp_mean), np.max(resp_spont_mean)]), 1000)
            # resp_mean_hist, _ = np.histogram(resp_mean, bin)
            resp_mean = resp_mean.values.transpose()
            resp_spont_mean = resp_spont_mean.values.transpose()

        return pd.DataFrame(index=self.unit_ids, data={'sig_fraction': sig_fraction}), \
               pd.DataFrame(resp_mean, index=self.unit_ids), \
               pd.DataFrame(resp_spont_mean, index=self.unit_ids)



    def _calc_shuffle_p_values(self, shift_s=1000):
        """ Calculates the fraction of responses to each unit's preferred condition
        that are above a 95% of similar intervals for shuffled trial times.

        Parameters
        ==========
        shift_s : window for uniform trial start time shift


        Returns
        =======
        responsiveness_vs_spontaneous : pd.DataFrame
            index: units

        """

        def random_start_time_shift(arr, amount):
            return np.apply_along_axis(lambda x: x + (np.random.rand(1) - 0.5) * amount, 1, arr)

        bin_edges = np.arange(0 + self._bin_offset,
                              self._trial_duration + self._bin_offset,
                              self._default_bin_resolution)

        shuffle_start_time_callback = lambda arr: random_start_time_shift(arr, shift_s)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            responses_shuff = self.ecephys_session.presentationwise_spike_counts(bin_edges,
                                                                                 stimulus_presentation_ids=self.stim_table.index.values,
                                                                                 unit_ids=self.unit_ids,
                                                                                 time_domain_callback=shuffle_start_time_callback)

            responses = self.ecephys_session.presentationwise_spike_counts(bin_edges,
                                                                           stimulus_presentation_ids=self.stim_table.index.values,
                                                                           unit_ids=self.unit_ids,
                                                                           )

            resp_mean = responses.mean(dim='time_relative_to_stimulus_onset')
            resp_shuf_mean = responses_shuff.mean(dim='time_relative_to_stimulus_onset')

            def get_sig_fraction(unit_id):
                resp_for_unit = resp_mean.sel(unit_id=unit_id).data
                shuffle_for_unit = resp_shuf_mean.sel(unit_id=unit_id).data

                mask = self.stim_table.stimulus_condition_id == self._get_preferred_condition(unit_id)

                sig_level_shuffle = np.quantile(shuffle_for_unit, 0.95)
                sig_fraction_shuffle = np.sum(resp_for_unit[mask] > sig_level_shuffle) / np.sum(mask)

                return sig_fraction_shuffle

            sig_fraction = [get_sig_fraction(unit_id) for unit_id in self.unit_ids]

        return pd.DataFrame(index=self.unit_ids,
                            data={'sig_fraction': sig_fraction})

    @property
    def metrics(self):
        """Returns a pandas DataFrame of the stimulus response metrics for each unit."""
        raise NotImplementedError()

    def empty_metrics_table(self):
        # pandas can have issues interpreting type and makes the column 'object' type, this should enforce the
        # correct data type for each column
        empty_array = np.empty(self.unit_count, dtype=np.dtype(self.METRICS_COLUMNS))
        empty_array[:] = np.nan

        return pd.DataFrame(empty_array, index=self.unit_ids).rename_axis('unit_id')

    def _find_stimuli(self):
        raise NotImplementedError()

    ## Helper functions for calling metrics of individual units. ##
    def _get_preferred_condition(self, unit_id):
        """Determines and caches the prefered stimulus_condition_id based on mean spikes, ignoring null conditions."""
        # TODO: Should probably be renamed to preferred_condition_id so there is no confusion.
        if unit_id not in self._preferred_condition:
            # Use conditionwise_statistics 'spike_mean' column to find stimulus_condition_id that gives the highest
            # value.
            try:
                df = self.conditionwise_statistics.drop(index=self.null_condition, level=1)
            except (IndexError, NotImplementedError) as err:
                df = self.conditionwise_statistics

            # TODO: Calculated preferred condition_id once for all units and store in a table.
            self._preferred_condition[unit_id] = df.loc[unit_id]['spike_mean'].idxmax()

        return self._preferred_condition[unit_id]

    def _check_multiple_pref_conditions(self, unit_id, stim_cond_col, valid_conditions):
        # find all stimulus_condition which share the same 'stim_cond_col' (eg TF, ORI, etc) value, calculate the avg
        # spiking
        similar_conditions = [self.stimulus_conditions.index[self.stimulus_conditions[stim_cond_col] == v].tolist()
                              for v in valid_conditions]
        spike_means = [self.conditionwise_statistics.loc[unit_id].loc[condition_inds]['spike_mean'].mean()
                       for condition_inds in similar_conditions]

        # Check if there is more than one stimulus condition that provokes a maximum response
        return len(np.argwhere(spike_means == np.amax(spike_means))) > 1

    def _get_running_modulation(self, unit_id, preferred_condition, threshold=1.0):
        """Get running modulation for the preferred condition of a given unit"""
        subset = self.presentationwise_statistics[
            self.presentationwise_statistics['stimulus_condition_id'] == preferred_condition
            ].xs(unit_id, level='unit_id')

        if self._use_amplitudes:
            spike_counts = subset['spike_amplitudes'].values
        else:
            spike_counts = subset['spike_counts'].values

        running_speeds = subset['running_speed'].values
        return running_modulation(spike_counts, running_speeds, threshold)

    def _get_lifetime_sparseness(self, unit_id):
        """Computes lifetime sparseness of responses for one unit"""
        df = self.conditionwise_statistics.drop(index=self.null_condition, level=1)
        responses = df.loc[unit_id]['spike_mean'].values

        return lifetime_sparseness(responses)

    def _get_reliability(self, unit_id, preferred_condition):
        # Reliability calculation goes here:
        #   Depends on the trial-to-trial correlation of the smoothed response
        #   What smoothing window is appropriate for ephys? We need to test this more
        # TODO: If not implemented soon should be removed
        return np.nan

    def _get_fano_factor(self, unit_id, preferred_condition):
        #   See: https://en.wikipedia.org/wiki/Fano_factor
        subset = self.presentationwise_statistics[
            self.presentationwise_statistics['stimulus_condition_id'] == preferred_condition
            ].xs(unit_id, level=1)

        if self._use_amplitudes:
            spike_counts = subset['spike_amplitudes'].values
        else:
            spike_counts = subset['spike_counts'].values

        return fano_factor(spike_counts)

    def _get_time_to_peak(self, unit_id, preferred_condition):
        """Equal to the time of the maximum firing rate of the average PSTH at the preferred condition"""
        try:
            # TODO: Try to find a way to generalize that doesn't rely on conditionwise_psth
            psth = self.conditionwise_psth.sel(unit_id=unit_id, stimulus_condition_id=preferred_condition)
            peak_time = psth.where(psth == psth.max(), drop=True)['time_relative_to_stimulus_onset'][0].values
        except Exception as e:
            peak_time = np.nan

        return peak_time

    def _get_time_to_first_spike(self, unit_id, preferred_condition, minimum_time_s=0.03, maximum_time_s=0.2):
        """Equal to the median latency to first spike at the preferred condition"""

        presentation_ids = self.presentationwise_statistics.xs(unit_id, level=1)[
            self.presentationwise_statistics.xs(unit_id, level=1)['stimulus_condition_id']
            == preferred_condition].index.values

        df = self.presentationwise_spike_times[
            (self.presentationwise_spike_times['stimulus_presentation_id'].isin(presentation_ids)) &
            (self.presentationwise_spike_times['unit_id'] == unit_id)
            ]

        df = df.reset_index('spike_time')

        df['spike_latency'] = df['spike_time'] - self.stim_table.loc[df.stimulus_presentation_id].start_time.values

        df = df[(df.spike_latency > minimum_time_s) & \
                (df.spike_latency < maximum_time_s)]

        first_spike_times = df.groupby('stimulus_presentation_id').min()['spike_latency']

        time_to_first_spike = np.around(np.median(first_spike_times), 3)

        return time_to_first_spike

    def _get_overall_firing_rate(self, unit_id):
        """ Average firing rate over the entire stimulus interval"""
        if self._block_starts is None:
            # For the stimulus, create a list of start and stop times for the given block of trials. Only needs to be
            # calculated once TODO: see if python allows for private property variables
            start_time_intervals = np.diff(self.stim_table['start_time'])

            interval_end_inds = np.concatenate((np.where(start_time_intervals > self.trial_duration * 2)[0],
                                                np.array([self.total_presentations - 1])))
            interval_start_inds = np.concatenate((np.array([0]),
                                                  np.where(start_time_intervals > self.trial_duration * 2)[0] + 1))

            self._block_starts = self.stim_table.iloc[interval_start_inds]['start_time'].values
            self._block_stops = self.stim_table.iloc[interval_end_inds]['stop_time'].values
            # TODO: Check start and start times that differences are positive

        return overall_firing_rate(start_times=self._block_starts, stop_times=self._block_stops,
                                   spike_times=self.ecephys_session.spike_times[unit_id])

    def get_intrinsic_timescale(self, unit_ids):
        """Calculates the intrinsic timescale for a subset of units"""
        # TODO: Recently added by not yet being used, should indicate if/how it will be used! Maybe make protected?
        dataset = self.ecephys_session.presentationwise_spike_counts(
            bin_edges=np.arange(0, self.trial_duration, 0.025),
            stimulus_presentation_ids=self.stim_table.index.values,
            unit_ids=unit_ids
        )
        rsc_time_matrix = calculate_time_delayed_correlation(dataset)
        t, y, y_std, a, intrinsic_timescale, c = fit_exp(rsc_time_matrix)
        return intrinsic_timescale

    ### VISUALIZATION ###
    def plot_conditionwise_raster(self, unit_id):
        """ Plot a matrix of rasters for each condition (orientations x temporal frequencies) """
        _ = [self.plot_raster(cond, unit_id) for cond in self.stimulus_conditions.index.values]

    def plot_raster(self, condition, unit_id):
        raise NotImplementedError()

    @classmethod
    def known_stimulus_keys(cls):
        """Used for discovering the correct stimulus_name key for a given StimulusAnalysis subclass (when stimulus_key
        is not explicity set). Should return a list of "stimulus_name" strings.
        """
        raise NotImplementedError()


def running_modulation(spike_counts, running_speeds, speed_threshold=1.0):
    """Given a series of trials that include the spike-counts and (averaged) running-speed, does a statistical
    comparison to see if there was any difference in spike firing while running and while stationary.

    Requires at least 2 trials while the mouse is running and two when the mouse is stationary.

    Parameters
    ----------
    spike_counts : array of floats of size N.
        The spike counts for each trial
    running_speeds: array floats of size N.
        The running velocities (cm/s) of each trial.
    speed_threshold: float
        The minimum threshold for which the animal can be considered running (default 1.0).

    Returns
    -------
    p_value : float or Nan
        T-test p-value between the running and stationary trials.
    run_mod : float or Nan
        Relative difference between running and stationary mean firing rates.
    """
    if (len(spike_counts) != len(running_speeds)):
        warnings.warn('spike_counts and running_speeds must be arrays of the same shape.')
        return np.NaN, np.NaN

    is_running = running_speeds >= speed_threshold  # keep track of when the animal is and isn't running

    # Requires at-least two periods when the mouse is running and two when the mouse is not running.
    if 1 < np.sum(is_running) < (len(running_speeds) - 1):
        # calculate the relative differerence between mean running and stationary spike counts
        run = spike_counts[is_running]
        stat = spike_counts[np.invert(is_running)]

        run_mean = np.mean(run)
        stat_mean = np.mean(stat)

        if run_mean == stat_mean == 0:
            return np.NaN, np.NaN
        if run_mean > stat_mean:
            run_mod = (run_mean - stat_mean) / run_mean
        else:
            run_mod = -1 * (stat_mean - run_mean) / stat_mean

        # Get the p-value between the two populations.
        (_, p) = st.ttest_ind(run, stat, equal_var=False)
        return p, run_mod
    else:
        return np.NaN, np.NaN


def lifetime_sparseness(responses):
    """Computes the lifetime sparseness for one unit. See Olsen & Wilson 2008.

    Parameters
    ----------
    responses : array of floats
        An array of a unit's spike-counts over the duration of multiple trials within a given session

    Returns
    -------
    lifetime_sparsness : float
        The lifetime sparseness for one unit
    """
    if len(responses) <= 1:
        # Unable to calculate, return nan
        warnings.warn('responses array must contain at least two or more values to calculate.')
        return np.nan

    coeff = 1.0 / len(responses)
    return (1.0 - coeff * ((np.power(np.sum(responses), 2)) / (np.sum(np.power(responses, 2))))) / (1.0 - coeff)


def fano_factor(spike_counts):
    """Computers the fano factor (var/mean) for the spike-counts across a series of trials.

    Parameters
    ----------
    spike_counts : array
        The spike counts across a series of 2 or more trials

    Returns
    -------
    fano_factor : float
    """
    spike_count_mean = np.mean(spike_counts)
    if spike_count_mean == 0:
        return np.nan

    return np.var(spike_counts) / spike_count_mean


def overall_firing_rate(start_times, stop_times, spike_times):
    """Computes the global firing rate of a series of spikes, for only those values within the given start and
    stop times.

    Parameters
    ----------
    start_times : array of N floats
        A series of stimulus block start times (seconds)
    stop_times : array of N floats
        Times when the stimulus block ends
    spike_times : array of floats
        A list of spikes for a given unit

    Returns
    -------
    firing_rate : float
    """
    if len(start_times) != len(stop_times):
        warnings.warn('start_times and stop_times must be arrays of the same length')
        return np.nan

    if len(spike_times) == 0:
        # No spikes, firing rate 0
        return 0.0

    total_time = np.sum(stop_times - start_times)
    if total_time <= 0:
        # Probably start and stop times got inverted.
        warnings.warn(f'The total duration was {total_time} seconds.')
        return np.nan

    return np.sum(spike_times.searchsorted(stop_times) - spike_times.searchsorted(start_times)) / total_time


def get_fr(spikes, num_timestep_second=30, sweep_length=3.1, filter_width=0.1):
    """Uses a gaussian convolution to convert the spike-times into a contiguous firing-rate series.

    Parameters
    ----------
    spikes : array
        An array of spike times (shifted to start at 0)
    num_timestep_second : float
        The sampling frequency
    sweep_length : float
        The lenght of the returned array
    filter_width: float
        The window of the gaussian method

    Returns
    -------
    firing_rate : float
        A linear-spaced array of length num_timestep_second*sweep_length of the smoothed firing rates series.
    """
    spikes = spikes.astype(float)
    spike_train = np.zeros((int(sweep_length * num_timestep_second)))
    spike_train[(spikes * num_timestep_second).astype(int)] = 1
    filter_width = int(filter_width * num_timestep_second)
    fr = ndi.gaussian_filter(spike_train, filter_width)
    return fr


def reliability(unit_sweeps, padding=1.0, num_timestep_second=30, filter_width=0.1, window_beg=0, window_end=None):
    """Computes the trial-to-trial reliability for a set of sweeps for a given cell

    :param unit_sweeps:
    :param padding:
    :return:
    """
    if isinstance(unit_sweeps, (list, tuple)):
        unit_sweeps = np.array([np.array(l) for l in unit_sweeps])

    unit_sweeps = unit_sweeps + padding  # DO NOT use the += as for python arrays that will do in-place modification
    corr_matrix = np.empty((len(unit_sweeps), len(unit_sweeps)))
    fr_window = slice(window_beg, window_end)
    for i in range(len(unit_sweeps)):
        fri = get_fr(unit_sweeps[i], num_timestep_second=num_timestep_second, filter_width=filter_width)
        for j in range(len(unit_sweeps)):
            frj = get_fr(unit_sweeps[j], num_timestep_second=num_timestep_second, filter_width=filter_width)
            # Warning: the pearson coefficient is likely to have a denominator of 0 for some cells/stimulus and give
            # a divide by 0 warning.
            r, p = st.pearsonr(fri[fr_window], frj[fr_window])
            corr_matrix[i, j] = r

    inds = np.triu_indices(len(unit_sweeps), k=1)
    upper = corr_matrix[inds[0], inds[1]]
    return np.nanmean(upper)


def osi(orivals, tuning):
    """Computes the orientation selectivity of a cell. The calculation of the orientation is done using the normalized
    circular variance (CirVar) as described in Ringbach 2002

    Parameters
    ----------
    ori_vals : complex array of length N
         Each value the oriention of the stimulus.
    tuning : float array of length N
        Each value the (averaged) response of the cell at a different orientation.

    Returns
    -------
    osi : float
        An N-dimensional array of the circular variance (scalar value, in radians) of the responses.
    """
    if len(orivals) == 0 or len(orivals) != len(tuning):
        warnings.warn('orivals and tunings are of different lengths')
        return np.nan

    tuning_sum = tuning.sum()
    if tuning_sum == 0.0:
        return np.nan

    cv_top = tuning * np.exp(1j * 2 * orivals)
    return np.abs(cv_top.sum()) / tuning_sum


def dsi(orivals, tuning):
    """Computes the direction selectivity of a cell. See Ringbach 2002, Van Hooser 2014

    Parameters
    ----------
    ori_vals : complex array of length N
         Each value the oriention of the stimulus.
    tuning : float array of length N
        Each value the (averaged) response of the cell at a different orientation.

    Returns
    -------
    osi : float
        An N-dimensional array of the circular variance (scalar value, in radians) of the responses.
    """
    if len(orivals) == 0 or len(orivals) != len(tuning):
        warnings.warn('orivals and tunings are of different lengths')
        return np.nan

    tuning_sum = tuning.sum()
    if tuning_sum == 0.0:
        return np.nan

    cv_top = tuning * np.exp(1j * orivals)
    return np.abs(cv_top.sum()) / tuning_sum


def deg2rad(arr):
    """ Converts array-like input from degrees to radians"""
    # TODO: Is there any reason not to use np.deg2rad?
    return arr / 180 * np.pi


def fit_exp(rsc_time_matrix):
    intr = abs(rsc_time_matrix)
    tmp = np.nanmean(intr, axis=0)
    n = intr.shape[0]

    t = np.arange(len(tmp))[1:]
    y = gaussian_filter(np.nanmean(tmp, axis=0)[1:], 0.8)

    p, amo = curve_fit(lambda t, a, b, c: a * np.exp(-1 / b * t) + c, t, y, p0=(-4, 2, 1), maxfev=1000000000)

    a = p[0]
    b = p[1]  # this is the intrinsic timescale
    c = p[2]
    y_std = np.nanstd(tmp, axis=0)[1:] / np.sqrt(n)

    return t, y, y_std, a, b, c


def calculate_time_delayed_correlation(dataset):
    nbins = dataset.time_relative_to_stimulus_onset.size
    num_units = dataset.unit_id.size

    rsc_time_matrix = np.zeros((num_units, nbins, nbins)) * np.nan

    for unit_idx, unit in enumerate(dataset.unit_id):

        spikes_for_unit = dataset.sel(unit_id=unit).data

        for i in np.arange(nbins - 1):
            for j in np.arange(i + 1, nbins):
                good_trials = (spikes_for_unit[:, i] * spikes_for_unit[:, j]) > 0  # remove zero spike count bins
                r, p = st.pearsonr(spikes_for_unit[good_trials, i], spikes_for_unit[good_trials, j])
                rsc_time_matrix[unit_idx, i, j] = r

    return rsc_time_matrix