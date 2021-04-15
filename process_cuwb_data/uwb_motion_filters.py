import numpy as np

from .filters import ButterFilter, FiltFiltFilter, SosFiltFiltFilter, SavGolFilter


class TrayMotionButterFiltFiltFilter:
    """
    Structure for storing Butter + FiltFilt signal filtering parameters

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    """

    def __init__(self, N=4, Wn=0.75, fs=10, useSosFiltFilt=False):
        """
        Parameters
        ----------
        N : int
            The order of the butter signal filter.
        Wn : array_like
            The critical frequency or frequencies for the butter signal filter.
        fs : float
            The sampling frequency for the butter signal filter.
        useSosFiltFilt: boolean
            Use a forward-backward digital filter with cascaded second-order sections
        """
        self.N = N
        self.Wn = Wn
        self.fs = fs
        self.useSosFiltFilt = useSosFiltFilt

    def filter(self, series, btype='lowpass'):
        if self.useSosFiltFilt:
            sos = ButterFilter(N=self.N, Wn=self.Wn, fs=self.fs, btype=btype, output='sos').filter()
            series_filtered = SosFiltFiltFilter(sos=sos, x=series).filter()
        else:
            butter_b, butter_a = ButterFilter(N=self.N, Wn=self.Wn, fs=self.fs, btype=btype).filter()
            series_filtered = FiltFiltFilter(b=butter_b, a=butter_a, x=series).filter()

        return series_filtered


class TrayMotionSavGolFilter:
    """
    Structure for storing SavGol signal filtering parameters

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    """

    def __init__(self, window_length=31, polyorder=3, fs=10):
        """
        Parameters
        ----------
        window_length : int
            The length of the filter window (i.e., the number of coefficients)
        polyorder : int
            The order of the polynomial used to fit the samples
        fs : float
            The sampling frequency for the signal filter
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.delta = 1 / fs

    def filter(self, series, deriv):
        if len(series) <= 1 or self.window_length == 0:
            return None

        window_length = self.window_length
        while window_length > len(series):
            window_length = int(window_length / 2)
            if window_length % 2 == 0:
                window_length += 1

        polyorder = self.polyorder
        if self.polyorder >= window_length:
            polyorder = window_length - 1

        return SavGolFilter(
            series, deriv=deriv, window_length=window_length, polyorder=polyorder, delta=self.delta).filter()


class TrayCarryHeuristicFilter:
    def __init__(self, stdevs=2, window=30):
        self.stdevs = stdevs
        self.window = window

    def filter(self, df_predictions, prediction_column_name, inplace=False):
        if not inplace:
            df_predictions = df_predictions.copy()

        rolling_average = df_predictions[prediction_column_name].rolling(window=self.window, center=True).mean()
        rolling_stdev = df_predictions[prediction_column_name].rolling(window=self.window, center=True).std()
        anomalies = (abs(df_predictions[prediction_column_name] - rolling_average) > (self.stdevs * rolling_stdev))

        for idx, (time, row) in enumerate(df_predictions.iterrows()):
            if anomalies.loc(idx) == True:
                df_predictions.loc[idx, prediction_column_name] = int(not bool(row[prediction_column_name]))

        if not inplace:
            return df_predictions


class TrayCarryHmmFilter:
    def __init__(self,
                 initial_probability_vector=np.array([0.999, 0.001]),
                 transition_matrix=np.array([[.9999, 0.0001], [0.03, 0.97]]),
                 observation_matrix=np.array([[0.95, 0.05], [0.15, 0.85]])
                 ):
        self.initial_probability = initial_probability_vector

        # Transition Matrix
        #                 Not Carry(t)   Carry(t)
        # Not Carry(t-1)  0.9999         0.0001
        # Carry(t-1)      0.03           0.97
        self.transition_matrix = transition_matrix

        # Observation Matrix
        #                 Not Carry(Yt)   Carry(Yt)
        # Not Carry(Xt)   0.95            0.05
        # Carry(Xt)       0.15            0.85
        self.observation_matrix = observation_matrix

    def filter(self, df_predictions, prediction_column_name, inplace=False):
        if not inplace:
            df_predictions = df_predictions.copy()

        hmm_probability = np.zeros((len(df_predictions), 2))
        for idx, (_, row) in enumerate(df_predictions.iterrows()):
            observed_state = row[prediction_column_name]
            if idx == 0:
                unnormalized_probabilities = self.initial_probability * self.observation_matrix[:, observed_state]
            else:
                previous_probability = hmm_probability[idx - 1]
                unnormalized_probabilities = previous_probability.dot(
                    self.transition_matrix) * self.observation_matrix[:, observed_state]

            hmm_probability[idx] = unnormalized_probabilities / np.linalg.norm(unnormalized_probabilities)

        hmm_predictions = []
        for probabilities in hmm_probability:
            hmm_predictions.append(np.argmax(probabilities))

        df_predictions[prediction_column_name] = hmm_predictions

        if not inplace:
            return df_predictions


class SmoothLabelsFilter:
    def __init__(self,
                 window=10):
        self.window = window

    def filter(self, df_predictions, prediction_column_name, inplace=False):
        """
        Smooth out predicted labels by smoothing out changes that don't occur within a stable
        rolling window of carry events. Stable rolling windows are when the number of frames (windows)
        are uniform. i.e. A carry isn't recognized (stable) unless the state of "Carried" occurs n (window)
        times in a row.

        :param df: Dataframe to smooth
        :param prediction_column_name: Predicted carry column
        :param window: Number of frames that must be uniform to be considered a "stable" event period
        :param inplace: Modify dataframe in-place
        :return: Modified dataframe (if inplace is False)
        """
        if not inplace:
            df_predictions = df_predictions.copy()

        # 1) First use a rolling window to classify moments that fall into stable or not-stable periods
        # 2) Then look for the moments when stability changes (NS -> S or S -> NS)
        # 3) Capture the moment right before the stability change and store both the index and the carry state
        # 4) Update all moments between these stability changes with the most recent stable carry state value

        rolling_window = df_predictions[prediction_column_name].rolling(window=self.window, center=True)
        carry_stability = (rolling_window.min() == rolling_window.max())

        rolling_stability_change_window = carry_stability.rolling(window=2)
        carry_stability_change_moments = (rolling_stability_change_window.min() !=
                                          rolling_stability_change_window.max())

        change_moments = carry_stability_change_moments[carry_stability_change_moments == True]
        for ii_idx, (time_idx, row) in enumerate(change_moments.iteritems()):
            range_mask = (df_predictions.index > time_idx)

            if ii_idx < len(change_moments) - 1:
                range_mask = range_mask & (df_predictions.index <= change_moments.index[ii_idx + 1])

            prediction_change_idx = df_predictions.index.get_loc(time_idx) + 1
            if prediction_change_idx < len(df_predictions):
                df_predictions.loc[range_mask,
                                   prediction_column_name] = df_predictions.iloc[prediction_change_idx][prediction_column_name]

        if not inplace:
            return df_predictions
