import numpy as np

from .filters import ButterFilter, FiltFiltFilter, SavGolFilter


class TrayMotionButterFiltFiltFilter:
    """
    Structure for storing Butter + FiltFilt signal filtering parameters

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    """

    def __init__(self, N=3, Wn=0.01, fs=10):
        """
        Parameters
        ----------
        N : int
            The order of the butter signal filter.
        Wn : array_like
            The critical frequency or frequencies for the butter signal filter.
        fs : float
            The sampling frequency for the butter signal filter.
        """
        self.N = N
        self.Wn = Wn
        self.fs = fs

    def filter(self, series, btype):
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
        return SavGolFilter(
            series, deriv=deriv, window_length=self.window_length, polyorder=self.polyorder, delta=self.delta).filter()


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
                 transition_matrix=np.array([[.9999, 0.0001], [0.05, 0.95]]),
                 observation_matrix=np.array([[0.95, 0.05], [0.2, 0.8]])
                 ):
        self.initial_probability = initial_probability_vector

        # Transition Matrix
        #                 Not Carry(t)   Carry(t)
        # Not Carry(t-1)  0.9999         0.0001
        # Carry(t-1)      0.05           0.95
        self.transition_matrix = transition_matrix

        # Observation Matrix
        #                 Not Carry(Yt)   Carry(Yt)
        # Not Carry(Xt)   0.95            0.05
        # Carry(Xt)       0.2             0.8
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
