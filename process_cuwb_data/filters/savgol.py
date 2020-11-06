from scipy.signal import savgol_filter


class SavGolFilter:
    def __init__(self, x, window_length, polyorder, **kwargs):
        self.x = x
        self.window_length = window_length
        self.polyorder = polyorder
        self.kwargs = kwargs

    def filter(self):
        return savgol_filter(x=self.x, window_length=self.window_length, polyorder=self.polyorder, **self.kwargs)
