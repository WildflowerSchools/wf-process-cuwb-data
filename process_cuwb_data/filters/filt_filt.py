from scipy.signal import filtfilt


class FiltFiltFilter:
    def __init__(self, b, a, x, **kwargs):
        self.b = b
        self.a = a
        self.x = x
        self.kwargs = kwargs

    def filter(self):
        return filtfilt(b=self.b, a=self.a, x=self.x, **self.kwargs)
