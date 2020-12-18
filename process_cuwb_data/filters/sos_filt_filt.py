from scipy.signal import sosfiltfilt


class SosFiltFiltFilter:
    def __init__(self, sos, x, **kwargs):
        self.sos = sos
        self.x = x
        self.kwargs = kwargs

    def filter(self):
        return sosfiltfilt(sos=self.sos, x=self.x, **self.kwargs)
