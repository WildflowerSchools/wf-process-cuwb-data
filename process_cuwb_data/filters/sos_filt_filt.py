from scipy.signal import sosfiltfilt


class SosFiltFiltFilter:
    def __init__(self, sos, x, **kwargs):
        self.sos = sos
        self.x = x
        self.kwargs = kwargs

    def filter(self):
        padlen = None
        if len(self.x) < 20:
            padlen = 0

        args = self.kwargs.copy()
        args['padlen'] = padlen

        return sosfiltfilt(sos=self.sos, x=self.x, **args)
