from scipy.signal import butter


class ButterFilter:
    def __init__(self, N, Wn, **kwargs):
        self.N = N
        self.Wn = Wn
        self.kwargs = kwargs

    def filter(self):
        return butter(N=self.N, Wn=self.Wn, **self.kwargs)
