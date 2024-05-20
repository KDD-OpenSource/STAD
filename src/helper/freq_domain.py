import numpy as np
from autoperiod import Autoperiod


class FreqDomain:
    def __init__(self, ts, max_len=15000):
        """

        @param max_len: max length of time series to be considered for estimating the period
        @type max_len: int
        @param ts: univariate time series data
        @type ts: pandas Series
        """
        self.ts = ts
        self.max_len = max_len

    def get_period(self, ):
        """

        @return: the period of given time series
        @rtype: int
        """

        return self._get_period()

    def _get_period_by_fft(self):
        fft_wave = np.fft.fft(self.ts)
        max_fq_idx = fft_wave[1:].argmax()  # position with the maximum power (except freq=0 Hz)

        fft_fre = np.fft.fftfreq(n=self.ts.size)

        return int(1 / fft_fre.real[max_fq_idx])

    def _get_period(self):
        """
        get time series period by AutoPeriod method
        Reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.6950&rep=rep1&type=pdf
        @return: period
        @rtype: int
        """
        p = Autoperiod(self.ts.index[:self.max_len], self.ts.values[:self.max_len], plotter=None)

        return p.period
