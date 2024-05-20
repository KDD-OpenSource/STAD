import numpy as np
from scipy import stats


class KSTest:

    def __init__(self, alpha, l_hist_size, l_new_size, hist_min_size):
        self.alpha = alpha
        self.l_hist_min_size = hist_min_size
        self.l_new_size = l_new_size  # the fixed size of queue L_new (#embeddings, i.e. #windows)
        self.l_hist_size = l_hist_size
        self.l_hist = []
        self.l_new = []

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

    def _one_d_test(self, one_d_hist, one_d_new):
        """
        st: the ks distance
        p_value: if p_value <= alpha, drift is detected
        """
        st, p_value = stats.ks_2samp(one_d_hist, one_d_new)
        return st, p_value

    def detect_drift(self, latent_representation):
        self.l_new.append(latent_representation.tolist())
        result = False
        p_value_buf = []
        if len(self.l_new) >= self.l_new_size:
            self.l_hist += self.l_new[:len(self.l_new) - self.l_new_size]
            self.l_new = self.l_new[len(self.l_new) - self.l_new_size:]
        else:
            return result, p_value_buf

        if len(self.l_hist) >= self.l_hist_size:
            self.l_hist = self.l_hist[len(self.l_hist)-self.l_hist_size:]

        if len(self.l_hist) < self.l_hist_min_size:
            return result, p_value_buf
        else:
            hist_df = np.array(self.l_hist)
            new_df = np.array(self.l_new)
            drift_dim = 0
            for i in range(hist_df.shape[1]):
                st, p_value = self._one_d_test(hist_df[:, i], new_df[:, i])
                p_value_buf.append(p_value)
                if p_value <= self.alpha:
                    drift_dim += 1
            if drift_dim == hist_df.shape[1]:
                result = True

            if result:
                self.l_hist = []
                self.l_new = []
        return result, p_value_buf


class KLDivergence:

    def __int__(self, ):
        pass

    def _kld(self, mu_1, sigma_1, mu_2, sigma_2):
        N = mu_1.shape[0]

        diff = mu_2 - mu_1

        # kl consists of three terms
        tr_term = np.trace(np.linalg.inv(sigma_2) @ sigma_1)
        det_term = np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1))
        quad_term = diff.T @ np.linalg.inv(sigma_2) @ diff
        return .5 * (tr_term + det_term + quad_term - N)

    def get_kld(self, histogram_1, histogram_2):
        kld = 0
        for i in range(len(histogram_1)):
            p1 = histogram_1[i] / sum(histogram_1) / len(histogram_1) if histogram_1[i] != 0 else np.inf
            p2 = histogram_2[i] / sum(histogram_2) / len(histogram_1) if histogram_2[i] != 0 else np.inf

            kld += 0.5 * (p1 * np.log2(p1 / p2) + p2 * np.log2(p2 / p1))

        return kld
