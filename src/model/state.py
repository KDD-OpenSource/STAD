import numpy as np


class state:

    def __init__(self, name, prob, autoencoder, epsilon=0.1):
        self.name = name
        self.prob = prob
        self.autoencoder = autoencoder
        self.epsilon = epsilon


class statemodel:

    def __init__(self, maxStateNum=10):
        self.state_set = list()
        self.transition_set = set()
        self.maxStateNum = maxStateNum

    def process_new_concept(self, old_state, new_latent_data, hidden_size):
        p_new = self._density_estimation(new_latent_data)
        klds = dict()
        for state in self.state_set:
            klds[state.name] = self._symmetric_kl_divergence(state.prob, p_new, hidden_size)
        min_kld_state = min(klds, key=klds.get)
        if klds[min_kld_state] <= self.epsilon:
            self.transition_set.add((old_state.name, min_kld_state.name))
            return self.state_set[min_kld_state.name]
        else:
            ae = self._train_ae()
            name = 'new_name'
            new_state = state(name, p_new, ae)
            self.state_set.add(new_state)
            self.transition_set.add((old_state.name, name))
            if len(self.state_set) > self.maxStateNum:
                oldest_state = self.state_set.pop()
                for transition in self.transition_set:
                    if oldest_state in transition:
                        self.transition_set.remove(oldest_state)
            return self.state_set[name]

    def _density_estimation(self, latent_data, hidden_size):
        """

        @param latent_data: N*hidden_size
        """
        hp_list = np.argmax(latent_data, axis=1)
        return [((hp_list == i).sum() + 0.5) / (latent_data.shape[0] + hidden_size / 2) for i in range(hidden_size)]

    def _symmetric_kl_divergence(self, p_hist, p_new, hidden_size):
        return np.sum([p_hist[h] * np.log2(p_hist[h] / p_new[h]) + p_new[h] * np.log2(p_new[h] / p_hist[h])
                       for h in range(hidden_size)])

    def _train_ae(self, ):
        return None
