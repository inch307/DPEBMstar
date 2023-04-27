######################
## This code is from 
## https://github.com/interpretml/interpret/blob/2007c485a11091312daab5afc03da8f49530560a/python/interpret-core/interpret/glassbox/ebm/utils.py#L1241
######################
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar, brentq

class DPUtils:
    @staticmethod
    def calc_classic_noise_multi(total_queries, target_epsilon, delta, sensitivity):
        variance = (8*total_queries*sensitivity**2 * np.log(np.exp(1) + target_epsilon / delta)) / target_epsilon ** 2
        return np.sqrt(variance)
    
    @staticmethod
    def calc_gdp_mu(target_epsilon, delta):
        ''' GDP analysis following Algorithm 2 in: https://arxiv.org/abs/2106.09680. 
        '''
        def f(mu, eps, delta):
            return DPUtils.delta_eps_mu(eps, mu) - delta


        if f(1e-10, target_epsilon, delta) * f(1000, target_epsilon, delta) > 0:
            return 0

        else:
            final_mu = brentq(lambda x: f(x, target_epsilon, delta), 1e-10, 1000)
            return final_mu

    @staticmethod
    def calc_gdp_noise_multi(total_queries, target_epsilon, delta):
        ''' GDP analysis following Algorithm 2 in: https://arxiv.org/abs/2106.09680. 
        '''
        def f(mu, eps, delta):
            return DPUtils.delta_eps_mu(eps, mu) - delta

        final_mu = brentq(lambda x: f(x, target_epsilon, delta), 1e-10, 1000)
        sigma = np.sqrt(total_queries) / final_mu
        return sigma

    @staticmethod
    def noise_from_mu(total_queries, mu):
        sigma = np.sqrt(total_queries) / mu
        return sigma

    # General calculations, largely borrowed from tensorflow/privacy and presented in https://arxiv.org/abs/1911.11607
    @staticmethod
    def delta_eps_mu(eps, mu):
        ''' Code adapted from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py#L44
        '''
        return norm.cdf(-eps/mu + mu/2) - np.exp(eps) * norm.cdf(-eps/mu - mu/2)

    @staticmethod
    def eps_from_mu(mu, delta):
        ''' Code adapted from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py#L50
        '''
        def f(x):
            return DPUtils.delta_eps_mu(x, mu)-delta    
        return root_scalar(f, bracket=[0, 500], method='brentq').root

    @staticmethod
    def private_numeric_binning(col_data, sample_weight, noise_scale, max_bins, min_val, max_val, lap):
        uniform_weights, uniform_edges = np.histogram(col_data, bins=max_bins*2, range=(min_val, max_val), weights=sample_weight)
        if lap:
            noisy_weights = uniform_weights + np.random.laplace(0, noise_scale, size=uniform_weights.shape[0])
        else:
            noisy_weights = uniform_weights + np.random.normal(0, noise_scale, size=uniform_weights.shape[0])
        
        # Postprocess to ensure realistic bin values (min=0)
        noisy_weights = np.clip(noisy_weights, 0, None)

        # TODO PK: check with Harsha, but we can probably alternate the taking of nibbles from both ends
        # so that the larger leftover bin tends to be in the center rather than on the right.

        # Greedily collapse bins until they meet or exceed target_weight threshold
        sample_weight_total = len(col_data) if sample_weight is None else np.sum(sample_weight)
        target_weight = sample_weight_total / max_bins
        bin_weights, bin_cuts = [], [uniform_edges[0]]
        curr_weight = 0
        for index, right_edge in enumerate(uniform_edges[1:]):
            curr_weight += noisy_weights[index]
            if curr_weight >= target_weight:
                bin_cuts.append(right_edge)
                bin_weights.append(curr_weight)
                curr_weight = 0

        if len(bin_weights) < 2:
            # we are going to remove this feature

            bin_weights, bin_cuts = [], []
        else:
            bin_cuts = np.array(bin_cuts, dtype=np.float64)

            # All leftover datapoints get collapsed into final bin
            bin_weights[-1] += curr_weight

        return bin_cuts, bin_weights

    @staticmethod
    def private_categorical_binning(col_data, sample_weight, noise_scale, max_bins, lap):
        # Initialize estimate
        col_data = col_data.astype('U')
        uniq_vals, uniq_idxs = np.unique(col_data, return_inverse=True)
        uniq_vals_non_mask = uniq_vals
        weights = np.bincount(uniq_idxs, weights=sample_weight, minlength=len(uniq_vals))

        if lap:
            weights = weights + np.random.laplace(0, noise_scale, size=weights.shape[0])
        else:
            weights = weights + np.random.normal(0, noise_scale, size=weights.shape[0])

        # Postprocess to ensure realistic bin values (min=0)
        weights = np.clip(weights, 0, None)

        # Collapse bins until target_weight is achieved.
        sample_weight_total = len(col_data) if sample_weight is None else np.sum(sample_weight)
        target_weight = sample_weight_total / max_bins
        small_bins = np.where(weights < target_weight)[0]
        if len(small_bins) > 0:
            other_weight = np.sum(weights[small_bins])
            mask = np.ones(weights.shape, dtype=bool)
            mask[small_bins] = False

            # Collapse all small bins into "DPOther"
            uniq_vals = np.append(uniq_vals[mask], "DPOther")
            weights = np.append(weights[mask], other_weight)

            if other_weight < target_weight:
                if len(weights) < 2:
                    # since we're adding unbounded random noise, it's possible that the total weight is less than the
                    # threshold required for a single bin.  It could in theory even be negative.
                    # clip to the target_weight
                    weights[0] = target_weight
                else:
                    # If "DPOther" bin is too small, absorb 1 more bin (guaranteed above threshold)
                    collapse_bin = np.argmin(weights[:-1])
                    mask = np.ones(weights.shape, dtype=bool)
                    mask[collapse_bin] = False

                    # Pack data into the final "DPOther" bin
                    weights[-1] += weights[collapse_bin]

                    # Delete absorbed bin
                    uniq_vals = uniq_vals[mask]
                    weights = weights[mask]

        return uniq_vals_non_mask, uniq_vals, weights

    @staticmethod
    def validate_eps_delta(eps, delta):
        if eps is None or eps <= 0 or delta is None or delta <= 0:
            raise ValueError(f"Epsilon: '{eps}' and delta: '{delta}' must be set to positive numbers")