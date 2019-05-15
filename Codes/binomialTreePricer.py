import numpy as np

class asianOptionBinomialTree:

    def __init__(self, num_steps, volatility, time_period, oneOverRho, interest_rate, reop_rate = 0.01):
        self.num_steps = num_steps
        self.volatility = volatility
        self.time_period = time_period
        self.oneOverRho = oneOverRho
        self.interest = np.array(interest_rate) - reop_rate
        self.discount_factor = np.exp(-1 * self.interest * self.time_period)
        self.half_len_grid = self.num_steps * self.oneOverRho

        self.averagePriceTree = np.zeros(2 * self.num_steps * oneOverRho + 1)
        self.assetPriceTree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        self.optionPriceTree = np.zeros((self.num_steps + 1,  2 * self.num_steps * self.oneOverRho + 1))

    def forwardInduction(self, is_call):
        self.up_factor = np.exp(self.volatility * np.sqrt(self.time_period))

        for i in range(self.num_steps + 1):
            lower_bound = -i
            for j in range(i + 1):
                self.assetPriceTree[i, j] = self.init_price * (self.up_factor ** lower_bound)
                lower_bound += 2
        
        for j in range(2 * self.num_steps * self.oneOverRho + 1):
            jump = j - self.half_len_grid
            self.averagePriceTree[j] = self.init_price * (self.up_factor ** (jump / self.oneOverRho))
        for s in range(self.num_steps + 1):
            for k in range(2 * self.num_steps * self.oneOverRho + 1):
                if is_call:
                    self.optionPriceTree[s, k] = max(self.averagePriceTree[k] - self.strike, 0)
                    self.std_payoff = np.std(self.averagePriceTree -  - self.strike)
                else:
                    self.optionPriceTree[s, k] = max(self.strike - self.averagePriceTree[k], 0)
                    self.std_payoff = np.std(self.strike - self.averagePriceTree)

    def grid(self, n, k, j, plus): 
        numerator = np.zeros((len(j), len(k)))
        denominator = self.volatility * np.sqrt(self.time_period) / self.oneOverRho
        for jj in j:
            numerator[jj] = (n + 1) *  self.up_factor ** (k / self.oneOverRho) + self.up_factor ** (jj + plus)
            numerator[jj] = np.log(numerator[jj] / (n + 2))

        return numerator / denominator

    def backwardInduction(self):
        delta_y = self.volatility * np.sqrt(self.time_period)/ self.oneOverRho
        proba_up = (1 / self.discount_factor - 1 / self.up_factor) / (self.up_factor - 1 / self.up_factor)
#        print(np.round(self.optionPriceTree))
        for n in reversed(range(self.num_steps)):
            k_idx = np.array([k for k in range(- n * self.oneOverRho, n * self.oneOverRho + 1)])
            j_idx = np.array([j for j in range(n + 1)])
            k_up = self.grid(n, k_idx, j_idx, 1)
            k_down = self.grid(n, k_idx, j_idx, -1)
            
            j_idx_ext = np.repeat(j_idx[:, np.newaxis], len(k_idx), axis=1)

            k_up_floor = np.maximum(np.floor(k_up + self.half_len_grid).astype(int), 0)
            k_up_ceil = np.minimum(k_up_floor + 1, self.half_len_grid * 2)

            # average_price_up = ((n + 1) * self.averagePriceTree[n, k] + self.assetPriceTree[n + 1, i + 1]) / (n + 2)
            average_price_up = self.init_price * self.up_factor ** (k_up / self.oneOverRho)
            factor_interpolation_up = np.log(average_price_up / self.averagePriceTree[k_up_floor]) / delta_y

            option_price_up = factor_interpolation_up[0:] * self.optionPriceTree[j_idx_ext[0:], k_up_ceil[0:]] + \
                (1 - factor_interpolation_up[0:]) * self.optionPriceTree[j_idx_ext[0:], k_up_floor[0:]]
            
            k_down_floor = np.maximum(np.floor(k_down + self.half_len_grid).astype(int), 0)
            k_down_ceil = np.minimum(k_down_floor + 1, self.half_len_grid * 2)

            # average_price_down = ((n + 1) * self.averagePriceTree[n, k] + self.assetPriceTree[n + 1, i - 1]) / (n + 2)
            average_price_down = self.init_price * self.up_factor ** (k_down / self.oneOverRho)
            factor_interpolation_down = np.log(average_price_down / self.averagePriceTree[k_down_floor]) / delta_y

            # assert self.averagePriceTree[n + 1, k_down_floor] != 0

            option_price_down = factor_interpolation_down[:(n + 1)] * self.optionPriceTree[j_idx_ext[:(n + 1)], k_down_ceil[:(n + 1)]] + \
                (1 - factor_interpolation_down[:(n + 1)]) * self.optionPriceTree[j_idx_ext[:(n + 1)], k_down_floor[:(n + 1)]]

            self.optionPriceTree[j_idx_ext[:(n + 1)], k_idx + self.half_len_grid] = proba_up[n] * option_price_up + (1 - proba_up[n]) * option_price_down
            self.optionPriceTree[j_idx_ext[:(n + 1)], k_idx + self.half_len_grid] *= self.discount_factor[n]
#            print(np.round(self.optionPriceTree[j_idx_ext[:(n + 1)], k_idx + self.half_len_grid]))
            

    def getOptionPrice(self, init_price, strike, is_call=True):
        self.init_price = init_price
        self.strike = strike
        self.is_call =is_call
        self.forwardInduction(self.is_call)
        self.backwardInduction()
        return self.optionPriceTree[0, self.half_len_grid], self.std_payoff



