import numpy as np

class asianOptionBinomialTree:

    def __init__(self, num_steps, volatility, time_period, oneOverRho, interest_rate):
        self.num_steps = num_steps
        self.volatility = volatility
        self.time_period = time_period
        self.oneOverRho = oneOverRho
        self.discount_factor = np.exp(-1 * interest_rate * self.time_period)
        self.half_len_grid = self.num_steps * self.oneOverRho

        self.averagePriceTree = np.zeros(2 * self.num_steps * oneOverRho + 1)
        self.assetPriceTree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        self.optionPriceTree = np.zeros((self.num_steps + 1,  2 * self.num_steps * self.oneOverRho + 1))

    def forwardInduction(self):
        self.up_factor = np.exp(self.volatility * np.sqrt(self.time_period))

        for i in range(self.num_steps + 1):
            lower_bound = -i
            for j in range(i + 1):
                self.assetPriceTree[i, j] = self.init_price * (self.up_factor ** lower_bound)
                lower_bound += 2
        
        for j in range(2 * self.num_steps * self.oneOverRho + 1):
            jump = j - self.num_steps * self.oneOverRho
            self.averagePriceTree[j] = self.init_price * (self.up_factor ** (jump / self.oneOverRho))

        for s in range(self.num_steps + 1):
            for k in range(2 * self.num_steps * self.oneOverRho + 1):
                self.optionPriceTree[s, k] = max(self.averagePriceTree[k] - self.strike, 0)
                # self.optionPriceTree[s, k] = max(self.averagePriceTree[k] - self.assetPriceTree[self.num_steps - s, s], 0)

    def grid(self, n, k, j, plus): 
        denominator = self.volatility * np.sqrt(self.time_period) / self.oneOverRho
        numerator = (n + 1) *  self.up_factor ** (k / self.oneOverRho) + self.up_factor ** (j + plus)
        numerator = np.log(numerator / (n + 2))

        return numerator / denominator

    def backwardInduction(self):
        delta_y = self.volatility * np.sqrt(self.time_period)/ self.oneOverRho
        proba_up = (1 / self.discount_factor - 1 / self.up_factor) / (self.up_factor - 1 / self.up_factor)

        for n in reversed(range(self.num_steps)):
            lower_bound = -n
            i = np.array([j for j in range(n + 1)]) + 1
            k_idx = [k for k in range(- n * self.oneOverRho, n * self.oneOverRho + 1)]
            k_idx = np.array(k_idx)
            k_up = self.grid(n, k_idx, lower_bound, 1)
            k_down = self.grid(n, k_idx, lower_bound, -1)

            k_up_floor = np.maximum(np.floor(k_up + self.half_len_grid).astype(int), 0)
            k_up_ceil = np.minimum(k_up_floor + 1, self.half_len_grid * 2)

            # average_price_up = ((n + 1) * self.averagePriceTree[n, k] + self.assetPriceTree[n + 1, i + 1]) / (n + 2)
            average_price_up = self.init_price * self.up_factor ** (k_up / self.oneOverRho)
            factor_interpolation_up = np.log(average_price_up / self.averagePriceTree[k_up_floor]) / delta_y

            option_price_up = factor_interpolation_up * self.optionPriceTree[1:(n + 2), k_up_ceil] + \
                (1 - factor_interpolation_up) * self.optionPriceTree[1:(n + 2), k_up_floor]
            
            k_down_floor = np.maximum(np.floor(k_down + self.half_len_grid).astype(int), 0)
            k_down_ceil = np.minimum(k_down_floor + 1, self.half_len_grid * 2)

            # average_price_down = ((n + 1) * self.averagePriceTree[n, k] + self.assetPriceTree[n + 1, i - 1]) / (n + 2)
            average_price_down = self.init_price * self.up_factor ** (k_down / self.oneOverRho)
            factor_interpolation_down = np.log(average_price_down / self.averagePriceTree[k_down_floor]) / delta_y

            # assert self.averagePriceTree[n + 1, k_down_floor] != 0

            option_price_down = factor_interpolation_down * self.optionPriceTree[0:(n + 1), k_down_ceil] + \
                (1 - factor_interpolation_down) * self.optionPriceTree[0:(n + 1), k_down_floor]

            self.optionPriceTree[0:(n + 1), k_idx + self.half_len_grid] = proba_up * option_price_up + (1 - proba_up) * option_price_down
            self.optionPriceTree[0:(n + 1), k_idx + self.half_len_grid] *= self.discount_factor
                
            lower_bound -= 2
            

    def getOptionPrice(self, init_price, strike):
        self.init_price = init_price
        self.strike = strike
        self.forwardInduction()
        self.backwardInduction()
        return self.optionPriceTree[0, self.half_len_grid]


def test_main():
    test_tree = averagePriceBinomialTree(63, 0.0075, 1 / 63, 3, 0.03)
    option_price = test_tree.getOptionPrice(2753, 2680)
    print(option_price)


if __name__ == "__main__":
    test_main()


