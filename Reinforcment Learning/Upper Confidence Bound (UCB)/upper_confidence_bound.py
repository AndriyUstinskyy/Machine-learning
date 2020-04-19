
import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#algoritmo de UCB
number_of_selectiohs = [0] * d
sums_of_reward