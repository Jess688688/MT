import performing_defense as pd
import random_target_train as rtt
import MIA.ShadowModelMIA as smm
import MIA.ClassMetricMIA as cmm
import numpy as np

if __name__ == "__main__":
    for alpha in np.arange(0, 1.1, 0.1):
        pd.perform_defense([0, 1], [0, 1], alpha)
        rtt.perform_random_target_train(2500)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for alpha = {alpha}")

    defense_params = [
    ([0, 1], [1, 0]),
    ([0, 1], [0.5, 0.5]),
    ([0, 1], [0, 1]),
    ([1, 2], [0.5, 0.5]),
    ([1, 2], [0, 1]),
    ([2, 3], [0.5, 0.5]),
    ([2, 3], [0, 1]),
    ([3, 4], [0.5, 0.5]),
    ([3, 4], [0, 1]),
    ]

    for num, weights in defense_params:
        pd.perform_defense(num, weights, 0.50)
        rtt.perform_random_target_train(2500)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for num = {num}, weights = {weights}")