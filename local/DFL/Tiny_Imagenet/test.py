import DFL_shadow_result as dsr
import DFL_target_model as dtm
import performing_defense as pd
import random_shadow_train as rst
import random_target_train as rtt
import MIA.ShadowModelMIA as smm
import MIA.ClassMetricMIA as cmm
import numpy as np

if __name__ == "__main__":
    dsr.generate_DFL_shadow_result(num_participants = 10, num_rounds = 15, epochs_per_round = 10)
    dtm.generate_DFL_target_model(num_participants = 10, num_rounds = 15, epochs_per_round = 10)
    rst.perform_random_shadow_train(50000)

    for alpha in np.arange(0, 1.1, 0.1):
        pd.perform_defense([0, 1], [0.5, 0.5], alpha)
        rtt.perform_random_target_train(50000)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for alpha = {alpha}")
        print("11111111111111111111111111111111111111111111111")
        print("11111111111111111111111111111111111111111111111")

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
        pd.perform_defense(num, weights, 0.60)
        rtt.perform_random_target_train(50000)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for num = {num}, weights = {weights}")
        print("2222222222222222222222222222222222222222222222222")
        print("2222222222222222222222222222222222222222222222222")