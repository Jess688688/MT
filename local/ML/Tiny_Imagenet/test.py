import target_model as tm
import shadow_result as sr
import random_query as rq
import random_shadow_train as rst
import performing_defense as pd
import MIA.ShadowModelMIA as smm
import MIA.ClassMetricMIA as cmm
import numpy as np

if __name__ == "__main__":
    tm.generate_target_model()
    sr.generate_shadow_result(10, 10000, 1000)
    rq.perform_random_query(5000)
    rst.perform_random_shadow_train(10000)
    
    for alpha in np.arange(0, 1.1, 0.1):
        pd.perform_defense([0, 1], [0, 1], alpha)
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
        pd.perform_defense(num, weights, 0.80)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for num = {num}, weights = {weights}")
