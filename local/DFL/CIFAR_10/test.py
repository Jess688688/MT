import DFL_shadow_result as dsr
import DFL_target_model as dtm
import random_target_train as rtt
import random_shadow_train as rst
import performing_defense as pd
import random_target_train as rtt
import MIA.ShadowModelMIA as smm
import MIA.ClassMetricMIA as cmm
import numpy as np
import optimal_defense_intensity as odi

if __name__ == "__main__":

    dsr.generate_DFL_shadow_result(10, 15, 10)
    dtm.generate_DFL_target_model(10, 15, 10)
    rst.perform_random_shadow_train(5000)
    
    best_result = odi.generate_optimal_defense_intensity()
    best_num, best_weights, best_alpha, diff, metrics = best_result
    print("\n>>> Best overall config (with minimum diff):")
    print(f"num={best_num}, weights={best_weights}, alpha={best_alpha:.2f}, diff={diff:.4f}, metrics={metrics}")

    for alpha in np.arange(0, 1.1, 0.1):
        pd.perform_defense(best_num, best_weights, alpha)
        rtt.perform_random_target_train(5000)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for num={best_num}, weights={best_weights}, alpha = {alpha}")
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
        pd.perform_defense(num, weights, best_alpha)
        rtt.perform_random_target_train(5000)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for num = {num}, weights = {weights}, alpha = {best_alpha}")
        print("2222222222222222222222222222222222222222222222222")
        print("2222222222222222222222222222222222222222222222222")