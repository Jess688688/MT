import target_model as tm
import shadow_result as sr
import random_query as rq
import random_shadow_train as rst
import performing_defense as pd
import MIA.ShadowModelMIA as smm
import MIA.ClassMetricMIA as cmm
import numpy as np
import optimal_defense_intensity as odi

if __name__ == "__main__":
    tm.generate_target_model()
    # There are two configurations for shadow training in ML pipeline, another parameter setting is sr.generate_shadow_result(1, 25000, 5000)
    sr.generate_shadow_result(10, 5000, 1000)
    rq.perform_random_query(5000)
    # For another configuration, parameter setting is 5000
    rst.perform_random_shadow_train(10000)
    
    # Finding suitable defense intensity for this dataset
    best_result = odi.generate_optimal_defense_intensity()
    best_num, best_weights, best_alpha, diff, metrics = best_result
    print("\n>>> Best overall config (with minimum diff):")
    print(f"num={best_num}, weights={best_weights}, alpha={best_alpha:.2f}, diff={diff:.4f}, metrics={metrics}")

    # study the impact of PCA composite image fusion weight on defense effectiveness
    for alpha in np.arange(0, 1.1, 0.1):
        pd.perform_defense(best_num, best_weights, alpha)
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for num={best_num}, weights={best_weights}, alpha = {alpha}")

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
        smm.perform_shadow_model_mia()
        cmm.perform_class_metric_mia()
        print(f"Running for num = {num}, weights = {weights}, alpha = {best_alpha}")
