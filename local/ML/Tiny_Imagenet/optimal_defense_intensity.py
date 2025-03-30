# This code assumes that without defense, F1-score is larger than 0.5
# If final overall_valid_configs is empty, then relax the condition in function all_in_range to 0.38 <= m <= 0.62 or 0.35 <= m <= 0.65;
# or change the value of 'step' in the 'next_weight_step' function from 0.1 to 0.05;
# or Make the alpha values more fine-grained and dense by changing the increment from 0.1 to 0.05.

import target_model as tm
import shadow_result as sr
import random_query as rq
import random_shadow_train as rst
import performing_defense as pd
import MIA.ShadowModelMIA as smm
import MIA.ClassMetricMIA as cmm
import numpy as np
from datetime import datetime

def get_metrics():
    return list(smm.perform_shadow_model_mia()) + list(cmm.perform_class_metric_mia())

def all_in_range(metrics):
    return all(0.4 <= m <= 0.6 for m in metrics)

def next_weight_step(num, weights, direct):
    step = 0.1
    w1, w2 = weights[0], weights[1]

    # direct=0: weaken data augmentation intensity
    if direct == 0:
        if num[0] == 0 and w1 == 1:
            return None
        if w1 == 1.0:
            new_num = [num[0] - 1, num[1] - 1]
            if new_num[0] < 0:
                return None
            w1_new = round(step, 2)
            w2_new = round(1.0 - w1_new, 2)
            return new_num, [w1_new, w2_new]
        else:
            w1_new = round(w1 + step, 2)
            w2_new = round(1.0 - w1_new, 2)
            return num, [w1_new, w2_new]

    # direct=1: enhance data augmentation intensity
    else:
        if w2 == 1.0:
            new_num = [num[0] + 1, num[1] + 1]
            return new_num, [1.0 - step, step]
        else:
            w2_new = round(w2 + step, 2)
            w1_new = round(1.0 - w2_new, 2)
            return num, [w1_new, w2_new]

def generate_optimal_defense_intensity():
    
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
    
    print("\nStarted at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    overall_valid_configs = []
    # best_configs = []
    
    for alpha in np.arange(0.5, 1.1, 0.1): 
        selected_param = None
        selected_metrics = None
        best_diff = float('inf')
        best_num = [0, 1]
        best_weights = [1, 0]
        best_metrics = None
        valid_configs = []
        
        for i in range(len(defense_params)):
            num, weights = defense_params[i]
            pd.perform_defense(num, weights, alpha)
            metrics = get_metrics()

            diff = abs(metrics[2] - 0.5)
            if all_in_range(metrics):
                valid_configs.append((num, weights, alpha, diff, metrics))
                print(f"num={num}, weights={weights}, alpha={alpha:.2f}, diff={diff:.4f}, metrics={metrics}")

            if diff < best_diff:
                best_diff = diff
                best_num = num
                best_weights = weights
                best_metrics = metrics

        if best_metrics[2] < 0.5:
            direct = 0
        else:
            direct = 1

        cur_direct = direct

        test_num, test_weights = best_num, best_weights
        
        while cur_direct == direct:
            result = next_weight_step(test_num, test_weights, cur_direct)
            
            if result is None:
                break

            test_num, test_weights = result

            pd.perform_defense(test_num, test_weights, alpha)
            metrics = get_metrics()
            diff = abs(metrics[2] - 0.5)

            if all_in_range(metrics):
                valid_configs.append((test_num, test_weights, alpha, diff, metrics))
                print(f"num={test_num}, weights={test_weights}, alpha={alpha:.2f}, diff={diff:.4f}, metrics={metrics}")
            
            if diff < best_diff:
                best_diff = diff
                best_num = test_num
                best_weights = test_weights
                best_metrics = metrics

            if metrics[2] < 0.5:
                cur_direct = 0
            else:
                cur_direct = 1
                
        overall_valid_configs.extend(valid_configs)
    
    unique_results = []
    seen = set()

    for cfg in overall_valid_configs:
        num, weights, alpha, diff, metrics = cfg
        key = (tuple(num), tuple(weights), round(alpha, 2))
        if key not in seen:
            unique_results.append(cfg)
            seen.add(key)

    for cfg in unique_results:
        num, weights, alpha, diff, metrics = cfg
        print(f"num={num}, weights={weights}, alpha={alpha:.2f}, diff={diff:.4f}, metrics={metrics}")

    best_result = min(
        unique_results,
        key=lambda x: sum(abs(x[4][i] - 0.5) for i in [2, 5, 8, 11])
    )
    best_result = list(best_result)
        
    num, weights, alpha, diff, metrics = best_result
    print("\n>>> Best overall config (with minimum diff):")
    print(f"best num={num}, best weights={weights}, best alpha={alpha:.2f}, diff={diff:.4f}, metrics={metrics}")
    print("\nFinished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return best_result

if __name__ == "__main__":
    generate_optimal_defense_intensity()

