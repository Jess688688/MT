#!/bin/bash

set -e

echo "Running target_model.py..."
python target_model.py

echo "Running shadow_model.py..."
python shadow_result.py

echo "Running random_query.py..."
python random_query.py

echo "Running random_shadow_train.py..."
python random_shadow_train.py

echo "Running performing_defense.py..."
python performing_defense.py

echo "Running MIA/ShadowModelMIA.py..."
python MIA/ShadowModelMIA.py

echo "Running MIA/ClassMetricMIA.py..."
python MIA/ClassMetricMIA.py

echo "All scripts executed successfully!"