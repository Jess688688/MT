# Prerequisite:
* Ubuntu 20.04
* Python 3.8

# Script Introduction:
* one_to_two.py: Split the image dataset evenly by class into two subsets, which will later be used for training the target model and the shadow model respectively.
* PCA.py: Perform Principal Component Analysis on the image dataset by class and generate a representative image for each class based on the first principal component.
* target_model.py: Train a classification model on an image dataset and save the trained model for future use.
* shadow_result_model.py: Train shadow models and compute the predicted labels for each training sample and test sample after training.
* random_query.py: Randomly select a subset of training samples from the target dataset with the same size to the test samples.
* random_target_train.py: Randomly select a subset of target model training dataset prediction results with the same size as that of test dataset.
* random_shadow_train.py: Randomly select a subset of shadow model training dataset prediction results with the same size as that of test dataset.
* performing_defense.py: Apply data augmentation and fusion with PCA composite images to the query images as a defense method against MIA.
* ShadowModelMIA.py: Perform shadow model based MIA.
* ClassMetricMIA.py: Perform metric based MIA.
* optimal_defense_intensity.py: Automatically determines the defense parameters, including data augmentation intensity and PCA composite image fusion weight, to find an appropriate defense intensity for each image dataset.
* test.py: Control the execution of other scripts and its content reflects the sequence in which other codes are run.

# code execution order
run `one_to_two.py` and `PCA.py` first, then run `test.py`.
