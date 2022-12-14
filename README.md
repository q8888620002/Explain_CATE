# Explaning Conditional Average Treatment Effect 

Main script that generates synthetic dataset and test spearman correlation between Kernel SHAP and shapley-regression with masks.

- experiment_missingness.py

```python  experiment_missingness.py -n number_of_samples -d feature_dimension -r random_seed```

[CATENets](https://github.com/AliciaCurth/CATENets) is a repo contains Torch/Jax-based, sklearn-style implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators by Alicia Curth. 

Model modifications for explanation with mask are in ```CATENets/catenets/models/``` 

- pseudo_outcome_nets_mask.py (It contains abstract class for PseudoOutcomeLearner e.g. RA, DR, and PW-learner.)
- torch/base_mask.py (This script contains the actual model e.g. BasicNet, PropensityNet ,and RepresentationNet.)
- torch/utils/model_utlis.py
