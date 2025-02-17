
import argparse
import os
import sys
import collections
import pickle as pkl

import numpy as np
import pandas as pd

from scipy import stats

from utilities import *

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.torch import pseudo_outcome_nets
from catenets.models.torch.base import BasicNet
import src.interpretability.logger as log
from src.interpretability.utils import attribution_ranking
from src.interpretability.explain import Explainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--dataset', help='Dataset', required=True)
    parser.add_argument('-n','--top_n_features', help='Dataset', required=True)

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    top_n_features = int(args["top_n_features"])
    DEVICE = "cuda:3"

    explainers = [
        "saliency",
        "integrated_gradients",
        "baseline_shapley_value_sampling",
        "marginal_shapley_value_sampling"
        # "kernel_shap"
        # "marginal_shap"
    ]

    top_n_results = {
        e:[] for e in explainers
    }

    selection_types = [
        "if_pehe",
        "pseudo_outcome_r",
        "pseudo_outcome_dr"
    ]

    results_train = pkl.load(open(f"results/{cohort_name}/train_xlearner.pkl", "rb"))
    results_test = pkl.load(open(f"results/{cohort_name}/test_xlearner.pkl", "rb"))

    data = Dataset(cohort_name, 42)
    names = data.get_feature_names()

    x_train, w_train, y_train = data.get_data("train")
    x_val, w_val, y_val = data.get_data("val")
    x_test, w_test, y_test = data.get_data("test")
    
    feature_size = x_train.shape[1]

    x_replacement = data.get_replacement_value()

    y_train = torch.from_numpy(np.mean(results_train, axis=0)).float()
    x_train = torch.from_numpy(x_train).float()

    insertion_deletion_data = []
    learner_explanations = {}

    result_sign = {
        e:np.zeros((1, feature_size)) for e in explainers
    }

    tau_star = pseudo_outcome_nets.XLearner(
        x_train.shape[1],
        binary_y=(len(np.unique(y_train)) == 2),
        n_layers_out=2,
        n_units_out=100,
        batch_size=128,
        n_iter=1000,
        nonlin="relu",
        device=DEVICE,
    )

    ensemble = BasicNet(
        "EnsembleNet",
        n_unit_in = x_train.shape[1],
        binary_y=False,
        n_layers_out=2,
        n_units_out=100,
        batch_size=128,
        n_iter=1500,
        nonlin="relu",
        device=DEVICE,
        prob_diff=True
    )

    # pseudo-ground truth. 
    if data.cohort_name == "crash_2" or data.cohort_name =="ist3":
        nuisance_functions = NuisanceFunctions(rct=True)
    else:
        nuisance_functions = NuisanceFunctions(rct=False)

    nuisance_functions.fit(x_val, y_val, w_val)
    
    tau_star.fit(x_train, y_train, w_train)

    y_hat = tau_star.predict(x_train).detach().cpu().numpy()

    # training student model with knowledge distilation

    ensemble.fit_knowledge_distillation(x_train, y_train, y_hat)
    
    # Explain CATE

    baseline = np.mean(x_train, axis=0)

    ## Setting the one-hot variables within the same group with the same baseline/replacement value.

    for k, v in data.categorical_indices.items():
        baseline[v] = 1/len(v)

    explainer = Explainer(
        ensemble,
        feature_names=list(range(x_train.shape[1])),
        explainer_list=explainers,
        baseline = baseline.reshape(1, -1)
    )

    log.info(f"Explaining EnsembleNet")

    learner_explanations = explainer.explain(
        x_test
    )


    for explainer_name in explainers:

        local_rank = attribution_ranking(learner_explanations[explainer_name])

        top_n_indices = np.argpartition(
            np.abs(
                learner_explanations[explainer_name]
            ).mean(0),
            -top_n_features
        )[-top_n_features:]

        insertion_results, deletion_results = insertion_deletion(
            data.get_data("test"),
            local_rank,
            ensemble,
            np.zeros(baseline).reshape(1,-1),
            selection_types,
            nuisance_functions
        )

        insertion_deletion_data.append(
            [
                "ensemble",
                explainer_name,
                insertion_results,
                deletion_results,
            ]
        )
        top_n_results[explainer_name].extend(names[top_n_indices].tolist())

        for col in range(feature_size):
            result_sign[explainer_name][0, col] = stats.pearsonr(
                x_test[:,col], learner_explanations[explainer_name][:, col]
            )[0]

        results = collections.Counter(top_n_results[explainer_name])
        summary = pd.DataFrame(
                results.items(),
                columns=[
                    'feature',
                    'count (%)'
                ]
            ).sort_values(
            by="count (%)",
            ascending=False
        )

        summary["count (%)"] = np.round(summary["count (%)"]/1, 2)*100

        indices = [names.tolist().index(i) for i in summary.feature.tolist()]
        summary["sign"] = np.sign(np.mean(result_sign[explainer_name], axis=0)[indices])

        summary.to_csv(
            f"results/{cohort_name}/"
            f"{explainer_name}_top_{top_n_features}_features_ensemble.csv"
        )

    with open(
        f"results/{cohort_name}/"
        f"insertion_deletion_ensemble.pkl", "wb") as output_file:
        pkl.dump(insertion_deletion_data, output_file)