"""
Calculating feature ranking for explanation methods
on clinical datasets, inlcuding ist3, responder, and massive transfusion
"""
import argparse
import collections
import os
import pickle
import random

import numpy as np
import pandas as pd
from scipy import stats

import src.CATENets.catenets.models as cate_models
from dataset import Dataset
from src.cate_utils import qini_score, qini_score_cal
from src.CATENets.catenets.models.torch import pseudo_outcome_nets
from src.interpretability.explain import Explainer
from src.model_utils import NuisanceFunctions
from src.utils import ablate, attribution_ranking, insertion_deletion

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-d", "--dataset", help="Dataset", required=True)
    parser.add_argument(
        "-s", "--shuffle", help="shuffle", default=True, action="store_false"
    )
    parser.add_argument(
        "-t", "--num_trials", help="number of runs ", required=True, type=int
    )
    parser.add_argument(
        "-n",
        "--top_n_features",
        help="how many features to extract",
        required=True,
        type=int,
    )
    parser.add_argument("-l", "--learner", help="learner", required=True)
    parser.add_argument(
        "-b",
        "--zero_baseline",
        help="whether to use zero_baseline",
        default=True,
        action="store_false",
    )

    parser.add_argument("-device", "--device", help="device", required=True)

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = args["num_trials"]
    top_n_features = args["top_n_features"]
    shuffle = args["shuffle"]
    learner = args["learner"]
    DEVICE = args["device"]
    zero_baseline = args["zero_baseline"]
    print(zero_baseline)
    print("shuffle dataset: ", shuffle)

    explainer_limit = 1000

    selection_types = ["if_pehe", "pseudo_outcome_r", "pseudo_outcome_dr"]

    data = Dataset(cohort_name, 10)
    names = data.get_feature_names()

    x_train, _, _ = data.get_data("train")
    x_test, _, _ = data.get_data("test")

    feature_size = x_train.shape[1]

    explainers = [
        "saliency",
        "smooth_grad",
        # "gradient_shap",
        # "lime",
        # "baseline_lime",
        "baseline_shapley_value_sampling",
        "marginal_shapley_value_sampling",
        "integrated_gradients",
        "baseline_integrated_gradients",
        # "kernel_shap"
        # "marginal_shap"
    ]

    top_n_results = {e: [] for e in explainers}

    result_sign = {e: np.zeros((trials, feature_size)) for e in explainers}

    results_train = np.zeros((trials, len(x_train)))
    results_test = np.zeros((trials, len(x_test)))

    for i in range(trials):

        data = Dataset(cohort_name, i, shuffle)

        x_train, w_train, y_train = data.get_data("train")
        x_val, w_val, y_val = data.get_data("val")
        x_test, w_test, y_test = data.get_data("test")

        models = {
            "XLearner": pseudo_outcome_nets.XLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
                seed=i,
            ),
            "SLearner": cate_models.torch.SLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
                seed=i,
            ),
            "RLearner": pseudo_outcome_nets.RLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                n_iter=1000,
                lr=1e-3,
                patience=10,
                batch_size=128,
                batch_norm=False,
                nonlin="relu",
                device=DEVICE,
                seed=i,
            ),
            "RALearner": pseudo_outcome_nets.RALearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                n_iter=1000,
                lr=1e-3,
                patience=10,
                batch_size=128,
                batch_norm=False,
                nonlin="relu",
                device=DEVICE,
                seed=i,
            ),
            "TLearner": cate_models.torch.TLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
            "TARNet": cate_models.torch.TARNet(
                x_train.shape[1],
                binary_y=True,
                n_layers_r=1,
                n_layers_out=1,
                n_units_out=100,
                n_units_r=100,
                batch_size=128,
                n_iter=1000,
                batch_norm=False,
                early_stopping=True,
                nonlin="relu",
            ),
            "CFRNet_0.01": cate_models.torch.TARNet(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_r=2,
                n_layers_out=2,
                n_units_out=100,
                n_units_r=100,
                batch_size=128,
                n_iter=1000,
                lr=1e-3,
                batch_norm=False,
                nonlin="relu",
                penalty_disc=0.01,
            ),
            "CFRNet_0.001": cate_models.torch.TARNet(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_r=2,
                n_layers_out=2,
                n_units_out=100,
                n_units_r=100,
                lr=1e-5,
                batch_size=128,
                n_iter=1000,
                batch_norm=False,
                nonlin="relu",
                penalty_disc=0.001,
                seed=i,
            ),
            "DRLearner": pseudo_outcome_nets.DRLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
        }

        learner_explanations = {}
        learner_explainers = {}
        insertion_deletion_data = []

        # Training nuisance function for pehe.

        if data.cohort_name in ["crash_2", "ist3", "sprint", "accord"]:
            nuisance_functions = NuisanceFunctions(rct=True)
        else:
            nuisance_functions = NuisanceFunctions(rct=False)

        nuisance_functions.fit(x_val, y_val, w_val)

        model = models[learner]

        # Create a matrix of zeros with the same shape as x_train

        noise_matrix = np.zeros(x_train.shape)
        baseline = np.mean(x_train, axis=0)

        # for _, idx_lst in data.discrete_indices.items():
        #     if len(idx_lst) == 1:

        #         # setting binary vars to 0.5

        #         baseline[idx_lst] = 0.5
        #     else:
        #         # setting categorical baseline to 1/n
        #         # category_counts = data[:, idx_lst].sum(axis=0)
        #         # baseline[idx_lst] = category_counts / category_counts.sum()

        #         baseline[idx_lst] = 1 / len(idx_lst)

        model.fit(x_train, y_train, w_train)

        results_train[i] = model.predict(X=x_train).detach().cpu().numpy().flatten()
        results_test[i] = model.predict(X=x_test).detach().cpu().numpy().flatten()

        print(f"Explaining dataset with: {learner}")

        # Explain CATE
        learner_explainers[learner] = Explainer(
            model,
            feature_names=list(range(x_train.shape[1])),
            explainer_list=explainers,
            perturbations_per_eval=1,
            baseline=baseline.reshape(1, -1),
        )

        learner_explanations[learner] = learner_explainers[learner].explain(x_test)

        # Calculate IF-PEHE for insertion and deletion for each explanation methods

        for explainer_name in explainers:

            train_score_results = []
            test_score_results = []
            mse_results = []

            rand_test_results = []
            rand_train_results = []
            rand_mse_results = []

            # obtaining global & local ranking for insertion & deletion

            abs_explanation = np.abs(learner_explanations[learner][explainer_name])

            local_rank = attribution_ranking(
                learner_explanations[learner][explainer_name]
            )
            global_rank = np.flip(np.argsort(abs_explanation.mean(0)))

            if zero_baseline:
                baseline = np.zeros(baseline.shape)

            print("Calculating insertion/deletion and ablation results. ")
            insertion_results, deletion_results = insertion_deletion(
                data.get_data("test"),
                local_rank,
                model,
                baseline,
                selection_types,
                nuisance_functions,
            )

            ablation_pos_results = ablate(
                data.get_data("test"),
                learner_explanations[learner][explainer_name],
                model,
                baseline,
                "pos",
                nuisance_functions,
            )

            ablation_neg_results = ablate(
                data.get_data("test"),
                learner_explanations[learner][explainer_name],
                model,
                baseline,
                "neg",
                nuisance_functions,
            )

            for feature_idx in range(1, feature_size + 1):

                print(
                    "obtaining subgroup results for %s, feature_num: %s."
                    % (explainer_name, feature_idx),
                    end="\r",
                )

                # Starting from 1 features
                train_score, test_score, mse = qini_score(
                    global_rank[:feature_idx],
                    (x_train, w_train, y_train),
                    (x_test, w_test, y_test),
                    model,
                    learner,
                )
                train_score_results.append(train_score)
                test_score_results.append(test_score)
                mse_results.append(mse)

                random.seed(i)

                rand_train_score, rand_test_score, rand_mse = qini_score(
                    random.sample(range(0, x_train.shape[1]), feature_idx),
                    (x_train, w_train, y_train),
                    (x_test, w_test, y_test),
                    model,
                    learner,
                )

                rand_train_results.append(rand_train_score)
                rand_test_results.append(rand_test_score)
                rand_mse_results.append(rand_mse)

            insertion_deletion_data.append(
                [
                    learner,
                    explainer_name,
                    insertion_results,
                    deletion_results,
                    train_score_results,
                    test_score_results,
                    mse_results,
                    rand_test_score,
                    rand_train_results,
                    rand_mse_results,
                    [qini_score_cal(w_train, y_train, results_train[i])],
                    [qini_score_cal(w_test, y_test, results_test[i])],
                    ablation_pos_results,
                    ablation_neg_results,
                ]
            )

        with open(
            os.path.join(
                f"results/{cohort_name}/",
                f"insertion_deletion_shuffle_{shuffle}_{learner}_zero_baseline_{zero_baseline}_seed_{i}.pkl",
            ),
            "wb",
        ) as output_file:
            pickle.dump(insertion_deletion_data, output_file)

        # Getting top n features

        for explainer_name in explainers:

            ind = np.argpartition(
                np.abs(learner_explanations[learner][explainer_name]).mean(0),
                -top_n_features,
            )[-top_n_features:]

            top_n_results[explainer_name].extend(names[ind].tolist())

            for col in range(feature_size):
                result_sign[explainer_name][i, col] = stats.pearsonr(
                    x_test[:, col],
                    learner_explanations[learner][explainer_name][:, col],
                )[0]

    for explainer_name in explainers:

        results = collections.Counter(top_n_results[explainer_name])
        summary = pd.DataFrame(
            results.items(), columns=["feature", "count (%)"]
        ).sort_values(by="count (%)", ascending=False)

        summary["count (%)"] = np.round(summary["count (%)"] / (trials), 2) * 100

        indices = [names.tolist().index(i) for i in summary.feature.tolist()]
        summary["sign"] = np.sign(np.mean(result_sign[explainer_name], axis=0)[indices])

        summary.to_csv(
            f"results/{cohort_name}/"
            f"{explainer_name}_top_{top_n_features}_features_shuffle_{shuffle}_{learner}.csv"
        )

    with open(
        os.path.join(
            f"results/{cohort_name}/",
            f"train_shuffle_{shuffle}_{learner}_zero_baseline_{zero_baseline}.pkl",
        ),
        "wb",
    ) as output_file:
        pickle.dump(results_train, output_file)

    with open(
        os.path.join(
            f"results/{cohort_name}",
            f"/test_shuffle_{shuffle}_{learner}_zero_baseline_{zero_baseline}.pkl",
        ),
        "wb",
    ) as output_file:
        pickle.dump(results_test, output_file)
