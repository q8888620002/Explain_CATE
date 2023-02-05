import sys
import os
import torch
import pickle as pkl
from scipy import stats

from pathlib import Path

#import catenets.models as cate_models

#### import CATE model
module_path = os.path.abspath(os.path.join('./CATENets/'))

if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.jax import TNet, SNet,SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet
import catenets.models as cate_models
import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets


import numpy as np
import pandas as pd

import src.interpretability.logger as log
from src.interpretability.explain import Explainer
from src.interpretability.datasets.data_loader import load
from src.interpretability.synthetic_simulate import (
    SyntheticSimulatorLinear,
    SyntheticSimulatorModulatedNonLinear,
)
from src.interpretability.utils import (
    attribution_accuracy,
    compute_pehe,
    attribution_ranking,
    attribution_insertion_deletion
)


class PredictiveSensitivity:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 3000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5,  1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")


        explainability_data = []
        insertion_deletion_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                   # "TLearner": cate_models.torch.TLearner(
                   # X_train.shape[1],
                   # device = "cuda:1",
                   # binary_y=(len(np.unique(Y_train)) == 2),
                   # n_layers_out=2,
                   # n_units_out=100,
                   # batch_size=1024,
                   # n_iter=self.n_iter,
                   # batch_norm=False,
                   # nonlin="relu",
               # ),
               # "SLearner": cate_models.torch.SLearner(
               #     X_train.shape[1],
               #     device = "cuda:1",
               #     binary_y=(len(np.unique(Y_train)) == 2),
               #     n_layers_out=2,
               #     n_units_out=100,
               #     n_iter=self.n_iter,
               #     batch_size=1024,
               #     batch_norm=False,
               #     nonlin="relu",
               # ),
              #  "TARNet": cate_models.torch.TARNet(
              #      X_train.shape[1],
              #      device = "cuda:1",
              #      binary_y=(len(np.unique(Y_train)) == 2),
              #      n_layers_r=1,
              #      n_layers_out=1,
              #      n_units_out=100,
              #      n_units_r=100,
              #      batch_size=1024,
              #      n_iter=self.n_iter,
              #      batch_norm=False,
              #      nonlin="relu",
              #  ),
                 "XLearnerMask": cate_models_masks.XLearnerMask(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=10,
                     batch_size=self.batch_size,
                    lr=1e-3,
                    patience=10,
                     batch_norm=False,
                     nonlin="relu",
                     device="cuda:1"
                 ),
                "XLearner": cate_models.torch.XLearner(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=self.n_iter,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     device="cuda:1"
                 ),
                 "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     device="cuda:1"
                     ),
                 "DRLearner": cate_models.torch.DRLearner(
                     X_train.shape[1],
                     device = "cuda:0",
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=1000,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu"
                 ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for name in learners:                    
                if not  "mask" in learner_name.lower():
                    learner_explaintion_lists[name] = [           
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap"
                    ]
                else:
                    learner_explaintion_lists[name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[name],
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[:self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score  = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features, acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    # Computing insertion & deletion results

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                        pate_model_name = learner_name + "Mask"
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)
                        pate_model_name = learner_name

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    # Obtain feature importance rank
                    rank_indices = attribution_ranking(learner_explanations[learner_name][explainer_name])

                    # Using PATE to predict CATE
                    insertion_results, deletion_results = attribution_insertion_deletion(
                            X_test[:self.explainer_limit, :],
                            rank_indices,
                            learners[pate_model_name]
                    )
                    
                    insertion_deletion_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            insertion_results,
                            deletion_results,
                            rank_indices
                        ]
                    )

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score ",
                "Pred features ACC",
                "Pred features ACC Score ",
                "Prog features ACC",
                "Prog features ACC Score ",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = self.save_path / "results/predictive_sensitivity/insertion_deletion"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )
        
        results_path = self.save_path / "results/predictive_sensitivity/insertion_deletion/insertion_deletion"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(insertion_deletion_data , handle)


class PredictiveSensitivityHeldOutOne:
    """
    Held out one analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1500,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5,  1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")


        explainability_data = []
        held_out_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )
                
            log.info("Fitting and explaining learners...")


            learners = {
                   # "TLearner": cate_models.torch.TLearner(
                   # X_train.shape[1],
                   # device = "cuda:1",
                   # binary_y=(len(np.unique(Y_train)) == 2),
                   # n_layers_out=2,
                   # n_units_out=100,
                   # batch_size=1024,
                   # n_iter=self.n_iter,
                   # batch_norm=False,
                   # nonlin="relu",
               # ),
               # "SLearner": cate_models.torch.SLearner(
               #     X_train.shape[1],
               #     device = "cuda:1",
               #     binary_y=(len(np.unique(Y_train)) == 2),
               #     n_layers_out=2,
               #     n_units_out=100,
               #     n_iter=self.n_iter,
               #     batch_size=1024,
               #     batch_norm=False,
               #     nonlin="relu",
               # ),
              #  "TARNet": cate_models.torch.TARNet(
              #      X_train.shape[1],
              #      device = "cuda:1",
              #      binary_y=(len(np.unique(Y_train)) == 2),
              #      n_layers_r=1,
              #      n_layers_out=1,
              #      n_units_out=100,
              #      n_units_r=100,
              #      batch_size=1024,
              #      n_iter=self.n_iter,
              #      batch_norm=False,
              #      nonlin="relu",
              #  ),
                #  "XLearnerMask": cate_models_masks.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #      lr=1e-3,
                #      patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": cate_models.torch.XLearner(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                  "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                      X_train.shape[1],
                      binary_y=(len(np.unique(Y_train)) == 2),
                      n_layers_out=2,
                      n_units_out=100,
                      n_iter=self.n_iter,
                      batch_size=self.batch_size,
                      batch_norm=False,
                      lr=1e-3,
                      patience=10,
                      nonlin="relu",
                      device="cuda:1"
                      ),
                  "DRLearner": cate_models.torch.DRLearner(
                      X_train.shape[1],
                      device = "cuda:0",
                      binary_y=(len(np.unique(Y_train)) == 2),
                      n_layers_out=2,
                      n_units_out=100,
                      n_iter=self.n_iter,
                      lr=1e-3,
                      patience=10,
                      batch_size=self.batch_size,
                      batch_norm=False,
                      nonlin="relu"
                  ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for name in learners:                    
                if not  "mask" in name.lower():
                    learner_explaintion_lists[name] = [           
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap"
                    ]
                else:
                    learner_explaintion_lists[name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)

                # Obtaining explanation 

                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[name],
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[:self.explainer_limit]
                )

            cate_test = sim.te(X_test)


            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=torch.ones((X_test.shape)))

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            # Held out experiment - iterating x_s = D \ {i} for all i in features. 

            for feature_index in range(X_train.shape[1]):
                
                masks = np.ones((X_train.shape[1]))
                masks[feature_index] = 0

                X_train_subset = X_train[:, masks.astype(bool)]
                X_test_subset = X_test[:, masks.astype(bool)]

                pate_learners = {
                    "DRLearner": pseudo_outcome_nets.DRLearnerPate(
                        X_train.shape[1],
                        X_train_subset.shape[1],
                        device="cuda:1",
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=self.n_iter,
                        batch_size=self.batch_size,
                        batch_norm=False,
                        lr=1e-3,
                        patience=10,
                        nonlin="relu"
                 )
                }

                subset_explainer_list = [
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap",
                    "explain_with_missingness"
                ]
                
                for learner_name in pate_learners:
                    log.info("Fitting PATE models"+ learner_name)
                    
                    pate_learners[learner_name].fit(
                        X=X_train,
                        X_subset=X_train_subset,
                        y=Y_train, 
                        w=W_train
                    )

                    cate_pred = learners[learner_name].predict(X=X_test[:self.explainer_limit]).detach().cpu().numpy()
                    pate_pred = pate_learners[learner_name].predict(X=X_test_subset[:self.explainer_limit ]).detach().cpu().numpy()

                    delta_tau = (cate_pred - pate_pred).reshape(-1)

                    # loading attribution score from learner_explanations

                    for explainer_name in subset_explainer_list:
                        if explainer_name == "explain_with_missingness":
                            learner_name += "Mask"

                        attribution = learner_explanations[learner_name][explainer_name][:, feature_index]

                        held_out_data.append(
                            [
                                predictive_scale,
                                feature_index,
                                learner_name,
                                explainer_name, 
                                attribution,
                                delta_tau
                            ]
                        )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = self.save_path / "results/predictive_sensitivity/held_out/model_performance"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )
        
        results_path = self.save_path / "results/predictive_sensitivity/held_out/"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(held_out_data , handle)

class NonLinearitySensitivity:
    """
    Sensitivity analysis for nonlinearity in prognostic and predictive functions.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 3000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        nonlinearity_scales: list = [0.0, 0.2, 0.5, 0.7, 1.0],
        predictive_scale: float = 1,
        synthetic_simulator_type: str = "random",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.nonlinearity_scales = nonlinearity_scales
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap"
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )
        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        explainability_data = []
        insertion_deletion_data = []

        for nonlinearity_scale in self.nonlinearity_scales:
            log.info(f"Now working with a nonlinearity scale {nonlinearity_scale}...")
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type=self.synthetic_simulator_type,
            )
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )


            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                 "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=10,
                     batch_size=self.batch_size,
                    lr=1e-3,
                    patience=10,
                     batch_norm=False,
                     nonlin="relu",
                     device="cuda:1"
                 ),
                "XLearner": cate_models.torch.XLearner(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=self.n_iter,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     device="cuda:1"
                 ),
                 "DRLearnerMask": cate_models_masks.DRLearnerMask(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     device="cuda:1"
                     ),
                 "DRLearner": cate_models.torch.DRLearner(
                     X_train.shape[1],
                     device = "cuda:0",
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=1000,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu"
                 ),
            }

            learner_explainers = {}
            learner_explaintion_lists = {}
            learner_explanations = {}

            for name in learners:                    
                if not  "mask" in learner_name.lower():
                    learner_explaintion_lists[name] = [           
                                                        "integrated_gradients",
                                                        "shapley_value_sampling",
                                                        "naive_shap"
                                                    ]
                else:
                    learner_explaintion_lists[name] = ["explain_with_missingness"]

                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[name],
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[:self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score  = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features, acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    ### computing insertion/deletion results

                    if not  "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                        pate_model_name = learner_name + "Mask"
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)
                        pate_model_name = learner_name

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    # Obtain feature importance rank
                    rank_indices = attribution_ranking(learner_explanations[learner_name][explainer_name])

                    # Using PATE to predict CATE
                    insertion_results, deletion_results = attribution_insertion_deletion(
                            X_test[:self.explainer_limit, :],
                            rank_indices,
                            learners[pate_model_name]
                    )

                    insertion_deletion_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            insertion_results,
                            deletion_results,
                            rank_indices
                        ]
                    )

                    explainability_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Nonlinearity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score ",
                "Pred features ACC",
                "Pred features ACC Score ",
                "Prog features ACC",
                "Prog features ACC Score ",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = (
            self.save_path
            / f"results/nonlinearity_sensitivity/insertion_deletion/{self.synthetic_simulator_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.csv"
        )

        results_path = self.save_path / "results/nonlinearity_sensitivity/insertion_deletion/insertion_deletion"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(insertion_deletion_data , handle)

class PropensitySensitivity:
    """
    Sensitivity analysis for confounding.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 10000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        propensity_type: str = "pred",
        propensity_scales: list = [0, 0.5, 1, 2, 5, 10],
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.propensity_type = propensity_type
        self.propensity_scales = propensity_scales

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        predictive_scale: float = 1,
        nonlinearity_scale: float = 0.5,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features} and predictive scale {predictive_scale}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type="random",
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []

        for propensity_scale in self.propensity_scales:
            log.info(f"Now working with propensity_scale = {propensity_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                "DRLearner": cate_models.torch.DRLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    lr=1e-3,
                    patience=10,
                    nonlin="relu",
                    device= "cuda:1"
                ),
                "XLearner": cate_models.torch.XLearner(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=self.n_iter,
                    lr=1e-3,
                    patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu",
                     device="cuda:1"
                 ),
                # "CFRNet_0.01": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.01,
                # ),
                # "CFRNet_0.001": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.001,
                # ),
                # "CFRNet_0.0001": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.0001,
                # ),
            }

            learner_explainers = {}
            learner_explanations = {}

            for name in learners:
                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)                    

                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=explainer_list,
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[: self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features,acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    cate_pred = learners[learner_name].predict(X=X_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            propensity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Propensity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score",
                "Pred features ACC",
                "Pred features ACC Score",
                "Prog features ACC",
                "Prog features ACC Score",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / f"results/propensity_sensitivity/sample/{self.synthetic_simulator_type}/{self.propensity_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )
