from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
# import shap

from shapreg import shapley, games, removal, shapley_sampling
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import (
    DeepLift,
    FeatureAblation,
    FeaturePermutation,
    IntegratedGradients,
    NoiseTunnel,
    KernelShap,
    Lime,
    ShapleyValueSampling,
    GradientShap,
    Saliency
)

from captum.attr._core.lime import get_exp_kernel_similarity_function
from torch import nn

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Explainer:
    """
    Explainer instance, consisting of several explainability methods.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List,
        explainer_list: List = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "deeplift",
            "shapley_value_sampling",
            "lime",
        ],
        n_steps: int = 500,
        perturbations_per_eval: int = 10,
        n_samples: int = 1000,
        kernel_width: float = 1.0,
        baseline: Optional[torch.Tensor] = None,
    ) -> None:

        self.device =  model.device
        self.baseline = baseline
        self.explainer_list = explainer_list
        self.feature_names = feature_names

        # Feature ablation
        feature_ablation_model = FeatureAblation(model)

        def feature_ablation_cbk(x_test: torch.Tensor) -> torch.Tensor:
            out = feature_ablation_model.attribute(
                x_test, n_steps=n_steps, perturbations_per_eval=perturbations_per_eval
            )

            return out

        # Integrated gradients
        integrated_gradients_model = IntegratedGradients(model)

        def integrated_gradients_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return integrated_gradients_model.attribute(
                x_test,
                n_steps=n_steps,
            )

        def baseline_integrated_gradients_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return integrated_gradients_model.attribute(
                x_test,
                n_steps=n_steps,
                baselines = self.baseline
            )

        def smooth_grad_cpk(x_test: torch.Tensor) -> torch.Tensor:

            noise_tunnel = NoiseTunnel(integrated_gradients_model)

            return noise_tunnel.attribute(
                x_test,
                nt_type='smoothgrad_sq'
            )


        # DeepLift
        deeplift_model = DeepLift(model)

        def deeplift_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return deeplift_model.attribute(x_test)

        # Feature permutation
        feature_permutation_model = FeaturePermutation(model)

        def feature_permutation_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return feature_permutation_model.attribute(
                x_test, n_steps=n_steps, perturbations_per_eval=perturbations_per_eval
            )

        # LIME
        exp_eucl_distance = get_exp_kernel_similarity_function(
            kernel_width=kernel_width
        )

        lime_model = Lime(
            model,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=exp_eucl_distance,
        )

        def lime_cbk(x_test: torch.Tensor) -> torch.Tensor:

            test_values = torch.zeros((x_test.size()))

            for test_ind in range(len(x_test)):

                lime_value = lime_model.attribute(
                    x_test[test_ind].view(1, -1),
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                )

                test_values[test_ind] = lime_value.detach().cpu()

            return self._check_tensor(test_values)

        def baseline_lime_cbk(x_test: torch.Tensor) -> torch.Tensor:

            test_values = torch.zeros((x_test.size()))

            for test_ind in range(len(x_test)):

                lime_value = lime_model.attribute(
                    x_test[test_ind].view(1, -1),
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                    baselines=self.baseline
                )

                test_values[test_ind] = lime_value.detach().cpu()

            return self._check_tensor(test_values)

        # Baseline shapley value sampling
        shapley_value_sampling_model = ShapleyValueSampling(model)

        def baseline_shapley_value_sampling_cbk(x_test: torch.Tensor) -> torch.Tensor:

            return shapley_value_sampling_model.attribute(
                x_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=True
            )

        # Marginal shapley value sampling

        def marginal_shapley_value_sampling_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return shapley_value_sampling_model.attribute(
                x_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
                baselines = self.baseline,
                show_progress=True
            )


        # Kernel SHAP
        kernel_shap_model = KernelShap(model)

        def kernel_shap_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return kernel_shap_model.attribute(
                x_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
            )

        # Gradient SHAP
        gradient_shap_model = GradientShap(model)

        def gradient_shap_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return gradient_shap_model.attribute(x_test, baselines=self.baseline)

        saliency_model = Saliency(model)

        def saliency_cpk(x_test: torch.tensor) -> torch.Tensor:
            return saliency_model.attribute(x_test)

        # Explain with missingness
        def explain_with_missingness_cbk(x_test:torch.Tensor) -> torch.Tensor:

            test_values = np.zeros((x_test.size()))

            for test_ind in range(len(x_test)):
                instance = x_test[test_ind, :][None, :]
                game  = games.CateGame(instance, model)
                explanation = shapley_sampling.ShapleySampling(game, batch_size=128)
                test_values[test_ind] = explanation.values

            return self._check_tensor(test_values)

        def marginal_shap_cbk(x_test: torch.Tensor) -> torch.Tensor:

            test_values = np.zeros((x_test.size()))
            x_test  = x_test.detach().cpu().numpy()
            baseline = self.baseline.detach().cpu().numpy()

            marginal_extension = removal.MarginalExtension(baseline, model)

            for test_ind in range(len(x_test)):
                instance = x_test[test_ind]
                game = games.PredictionGame(marginal_extension, instance)
                explanation = shapley_sampling.ShapleySampling(game, thresh = 0.01, batch_size=128)
                test_values[test_ind] = explanation.values.reshape(-1, x_test.shape[1])

            return self._check_tensor(test_values)


        self.explainers = {
            "feature_ablation": feature_ablation_cbk,
            "integrated_gradients": integrated_gradients_cbk,
            "baseline_integrated_gradients": baseline_integrated_gradients_cbk,
            "smooth_grad": smooth_grad_cpk,
            "deeplift": deeplift_cbk,
            "feature_permutation": feature_permutation_cbk,
            "lime": lime_cbk,
            "baseline_lime": baseline_lime_cbk,
            "baseline_shapley_value_sampling": baseline_shapley_value_sampling_cbk,
            "marginal_shapley_value_sampling": marginal_shapley_value_sampling_cbk,
            "kernel_shap": kernel_shap_cbk,
            "gradient_shap": gradient_shap_cbk,
            "explain_with_missingness":explain_with_missingness_cbk,
            "marginal_shap": marginal_shap_cbk,
            "saliency": saliency_cpk
        }

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(self.device)

    def explain(self, X: torch.Tensor) -> Dict:
        output = {}
        if self.baseline is None:
            self.baseline = torch.zeros(
                X.shape
            )  # Zero tensor as baseline if no baseline specified
        for name in self.explainer_list:
            x_test = self._check_tensor(X)
            self.baseline = self._check_tensor(self.baseline)
            x_test.requires_grad_()
            explainer = self.explainers[name]
            output[name] = explainer(x_test).detach().cpu().numpy()
        return output


    def plot(self, X: torch.Tensor) -> None:
        explanations = self.explain(X)

        fig, axs = plt.subplots(int((len(explanations) + 1) / 2), 2)

        idx = 0
        for name in explanations:
            x_pos = np.arange(len(self.feature_names))

            ax = axs[int(idx / 2), idx % 2]

            ax.bar(x_pos, np.mean(np.abs(explanations[name]), axis=0), align="center")
            ax.set_xlabel("Features")
            ax.set_title(f"{name}")

            idx += 1
        plt.tight_layout()
