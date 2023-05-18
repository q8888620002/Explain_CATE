
import argparse
import os
import sys
import pickle
import numpy as np

from utilities import calculate_pehe, Dataset

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.torch import pseudo_outcome_nets
import catenets.models as cate_models
import catenets.models.torch.tlearner as tlearner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--dataset', help='Dataset', required=True)
    parser.add_argument('-t','--num_trials', help='Dataset', required=True)

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = int(args["num_trials"])
    DEVICE = "cuda:1"

    data = Dataset(cohort_name)
    x_test, _, _ = data.get_testing_data()

    learners = [
        "XLearner",
        "DRLearner",
        "SLearner",
        "TLearner"
    ]

    selection_types = [
        "if_pehe",
        "pseudo_outcome_r",
        "pseudo_outcome_dr"
    ]

    results = {
         learner: {
             **{sec: np.zeros((trials)) for sec in selection_types}, 
             "prediction": np.zeros((trials, x_test.shape[0]))
            } for learner in learners         
    }

    for i in range(trials):

        np.random.seed(i)
        data = Dataset(cohort_name, i)

        x_train, w_train, y_train = data.get_training_data()
        x_test, w_test, y_test = data.get_testing_data()

        learners = {
            "XLearner":pseudo_outcome_nets.XLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1500,
                        nonlin="relu",
                        device=DEVICE
                    ),
            "DRLearner": pseudo_outcome_nets.DRLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1500,
                        nonlin="relu",
                        device=DEVICE
                ),
            "SLearner": cate_models.torch.SLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1500,
                        nonlin="relu",
                        device=DEVICE
                ),
            "TLearner": cate_models.torch.TLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1500,
                        nonlin="relu",
                        device=DEVICE
                )
        }

        for learner_name, cate_model in learners.items():

            cate_model.fit(x_train, y_train, w_train)

            prediction = cate_model.predict(x_test).detach().cpu().numpy()
            prediction = prediction.flatten()

            results[learner_name]["prediction"][i] = prediction

            for sec in selection_types:
                results[learner_name][sec][i] = calculate_pehe(
                    prediction,
                    data,
                    sec
                )

    with open(f"results/{cohort_name}/model_selection.pkl", "wb") as output_file:
        pickle.dump(results, output_file)

