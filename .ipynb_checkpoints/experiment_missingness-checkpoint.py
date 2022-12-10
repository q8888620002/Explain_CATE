import os
import sys
import xgboost
import shap
import argparse
import pickle as pkl

import matplotlib.pyplot as plt

from scipy import stats
from utilities import *
from models import *
from sklearn import preprocessing
from shapreg import removal, games, shapley


#### import CATE model
module_path = os.path.abspath(os.path.join('CATENets/'))

if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.jax import TNet, SNet,SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet
import catenets.models as cate_models


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--d', help='feature dimension', required=True)
    parser.add_argument('-n','--n', help='sample size',required=True)
    parser.add_argument('-r','--r', help='random state',required=True)

    args = vars(parser.parse_args())
    
    n = int(args["n"])
    feature_size = int(args["d"])
    random_state = int(args["r"])
    
    #path = ("results_d=%s_n=%s_r=%s/"%(feature_size, n, random_state))
    
    #if not os.path.exists(path):
    #    os.makedirs(path)
    
    X, y_po, w, p, KL = simulation(feature_size, n, 0, 0, random_state, False)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    
    #print(KL)
    rng = np.random.default_rng(random_state)
    inds = np.arange(n)
    rng.shuffle(inds)
    
    n_train = int(0.8 * n)

    train_inds = inds[:n_train]
    test_inds = inds[n_train:]
    
    x_oracle_train = torch.from_numpy(X_scaled[train_inds,:])
    w_oracle_train = torch.from_numpy(w[train_inds,:])
    y_oracle_train = torch.from_numpy(np.take_along_axis(y_po,w, 1)[train_inds, :])
    y_test_cate = y_po[test_inds, 1] - y_po[test_inds, 0]
    
    ### Create Cate model 
    
    torch_DR = cate_models.torch.DRLearner(
                    2*feature_size,
                    binary_y=(len(np.unique(y_po)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=1000,
                    batch_size=512,
                    batch_norm=False,
                    nonlin="relu",
                    )
    
    maskes = generate_maskes(x_oracle_train)
    cate_model = Cate(torch_DR, MaskLayer())
    
    ### Train model with maskes
    
    cate_model.fit(x_oracle_train, maskes ,y_oracle_train, w_oracle_train)
    
    x_oracle_test = torch.from_numpy(X_scaled[test_inds,:])
    w_oracle_test = torch.from_numpy(w[test_inds,:])
    
    
    test = torch.cat([x_oracle_test, torch.ones( x_oracle_test.size()[0],feature_size)], dim=1)
    test_phe = cate_model.predict(test).cpu().detach().numpy()
    
    print("phe is %s" %mse(test_phe, y_test_cate))
    
    
    #### Explanation method
    
    # Make model callable
    model_lam = lambda x: cate_model.predict(x)

    # Model extension
    test_with_maskes = torch.cat([x_oracle_test, torch.ones(x_oracle_test.size()[0],feature_size)],dim=1)
    marginal_extension = removal.MarginalExtensionTorch(test_with_maskes, model_lam)
    
    instance = torch.from_numpy(X_scaled[0, :])
    instance = torch.cat([instance, torch.ones(feature_size)], dim=0)

    game = games.PredictionGame(marginal_extension, instance[:10])
    explanation = shapley.ShapleyRegression(game, batch_size=32)

    # Plot with 95% confidence intervals
    names = ["feature " + str(i) for i in range(feature_size)]
    explanation.plot(names, title='SHAP Values', sort_features=False)