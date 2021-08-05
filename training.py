import sys
sys.path.append("/home/muammar/software/production/ml4chem")
import json
import numpy as np
import itertools
from ase.io import Trajectory
import os

from ml4chem.data.handler import Data
from ml4chem.atomistic import Potentials
from ml4chem.atomistic.features import Gaussian
from ml4chem.atomistic.models.neuralnetwork import NeuralNetwork, train
from ml4chem.utils import logger
from ml4chem.visualization import parity
import pandas as pd
import ast


from dask.distributed import Client, LocalCluster

#FIX dask client

#distributed.comm.inproc - WARNING - Closing dangling queue in
#https://github.com/dask/distributed/issues/2507
client = Client(processes=False)


df = pd.read_csv('./data/miniset.csv')
uncertainty = df["uncertainty"].to_list()


training_uncertainty = {}
cases = range(5)
for i in cases:
    f = open(f"./data/3D_generation/{i}_data_split", "r").readlines()

    for index, line in enumerate(f):
        if "Training set" in line:
            training = ast.literal_eval(f[index + 1])
            training_uncertainty[i] = [uncertainty[u] for u in training]



uncertainty.index(training_uncertainty[0][-2])


cases = range(1, 5)
repeat = range(5)
case = "data/3D_generation/models.test.uncertainty"


print("curr dir", os.getcwd())
for i in cases:
    print("cases: ", i)
    if os.path.isdir(case) is False:
        os.mkdir(case)
    for j in repeat:
        print("repeat: ", j)
        d = f"{case}/_{i}"

        if os.path.isdir(d) is False:
            print(f"{d} does not exist")
            os.mkdir(d)
            
        batch_size = 10
        filename = f"{d}/{j}_training.db"

        if os.path.isfile(filename) == False:
            print("filename is false: ", filename)
            save_preprocessor = f"{d}/{j}_training.scaler"
            
            log_file = f"{d}/{j}_training.log"
            logger(filename=log_file)
            label = f"{d}/{j}_training"

            test = Trajectory(f"./data/3D_generation/{i}_test.traj")  
            images = Trajectory(f"./data/3D_generation/{i}_training_images.traj")

            # Arguments for fingerprinting the images
            normalized = True
            preprocessor = ('MinMaxScaler', {'feature_range': (0, 1)})

            # Arguments for building the model
            n = 40
            activation = 'relu'

            # Training images
            data_t = Data(images, purpose="training")
            training_set, t_targets = data_t.get_data(purpose="training")    

            t_features = Gaussian(cutoff=6.5, normalized=normalized,
                                  preprocessor=preprocessor, overwrite=False,
                                  save_preprocessor=save_preprocessor,
                                  filename=filename, batch_size=batch_size)
            print("past t_features")
            feature_space = t_features.calculate(training_set, data=data_t, purpose="training", svm=False)

            # Test images
            data_test = Data(test, purpose="training")
            print("past DATA()")
            test_set, test_targets = data_test.get_data(purpose="training")
    
            preprocessor = label + ".scaler"
            test_features = Gaussian(
                cutoff=6.5, 
                normalized=normalized,
                preprocessor=preprocessor, 
                overwrite=False
            )
    
            test_space = test_features.calculate(
                test_set, 
                data=data_test, 
                purpose="inference", 
                svm=False
            )
            print("past test")

            test = {
                "features": test_space, "targets": test_targets, "data": data_test
            }

            # Model
            nn = NeuralNetwork(hiddenlayers=(n, n), activation=activation)
            input_dimension = t_features.dimension

            nn.prepare_model(input_dimension, data=data_t)

            # Arguments for training the potential
            convergence = {'training': 80}
            epochs = 5000
            #lr = 1e-3
            lr = 1e-1
            weight_decay = 1e-6
            regularization = 0.
            optimizer = (
                'adam', 
                {'lr': lr, 'weight_decay': weight_decay, 'amsgrad': True}
            )

            _path = f"{d}/{j}_training"
            checkpoint = {"label": None, "checkpoint": 50, "path": _path}
    
            uncertainty = training_uncertainty[i]
            print("training")
            train(
                feature_space,
                targets=t_targets,
                model=nn,
                data=data_t,
                epochs=epochs, 
                regularization=regularization, 
                convergence=None,               
                device='cpu', 
                batch_size=batch_size, 
                optimizer=optimizer, 
                test=test,
                checkpoint=checkpoint,
                uncertainty=uncertainty
            )

            label = f"{d}/{j}_training"

            Potentials.save(model=nn, features=t_features, label=label)










for i in cases:
    f = open(f"./data/3D_generation/{i}_data_split", "r").readlines()
    data = Trajectory("./data/3D_generation/final_retention_metlin.traj")
    
    test_traj = Trajectory(f"./data/3D_generation/{i}_test.traj", mode="w")
    for index, line in enumerate(f):
        if "Test set" in line:
            test = ast.literal_eval(f[index + 1])
            for j in test:
                test_traj.write(data[j])
                
    test_traj.close()           


from ml4chem.visualization import read_log


init = 0
cases = range(5)
directory = "data/3D_generation/models.test.uncertainty/"

# Test set is orange curve
training_files = [f"{directory}_{case}/{i}_training.log" for case in cases for i in repeat]
print(training_files)

dfs = []
for f in training_files:
    print(f)
    dfs.append(read_log(f, metric='combined', data_only=True))
print(dfs[0].shape[0])
print(dfs[0].head())


import matplotlib.pyplot as plt

for index, model in enumerate(dfs):
    print(f"Model {index}")
    plt.plot(model.epochs, model.training, color="red")
    plt.plot(model.epochs, model.test, color="black")
    plt.show()
    plt.clf()
plt.legend(loc='best')

training = np.array([model.training for model in dfs])
training = np.mean(training, axis=0)
test = np.array([model.test for model in dfs])
test = np.mean(test, axis=0)

plt.plot(model.epochs, training, color="red", label="training error")
plt.plot(model.epochs, test, color="black", label="testing error")
plt.legend(loc='best')

gap = test - training
# plt.plot(model.epochs, gap, color="black")
print(gap[-1])

import plotly.graph_objects as go

model = 4

print(f"Model {model}")
df = dfs[model]

fig = go.Figure()

fig.add_trace(go.Scatter(x=df.epochs, y=df.training, mode='markers', name='training'))
fig.add_trace(go.Scatter(x=df.epochs, y=df.test, mode='markers', name='test'))
fig.update_traces(marker_size=4)
fig.show()


cases = range(5)
epoch = 4000
# best_models = {0: 600, 1: 350, 2: 1600, 3: 800, 4: 1600}
best_models = {0: epoch, 1: epoch, 2: epoch, 3: epoch, 4: epoch}

append = f"{epoch}_epochs"

for i in cases:
    for j in repeat:
        d = f"{case}/_{i}"
        filename = f"{case}/_{i}/{j}_test_{append}.log"

        if os.path.isfile(filename):
            os.remove(filename)

        logger(filename=filename)
        test_file = f"./data/3D_generation/{i}_test.traj"
        test = Trajectory(test_file)
        model = f"{case}/_{i}/{j}_training/checkpoint-{best_models[i]}.ml4c"

        params = f"{case}/_{i}/{j}_training.params"
        preprocessor = f"{case}/_{i}/{j}_training.scaler"

        calc = Potentials.load(
            model=model,
            params=params,
            preprocessor=preprocessor,
        )

        predictions = []
        true = []
        error = []

        for index, atoms in enumerate(test):
            rt_ml = calc.get_potential_energy(atoms)
            rt_ex = atoms.get_potential_energy()
            predictions.append(rt_ml)
            true.append(rt_ex)

        key_error_file = f"{case}/_{i}/{j}_testing_keyerrors_{append}.txt"
        f = open(key_error_file, "w")
        f.write("These molecules gave errors: \n {}".format(error))
        f.close()

        predictions_df = pd.DataFrame.from_dict(
            {"predictions": predictions, "exp": true}
        )
        df_file = f"{case}/_{i}/{j}_results_{append}.pkl"
        predictions_df.to_pickle(df_file)

        pred = predictions_df["predictions"].to_numpy()
        exp = predictions_df["exp"].to_numpy()

        absolute_error = abs((pred - exp) / exp)
        indices = np.where(absolute_error < 1)[0]
        parity_file = f"{case}/_{i}/{j}_parity_{append}.png"
        parity(
            np.take(pred, indices),
            np.take(exp, indices),
            scores=True,
            filename=parity_file,
        )



cases = range(5)
# best_models = {0: 1000, 1: 2000, 2: 1500, 3: 400, 4: 2000}
best_models = {0: 4000, 1: 4000, 2: 4000, 3: 4000, 4: 4000}

append = "4000_epochs"

results = {}
for i in cases:
    results[i] = {"ml": [], "true": []}
    
    for j in repeat:
        d = f"{case}/_{i}"

        images = Trajectory(f"./data/3D_generation/{i}_training_images.traj")
        model = f"{case}/_{i}/{j}_training/checkpoint-{best_models[i]}.ml4c"
        params = f"{case}/_{i}/{j}_training.params"
        preprocessor = f"{case}/_{i}/{j}_training.scaler"

        calc = Potentials.load(
            model=model,
            params=params,
            preprocessor=preprocessor,
        )
        
        predictions = []
        true = []

        for index, atoms in enumerate(images):
            rt_ml = calc.get_potential_energy(atoms)
            rt_ex = atoms.get_potential_energy()
            print(rt_ml, rt_ex)
            asdad
            predictions.append(rt_ml)
            true.append(rt_ex)
            
        results[i]["ml"].append(predictions)
        results[i]["true"].append(true)


results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_pickle("4000_epochs_results_uncertainty.pkl")
results_df = pd.read_pickle("4000_epochs_results_uncertainty.pkl")
results_df.iloc[0].ml