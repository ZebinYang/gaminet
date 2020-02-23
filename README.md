#  Generalized additive model with pairwise interactions (GAMI-Net)

## Installation 

### Prerequisite

The following environments are required:

- Python 3.7 (anaconda is preferable)
- tensorflow 2.0


### Github Installation

You can install the package by the following console command:

```shell
pip install git+https://github.com/zebinyang/gaminet.git
```

### Manual Installation

If git is not available, you can manually install the package by downloading the source codes and then compiling it by hand:

- Download the source codes from https://github.com/ZebinYang/gaminet.git.

- unzip and switch to the root folder.

- Run the following shell commands to finish installation.

```shell
pip install -r requirements.txt
python setup.py install
```


## Usage

Import library
```python
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from gaminet import GAMINet
from gaminet.utils import feature_importance
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import global_visualize_wo_density
```

Load data 
```python
def metric_wrappper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def rmse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.sqrt(np.mean((pred - label)**2))

def data_generator1(datanum, testnum=10000, noise_sigma=1, rand_seed=0):
    
    np.random.seed(rand_seed)
    x = np.zeros((datanum + testnum, 10))
    for i in range(10):
        x[:, i:i+1] = np.random.uniform(0, 1,[datanum + testnum,1])
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [x[:, [i]] for i in range(10)]

    def cliff(x1, x2):
        # x1: -20,20
        # x2: -10,5
        x1 = (2 * x1 - 1) * 20
        x2 = (2 * x2 - 1) * 7.5 - 2.5
        term1 = -0.5 * x1 ** 2 / 100
        term2 = -0.5 * (x2 + 0.03 * x1 ** 2 - 3) ** 2
        y = 10 * np.exp(term1 + term2)
        return  y

    y = (8 * (x1 - 0.5) ** 2
         + 0.1 * np.exp(-8 * x2 + 4)
         + 3 * np.sin(2 * np.pi * x3 * x4)
         + cliff(x5, x6)).reshape([-1,1]) + noise_sigma*np.random.normal(0, 1, [datanum + testnum, 1])

    task_type = "Regression"
    meta_info = {"X1":{"type":"continuous"},
             "X2":{"type":"continuous"},
             "X3":{"type":"continuous"},
             "X4":{"type":"continuous"},
             "X5":{"type":"continuous"},
             "X6":{"type":"continuous"},
             "X7":{"type":"continuous"},
             "X8":{"type":"continuous"},
             "X9":{"type":"continuous"},
             "X10":{"type":"continuous"},
             "Y":{"type":"target"}}
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        else:
            sx = MinMaxScaler((0, 1))
            sx.fit([[0], [1]])
            x[:,[i]] = sx.transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=testnum, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrappper(rmse, sy)

train_x, test_x, train_y, test_y, task_type, meta_info, get_metric = data_generator1(datanum=10000, testnum=10000, noise_sigma=1, rand_seed=0)
```

Run GAMI-Net
```python
gaminet = GAMINet(meta_info=meta_info, interact_num=10, interact_arch=[20, 10],
               subnet_arch=[10, 6], task_type=task_type, activation_func=tf.tanh, main_grid_size=101, interact_grid_size=51,
               batch_size=min(500, int(0.2*train_x.shape[0])),
               lr_bp=0.001, main_threshold=0.05, total_threshold=0.05,
               init_training_epochs=2000, interact_training_epochs=2000, tuning_epochs=10,
               verbose=True, val_ratio=0.2, early_stop_thres=10, random_state=0)
gaminet.fit(train_x, train_y)

val_x = train_x[gaminet.val_idx, :]
val_y = train_y[gaminet.val_idx, :]
tr_x = train_x[gaminet.tr_idx, :]
tr_y = train_y[gaminet.tr_idx, :]
pred_train = gaminet.predict(tr_x)
pred_val = gaminet.predict(val_x)
pred_test = gaminet.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(tr_y, pred_train),5), 
                      np.round(get_metric(val_y, pred_val),5),
                      np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)
```

Visualization
```python 
data_dict = gaminet.global_explain(save_dict=False, folder=simu_dir, name='demo_gaminet_simu1_global')
global_visualize_wo_density(data_dict)
```
 ![global_visu_demo](https://github.com/ZebinYang/seqmml/blob/master/examples/s1_model.png)


References
----------
Yang, Zebin, Aijun Zhang, and Agus Sudjianto. "GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions." 2020.
