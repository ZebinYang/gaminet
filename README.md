#  GAMI-Net
Generalized additive models with structured interactions

## Installation 

The following environments are required:

- Python 3.7 + (anaconda is preferable)
- tensorflow>=2.0.0
- numpy>=1.15.2
- pandas>=0.19.2
- matplotlib>=3.1.3
- scikit-learn>=0.23.0

```shell
pip install gaminet
```

## Usage

Import library
```python
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from gaminet import GAMINet
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import feature_importance_visualize
from gaminet.utils import plot_trajectory
from gaminet.utils import plot_regularization
```

Load data 
```python
def metric_wrapper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def rmse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.sqrt(np.mean((pred - label)**2))

def data_generator1(datanum, dist="uniform", random_state=0):
    
    nfeatures = 100
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, [datanum, nfeatures])
    x1, x2, x3, x4, x5, x6 = [x[:, [i]] for i in range(6)]

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
        + cliff(x5, x6)).reshape([-1,1]) + 1 * np.random.normal(0, 1, [datanum, 1])

    task_type = "Regression"
    meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(nfeatures)}
    meta_info.update({'Y':{'type':'target'}})         
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            meta_info[key]['scaler'] = sy
        else:
            sx = MinMaxScaler((0, 1))
            sx.fit([[0], [1]])
            x[:,[i]] = sx.transform(x[:,[i]])
            meta_info[key]['scaler'] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=random_state)
    return train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse, sy)

train_x, test_x, train_y, test_y, task_type, meta_info, get_metric = data_generator1(10000, 0)
```

Run GAMI-Net
```python
## Note the current GAMINet API requires input features being normalized within 0 to 1. 
model = GAMINet(meta_info=meta_info, interact_num=20, 
                interact_arch=[40] * 5, subnet_arch=[40] * 5, 
                batch_size=200, task_type=task_type, activation_func=tf.nn.relu, 
                main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500, 
                lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                heredity=True, loss_threshold=0.01, reg_clarity=1,
                mono_increasing_list=[], mono_decreasing_list=[], ## the indices list of features
                verbose=False, val_ratio=0.2, random_state=random_state)

model.fit(train_x, train_y)

val_x = train_x[model.val_idx, :]
val_y = train_y[model.val_idx, :]
tr_x = train_x[model.tr_idx, :]
tr_y = train_y[model.tr_idx, :]
pred_train = model.predict(tr_x)
pred_val = model.predict(val_x)
pred_test = model.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(tr_y, pred_train),5), 
                      np.round(get_metric(val_y, pred_val),5),
                      np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)
```

Training Logs
```python 
simu_dir = "./results/"
if not os.path.exists(simu_dir):
    os.makedirs(simu_dir)

data_dict_logs = model.summary_logs(save_dict=False)
plot_trajectory(data_dict_logs, folder=simu_dir, name="s1_traj_plot", log_scale=True, save_png=True)
plot_regularization(data_dict_logs, folder=simu_dir, name="s1_regu_plot", log_scale=True, save_png=True)
```
![traj_visu_demo](https://github.com/ZebinYang/gaminet/blob/master/examples/results/s1_traj_plot.png)
![regu_visu_demo](https://github.com/ZebinYang/gaminet/blob/master/examples/results/s1_regu_plot.png)

Global Visualization
```python 
data_dict = model.global_explain(save_dict=False)
global_visualize_density(data_dict, save_png=True, folder=simu_dir, name='s1_global')
```
![global_visu_demo](https://github.com/ZebinYang/gaminet/blob/master/examples/results/s1_global.png)

Feature Importance
```python 
feature_importance_visualize(data_dict, save_png=True, folder=simu_dir, name='s1_feature')
```
<img src="https://github.com/ZebinYang/gaminet/blob/master/examples/results/s1_feature.png" width="480">

Local Visualization
```python 
data_dict_local = model.local_explain(train_x[:10], train_y[:10], save_dict=False)
local_visualize(data_dict_local[0], save_png=True, folder=simu_dir, name='s1_local')
```
<img src="https://github.com/ZebinYang/gaminet/blob/master/examples/results/s1_local.png" width="480">

## Citations
----------
Yang, Z., Zhang, A. and Sudjianto, A., 2020. GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions. [arXiv:2003.0713](https://arxiv.org/abs/2003.07132)

```latex
@article{yang2020gami,
  title={GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions},
  author={Yang, Zebin and Zhang, Aijun and Sudjianto, Agus},
  journal={arXiv preprint arXiv:2003.07132},
  year={2020}
}
```
