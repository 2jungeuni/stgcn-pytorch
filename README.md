# stgcn-pytorch

This repository contains an implementation of the Spatio-Temporal Graph Convolutional Network (STGCN) model in **PyTorch**.
It is adapted from the [official STGCN code](https://github.com/VeritasYin/STGCN_IJCAI-18), which is originally based on **TensorFlow**.

---
## Installation
1. Clone this repository:
```bash
git clone git@github.com:2jungeuni/stgcn-pytorch.git
```
2. Adjust the parameters in the `config/cfg.py` according to your needs. For instance, set `n_pred: 3` to predict **15 minutes** or set `n_pred: 6` to predict **30 minutes**. Adjust other hyperparameters such as learning rate, batch size, and epochs as needed.
3. To run the training process:
```bash
python3 main.py
```
4. To run the testing process:
```bash
python3 tester.py
```
**Note**: Make sure to load the best-performing weights into the model. Pay attention to the weight path and configure it properly before running the model. 

---
## Results
<div align="center">
<table>
  <tr>
    <!-- (1,1) merged with (2,1) by rowspan="2" -->
    <td rowspan="2"></td>
    <td align="center">45 mins prediction</td>
    <td align="center">30 mins prediction</td>
    <td align="center">15 mins prediction</td>
  </tr>
  <tr>
    <!-- First column is merged above, so only three cells here -->
    <td align="center">MAE/MAPE/RMSE</td>
    <td align="center">MAE/MAPE/RMSE</td>
    <td align="center">MAE/MAPE/RMSE</td>
  </tr>
  <tr>
    <td align="center">Original performance</td>
    <td align="center">3.57 / 8.69 / 6.77</td>
    <td align="center">3.03 / 7.33 / 5.70</td>
    <td align="center">2.25 / 5.26 / 4.04</td>
  </tr>
  <tr>
    <td align="center">Our performance</td>
    <td align="center">3.07 / 7.30 / 5.54</td>
    <td align="center">3.51 / 8.36 / 6.56</td>
    <td align="center">3.86 / 9.55 / 7.44</td>
  </tr>
</table>
</div>

Comparing **MAE loss** and **copy loss**, the copy acts as a baseline that indicates the error incurred if the model simply **copies the output from the previous time step** rather than predicting a new value.
In many sequence-to-sequence tasks, this naive copying strategy can greatly reduce the overall loss because it does not attempt to forecast future changes. It just reuses what was already observed.
Therefore, we compare the model's MAE loss to the copy loss to see whether the model is actually predicting future values rather than just repeating past data.
<div align="center">
    <img src=plot/pemsd7-m/pemsd7m-mae-loss_45.svg width="33%"><img src=plot/pemsd7-m/pemsd7m-mae-loss_30.svg width="33%"><img src=plot/pemsd7-m/pemsd7m-mae-loss_15.svg width="33%">
</div>

Here are the test results.

<div align="center">
<img src=plot/pemsd7-m/test_45.svg width="33%"><img src=plot/pemsd7-m/test_30.svg width="33%"><img src=plot/pemsd7-m/test_15.svg width="33%">
</div>