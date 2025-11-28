# Deep Learning Regression – Atelier 1 (PyTorch GPU Version)

This repository contains the implementation of a deep learning regression model applied to the NYSE stock market dataset.  
The project includes exploratory data analysis, neural network development, hyperparameter optimization, and regularization techniques.  
All computations were performed on a CUDA-enabled GPU for performance.

Dataset source: https://www.kaggle.com/datasets/dgawlik/nyse

---

## 1. Project Overview

This workshop focuses on using deep learning methods to predict stock closing prices.  
The objectives are:

1. Perform exploratory data analysis (EDA) on the NYSE dataset.  
2. Build a regression model using a fully connected neural network in PyTorch.  
3. Apply GridSearch to identify optimal hyperparameters.  
4. Visualize training and testing loss curves.  
5. Apply regularization techniques and compare results.

---

## 2. Dataset Description

The dataset used is `prices.csv`, which contains over 850,000 rows of daily stock information for multiple companies.  
Important columns include:

- date  
- symbol  
- open  
- high  
- low  
- close  
- volume  

The data is time-series oriented and covers several years of trading activity.

---

## 3. Exploratory Data Analysis

The following analyses were performed:

- Time-series visualization of Apple (AAPL) closing prices.  
- Distribution analysis of closing prices using histograms and kernel density estimation.  
- Correlation heatmap between numerical features.  
- Average trading volume per year.  

These steps help understand the dataset and validate the relevance of chosen features.

---

## 4. Data Preprocessing

Preparation steps:

1. Features selected: open, high, low, volume  
2. Target variable: close  
3. Normalization using StandardScaler  
4. Train-test split (80% training, 20% testing)  
5. Conversion to PyTorch tensors and GPU upload  
6. Creation of DataLoader for batch processing  

---

## 5. Model Architecture

The base neural network is a multilayer perceptron defined as follows:

- Input size: 4  
- Hidden layers: [128, 64, 32]  
- Activation: ReLU  
- Dropout applied after each hidden layer  
- Output size: 1 (regression target)  

Loss function: MSELoss  
Optimizer: Adam  

---

## 6. Training Process

A custom training function was developed to:

- Perform forward and backward propagation  
- Update weights with mini-batch training  
- Compute training and testing loss at each epoch  
- Utilize GPU acceleration for faster computation  

---

## 7. Hyperparameter Optimization (GridSearch)

A manual GridSearch was implemented using sklearn’s ParameterGrid.

Hyperparameters tested:

- Layer sizes: [128, 64, 32], [256, 128, 64], [64, 32]  
- Learning rates: 0.001, 0.0005, 0.01  
- Dropout rates: 0.2, 0.3, 0.4  

Total combinations tested: 27

Best model obtained:

- Layers: [256, 128, 64]  
- Dropout: 0.2  
- Learning rate: 0.001  
- Validation MSE: approximately 7.54  

---

## 8. Final Model Training and Evaluation

The best model was retrained for 150 epochs using weight decay for improved convergence.

Performance metrics:

- RMSE: approximately 6.07  
- R²: approximately 0.9947  

The loss curves show:

- Stable train loss  
- Acceptable fluctuations in test loss  
- No significant overfitting  

---

## 9. Regularization Techniques

A second model was trained using:

- Batch Normalization  
- Dropout of 0.5  
- Weight decay (L2)  

Comparison results:

- The regularized model showed unstable training and increased test loss.  
- The base model performed better on this dataset.  

---

## 10. Conclusion

This workshop demonstrates how to:

- Explore a large financial dataset  
- Build and train a neural network on GPU  
- Use hyperparameter tuning to improve performance  
- Evaluate models using RMSE and R²  
- Understand the impact of regularization  

The final model achieved strong predictive performance with R² close to 0.995.

---

## 11. Requirements

- Python 3.x  
- PyTorch (with CUDA support)  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-Learn  

---

