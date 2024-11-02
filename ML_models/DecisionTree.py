# heartouchan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import data_process
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_error


#Figure_font
plt.rcParams['font.family'] = 'Times New Roman'

#Input X and Y
X = data_process.X
Y= data_process.Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state=2021)

#Establish model with GridSearch for hyperparameters
model=DecisionTreeRegressor(random_state=2021)
scores = ["r2", 'neg_root_mean_squared_error','neg_mean_absolute_error']
param_grid = {
    'max_depth': [5,8, 10,15],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4],
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scores, refit='r2', verbose=4, return_train_score=True)
grid_search.fit(X_train, Y_train)

# # Performance metrics output
best_index = grid_search.best_index_
mean_train_r2 = grid_search.cv_results_['mean_train_r2'][best_index]
mean_train_rmse = -grid_search.cv_results_['mean_train_neg_root_mean_squared_error'][best_index]
mean_train_mae = -grid_search.cv_results_['mean_train_neg_mean_absolute_error'][best_index]
mean_test_r2 = grid_search.cv_results_['mean_test_r2'][best_index]
mean_test_rmse = -grid_search.cv_results_['mean_test_neg_root_mean_squared_error'][best_index]
mean_test_mae = -grid_search.cv_results_['mean_test_neg_mean_absolute_error'][best_index]
print("Average Train R^2 (5-fold CV):", mean_train_r2)
print("Average Train RMSE (5-fold CV):", mean_train_rmse)
print("Average Train MAE (5-fold CV):", mean_train_mae)
print("Average Test R^2 (5-fold CV):", mean_test_r2)
print("Average Test RMSE (5-fold CV):", mean_test_rmse)
print("Average Test MAE (5-fold CV):", mean_test_mae)
print("Best Parameters:", grid_search.best_params_)

# Prediction on the whole datset
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X)
r2=r2_score(Y,Y_pred)
print('R2', r2)
# Plot the actual vs predicted values
plt.figure(figsize=(10, 8))
plt.scatter(Y, Y_pred, color='black', label='Actual vs Predicted')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='-', label='Perfect Prediction')
plt.xlabel('Actual Value', fontsize=28)
plt.ylabel('Predicted Value', fontsize=28)
plt.legend(fontsize=28)
plt.grid(True)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()
