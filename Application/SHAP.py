# heartouchan
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import data_process
from sklearn.model_selection import train_test_split
import shap
from SALib.sample import saltelli
from SALib.analyze import morris
import pickle
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_error


# Selected model:
with open('best_xgb_model.pkl', 'rb') as file:
    best_model = pickle.load(file)
X = data_process.X

#SHAP

#Absolute SHAP value and beeswarm plot
plt.rcParams['font.size'] = 20
plt.rcParams.update({'font.size': 22})
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X)
shap.plots.bar(shap_values, show=False)
plt.xticks(fontsize=20)
plt.xlabel('Average absolute SHAP values', fontsize=20)
plt.show()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.xticks(fontsize=20)
plt.xlabel('SHAP values', fontsize=20)
plt.show()

## Dependence plot
shap_min = -30
shap_max = 60
yticks_values = list(range(shap_min, shap_max+ 1, 20))


shap.plots.scatter(shap_values[:,'$W_{t}$'], show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('$W_{t}$ (mm)', fontsize=24)
plt.ylabel('SHAP value for $W_{t}$', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'t'], show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('t (mm)', fontsize=24)
plt.ylabel('SHAP value for t', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'$W_{s}$'],show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('$W_{s}$ (mm)', fontsize=24)
plt.ylabel('SHAP value for $W_{s}$',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'$\delta$'],show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('$\delta$ (mm)', fontsize=24)
plt.ylabel('SHAP value for $\delta$', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'d'], show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('d', fontsize=24)
plt.ylabel('SHAP value for d', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'n'], show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('n', fontsize=24)
plt.ylabel('SHAP value for n', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'$e_{d}$'], show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('$e_{d}$ (mm)', fontsize=24)
plt.ylabel('SHAP value for $e_{d}$',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'C'], show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('C', fontsize=24)
plt.ylabel('SHAP value for C', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()

shap.plots.scatter(shap_values[:,'$\eta$'], show=False, color='black', ymin=shap_min, ymax=shap_max)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.xlabel('$\eta$', fontsize=24)
plt.ylabel('SHAP value for $\eta$', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(yticks_values, fontsize=24)
plt.show()
