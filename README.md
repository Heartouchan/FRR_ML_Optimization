# FRR_ML_Optimization
This code is to 1) predict the fire resistance rating (FRR) of wood-steel-wood (WSW) connections based on machine-learning (ML) models and 2) optimize the FRR of the connection using Nondominated Sorting Genetic Algorithm II (NSGA=II).

140 test and modelling results are collected and performances of nine ML models on the dataset are evaluated. The best ML model is employed for SHAP analysis. The GUI for predicting the FRR of WSW connections is provided in Application folder. The ML model is embeded in deap framework for NSGA-II to optimize the self-weight of the connection and FRR. 

**Requirement**

To run the code package, the following dependencies should be installed:

```bash
Python 3.9 or higher
PySimpleGUI
NumPy
Pandas
Pillow (PIL)
pickle
scikit-learn
shap
deap
