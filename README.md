# FRR_ML_Optimization

This project aims to:
1. Predict the Fire Resistance Rating (FRR) of wood-steel-wood (WSW) connections using machine learning (ML) models.
2. Optimize the FRR of the connection using the Nondominated Sorting Genetic Algorithm II (NSGA-II).

A dataset of 140 test and modeling results has been collected, and the performance of nine ML models on this dataset has been evaluated, check `ML_models` folder. The best-performing ML model is then used for SHAP analysis to understand feature importance. A graphical user interface (GUI) for predicting the FRR of WSW connections is provided in the `Application` folder. This ML model is embedded in the DEAP framework to apply NSGA-II optimization, aiming to optimize both the self-weight and FRR of the connection.

## Requirements

To run this code package, please install the following dependencies:

```bash
Python 3.9 or higher
PySimpleGUI
NumPy
Pandas
Pillow
pickle
scikit-learn
shap
deap
