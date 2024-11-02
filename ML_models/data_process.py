# heartouchan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data from csv
data=pd.read_csv('Data_connection.csv', encoding='gbk', index_col=0)

# def MinMaxScale(data):
#     data=(data-data.min())/(data.max()-data.min())
#     return data

timber_width = data.iloc[:,0]
timber_thickness = data.iloc[:,1]
steel_plate_width = data.iloc[:,2]
steel_plate_thickness = data.iloc[:,3]
diameter = data.iloc[:,4]
number = data.iloc[:,5]
edge_distance = data.iloc[:,6]
fastener_type=data.iloc[:,7]
load_ratio = data.iloc[:,8]
fire_resistance = data.iloc[:,9]

X = pd.concat([timber_width, timber_thickness, steel_plate_width, steel_plate_thickness, edge_distance, diameter, number,
               fastener_type, load_ratio], axis=1)
Y= fire_resistance
