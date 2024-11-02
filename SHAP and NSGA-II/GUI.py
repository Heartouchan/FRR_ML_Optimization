from pickle import load
import pandas as pd
import numpy as np
import PySimpleGUI as sg
from PIL import Image
from PIL import ImageOps
import pickle

# import the dataset
dd1 = pd.read_csv('Data_connection.csv',encoding='gbk', index_col=0)
df = dd1.copy(deep=True)

# Set text width for labels and input fields
t = 40
td = 35
td2 = 10


# Define parameter ranges for user input validation
timber_width_range = [140, 400]
timber_thickness_range = [38, 236]
steel_plate_width_range = [87, 214]
steel_plate_thickness_range = [6, 15]
diameter_range = [5, 20]
number_range= [1, 32]
edge_distance_range = [20, 130]
fastener_type_range = [1, 0]
load_ratio_range = [10, 30]

# Set GUI theme
sg.theme('LightGreen')

# Define layout for GUI components
layout = [
    [sg.Text('Developed by Tongchen Han, Zhidong Zhang, Weiwei Wu'),],
    [sg.Text('University of British Columbia, University of Virginia, The Hong Kong Polytechnic University')],

    [
        sg.Column(layout=[
            [sg.Frame(layout=[
                [sg.Text('Timber panel width Wt', size=(t, 1)), sg.InputText(key='-f1-', size=(td2, 1)),
                 sg.Text('mm')],
                [sg.Text('Timber panel thickness t1 ', size=(t, 1)), sg.InputText(key='-f2-', size=(td2, 1)),
                 sg.Text('mm')],
                [sg.Text('Steel plate width Ws', size=(t, 1)), sg.InputText(key='-f3-', size=(td2, 1)), sg.Text('mm')],
                [sg.Text('Steel plate thickness δ', size=(t, 1)), sg.InputText(key='-f4-', size=(td2, 1)), sg.Text('mm')],
                [sg.Text('Diameter of fastener d', size=(t, 1)), sg.InputText(key='-f5-', size=(td2, 1)),
                 sg.Text('mm')],
                [sg.Text('Number of fastener n', size=(t, 1)), sg.InputText(key='-f6-', size=(td2, 1)),
                 sg.Text('--')],
                [sg.Text('Edge distance ed', size=(t, 1)), sg.InputText(key='-f7-', size=(td2, 1)),
                 sg.Text('mm')],
                [sg.Text('Fastener type C (1:Bolt, 0:Dowel)', size=(t, 1)),
                 sg.InputCombo(fastener_type_range, key='-f8-',
                               default_value=1, size=(td2, 1)), sg.Text('--')],
                [sg.Text('Load ratio η', size=(t, 1)), sg.InputText(key='-f9-', size=(td2, 1)),
                 sg.Text('%')],

],
                title='Input parameters')],
        ], justification='left'),

        sg.Column(layout=[
            [sg.Frame(layout=[
                [sg.Text('140 mm ≤ Wt ≤ 400 mm')],
                [sg.Text('38 mm ≤ t1 ≤ 236 mm')],
                [sg.Text('87 mm ≤ Ws ≤ 214 mm')],
                [sg.Text('6 mm ≤ δ ≤ 15 mm')],
                [sg.Text('5 mm ≤ d ≤ 20 mm')],
                [sg.Text('1 ≤ n ≤ 20')],
                [sg.Text('20 mm ≤ ed ≤ 130 mm')],
                [sg.Text('1:Bolt,    0:Dowel')],
                [sg.Text('10 %  ≤ η ≤ 30 %')],
],
                title='Range of application of the models')],

        ], justification='center')
    ],
    [sg.Frame(layout=[
        [sg.Text('Fire resistance time', size=(32, 1)), sg.InputText(key='-OP-', size=(td2, 1)), sg.Text('min')]],
        title='Output')],
    [sg.Button('Predict'), sg.Button('Cancel')]
]


# Load and resize images
img1 = Image.open('image1.png')
img2 = Image.open('image2.png')

# Get original dimensions of images
width, height = img1.size
width2, height2 = img2.size

# Define scale factors for resizing
scale_factor = 0.15
scale_factor_1=0.18

# Resize the images
img1 = img1.resize((int(width * scale_factor), int(height * scale_factor)))
img2 = img2.resize((int(width2 * scale_factor_1), int(height2 * scale_factor_1)))

# Save the resized images
img1.save('image11.png')
img2.save('image22.png')

# Add images to layout
fig1 = sg.Image(filename='image11.png', key='-fig1-', size=(width * scale_factor, height * scale_factor))
fig2 = sg.Image(filename='image22.png', key='-fig2-', size=(width2 * scale_factor_1, height2 * scale_factor_1))


# Add image descriptions to layout
layout += [[sg.Text(' ')],
           [sg.Text('Schematic of connection and importance of the input features')],
           [fig1, fig2],
           ]


# Create the Window
window = sg.Window('Fire resistance rating prediction of WSW connections informed by ML models', layout)

# Load trained XGBoost model
filename = 'best_xgb_model.pkl'
model = load(open(filename, 'rb'))

# Event loop for handling user actions
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    elif event == 'Predict':
        try:
            # get the input values
            timber_width = float(values['-f1-'])
            timber_thickness= int(values['-f2-'])
            steel_plate_width = float(values['-f3-'])
            steel_plate_thickness = float(values['-f4-'])
            diameter = float(values['-f5-'])
            number = float(values['-f6-'])
            edge_distance = int(values['-f7-'])
            fastener_type = int(values['-f8-'])
            load_ratio = int(values['-f9-'])

            # Validate input values
            if timber_width < timber_width_range[0] or timber_width> timber_width_range[1]:
                sg.popup("Timber panel width must be between 140 mm and 400 mm.")
                continue
            if fastener_type not in fastener_type_range:
                sg.popup("Fastener type must be 1 for bolts or 0 for dowel.")
                continue
            if timber_thickness < timber_thickness_range[0] or timber_thickness > timber_thickness_range[1]:
                sg.popup("Timber panel thickness must be between 38 mm and 236 mm.")
                continue
            if steel_plate_width < steel_plate_width_range[0] or steel_plate_width > steel_plate_width_range[1]:
                sg.popup("Steel plate width must be between 6 mm and 15 mm.")
                continue
            if steel_plate_thickness < steel_plate_thickness_range[0] or steel_plate_thickness > steel_plate_thickness_range[1]:
                sg.popup("Steel plate thickness must be between 6 mm and 15 mm.")
                continue
            if diameter < diameter_range[0] or diameter> diameter_range[1]:
                sg.popup("Diameter of fastener must be between 5 mm and 20 mm.")
                continue
            if number < number_range[0] or number> number_range[1]:
                sg.popup("Number of fastener must be between 1 and 20.")
                continue
            if edge_distance < edge_distance_range[0] or edge_distance > edge_distance_range[1]:
                sg.popup("Edge distance must be between 20 mm and 130 mm.")
                continue
            if load_ratio < load_ratio_range[0] or load_ratio > load_ratio_range[1]:
                sg.popup("Load ratio must be between 10% and 30%.")
                continue




            # Predict fire resistance time using the model
            df11 = np.array([[timber_width, timber_thickness, steel_plate_width, steel_plate_thickness, edge_distance, diameter, number,
               fastener_type, load_ratio]])
            prediction = model.predict(df11)[0]
            window['-OP-'].update(np.round(prediction, 2))

        except:
            sg.popup(
                "Invalid input. Please make sure to enter numeric values and make sure the input values are within the defined range.")
            continue

window.close()