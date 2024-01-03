import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np
import joblib

# Initialize the main window
root = tk.Tk()
root.title("Rock UCS Intelligent Prediction Platform V1.0")

# Define StringVar for the prediction result
prediction_result_var = tk.StringVar()

# Load your model here
# Replace 'your_model.pkl' with the actual path to your model file
# Make sure to handle this part in your actual application
with open('XGB-ABC_predict_GUI.pkl', 'rb') as file:
    model = pickle.load(file)
scaler_x=joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Function to handle the prediction
def predict():
    try:
        # Extract input values from the GUI entries
        rock_density = float(rock_density_entry.get())
        p_wave_velocity = float(p_wave_velocity_entry.get())
        point_load_strength = float(point_load_strength_entry.get())

        # Preprocess input data as required by your model
        # Assuming your model expects a 1D array with three features
        input_data = np.array([rock_density, p_wave_velocity, point_load_strength]).reshape(1, -1)

        normalized_data = scaler_x.transform(input_data)
        # Make the prediction using the model
        prediction = model.predict(normalized_data)
        print(prediction)
        normalized_pre = scaler_y.inverse_transform(prediction.reshape(-1, 1))
        # Update the prediction_result_var with the prediction result
        prediction_result_var.set(f"{normalized_pre[0][0]:.2f} MPa")
    except ValueError:
        # Handle the error if the input is not a float
        prediction_result_var.set("Please enter valid numbers.")
    except Exception as e:
        # Handle other exceptions
        prediction_result_var.set("Error in prediction.")
        print(e)

# Configure the main window grid weight
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Create the main content frame and configure its grid
content_frame = tk.Frame(root)
content_frame.grid(sticky='nsew', padx=10, pady=10)
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=1)  # Make both columns expand equally

# Add widgets to the content frame
# Data dropdown menu
data_label = tk.Label(content_frame, text="Data")
data_label.grid(row=0, column=0, sticky='w')
data_options = ["Deep Mine"]
data_dropdown = ttk.Combobox(content_frame, values=data_options, state="readonly")
data_dropdown.grid(row=0, column=1, sticky='ew')
data_dropdown.set(data_options[0])  # Set default value

# Rock Density Input
rock_density_label = tk.Label(content_frame, text="Rock density (g/cm³):")
rock_density_label.grid(row=1, column=0, sticky='w')
rock_density_entry = tk.Entry(content_frame)
rock_density_entry.grid(row=1, column=1, sticky='ew')

# P-wave Velocity Input
p_wave_velocity_label = tk.Label(content_frame, text="P-wave Velocity (m/s):")
p_wave_velocity_label.grid(row=2, column=0, sticky='w')
p_wave_velocity_entry = tk.Entry(content_frame)
p_wave_velocity_entry.grid(row=2, column=1, sticky='ew')

# Point Load Strength Input
point_load_strength_label = tk.Label(content_frame, text="Point load strength index (MPa):")
point_load_strength_label.grid(row=3, column=0, sticky='w')
point_load_strength_entry = tk.Entry(content_frame)
point_load_strength_entry.grid(row=3, column=1, sticky='ew')

# Prediction Result Output
result_label = tk.Label(content_frame, text="Rock UCS prediction result (MPa)")
result_label.grid(row=4, column=0, sticky='w')
result_entry = tk.Entry(content_frame, textvariable=prediction_result_var, state="readonly")
result_entry.grid(row=4, column=1, sticky='ew')

# Output Button
output_button = tk.Button(content_frame, text="Output", command=predict)
output_button.grid(row=5, column=0, columnspan=2, sticky='ew')

# Make the row of the output button not expand by setting its weight to 0
content_frame.grid_rowconfigure(5, weight=0)

# Make all other rows expand equally by giving them a weight
for i in range(5):  # For the five rows above the output button
    content_frame.grid_rowconfigure(i, weight=1)

# Technical Support Label
tech_support_label = tk.Label(content_frame, text="Technical support: junjie-zhao@csu.edu.cn")
tech_support_label.grid(row=6, column=0, columnspan=2, sticky='w')

# Run the main loop
root.mainloop()
