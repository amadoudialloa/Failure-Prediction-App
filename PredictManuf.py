import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
np.random.seed(42)

# Define the number of samples
n_samples = 1000

# Generate random features (sensors) for the motors
sensor_columns = ["Temperature", "Pressure", "Vibration", "Voltage"]
sensor_data = np.random.rand(n_samples, len(sensor_columns))

# Generate random labels (1 for failure, 0 for no failure) for the motors
labels = np.random.randint(2, size=n_samples)

# Create a DataFrame from the synthetic data
data = pd.DataFrame(data=np.column_stack((sensor_data, labels)), columns=sensor_columns + ["Failure"])

# Split the data into training and testing sets
X = data[sensor_columns]
y = data["Failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Fit the model with training data
model.fit(X_train, y_train)

Description = """

Developed by: Alpha Diallo

Summary: 

The Manufacturing Failure Prediction App is a cutting-edge tool designed to assist manufacturing industries in predicting equipment or product failures with exceptional accuracy. Leveraging machine learning techniques, this application takes sensor data as input and provides valuable insights into whether a failure event is imminent or not.

Description:"

In today's highly competitive manufacturing landscape, minimizing downtime and optimizing production processes are paramount. The Manufacturing Failure Prediction App empowers manufacturing teams to make proactive decisions by predicting equipment or product failures before they occur.

How It Works:

Data Collection: The app collects sensor data from various sources within the manufacturing environment, such as temperature, pressure, vibration, and voltage.

Machine Learning Model: Behind the scenes, a robust Random Forest Classifier model has been trained on historical data. This model has learned to recognize patterns and trends in the sensor data that correlate with equipment or product failures.

Real-Time Predictions: Users enter current sensor data into the app, such as temperature, pressure, vibration, and voltage. When they click the "Predict" button, the app sends this data to the machine learning model for analysis.

Predictions: The model processes the input data and returns a prediction, classifying it as either "Failure" or "No Failure."

Key Benefits:

 - Reduced Downtime 
 - Improved Productivity 
 - Enhanced Safety
 - Data-Driven Decision-Making
"""  



# Streamlit app title and description
st.title("Manufacturing Failure Prediction App")
st.write(Description)

st.write("Enter sensor data to predict whether a failure will occur or not.")

# Create input fields for sensor data
temperature = st.number_input("Temperature")
pressure = st.number_input("Pressure")
vibration = st.number_input("Vibration")
voltage = st.number_input("Voltage")

# Create a button to make predictions
if st.button("Predict"):
    # Preprocess the input data into a DataFrame
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Pressure": [pressure],
        "Vibration": [vibration],
        "Voltage": [voltage]
    })
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Determine the prediction label
    if prediction[0] == 1:
        prediction_label = "Failure"
    else:
        prediction_label = "No Failure"
    
    # Display the prediction
    st.write(f"Predicted Result: {prediction_label}")

