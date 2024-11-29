import sys
import json
import time
import math
import logging
import requests
import spidev  # SPI library for MCP3008
from openvino.runtime import Core  # Intel AI Toolkit OpenVINO

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize SPI for MQ-135 and MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)  # MCP3008 connected to CE0
spi.max_speed_hz = 1350000

# MCP3008 channel where MQ-135 is connected
MQ135_CHANNEL = 0  # Change this if the sensor is connected to a different channel

# Constants for CO2 PPM to AQI conversion (hypothetical values)
R0 = 10.0  # Replace with your calibrated R0 value for clean air
known_resistor_value = 10.0  # Load resistor value
supply_voltage = 3.3  # Adjust depending on your setup (3.3V or 5V)

# ThingsBoard API endpoint and access token
THINGSBOARD_HOST = 'thingsboard.cloud'
ACCESS_TOKEN = '8ymecb4hog7tirg0mphd'  # Replace with your actual access token

# Load the AI model using OpenVINO
MODEL_PATH = "path_to_your_model/model.xml"  # Replace with your model path
core = Core()
model = core.read_model(MODEL_PATH)
compiled_model = core.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


# Function to read data from MCP3008
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    logging.info(f"Read ADC value: {data}")
    return data


# Function to calculate PPM based on the MQ-135 sensor readings
def calculate_ppm(analog_value):
    voltage = (analog_value / 1023.0) * supply_voltage
    if voltage == 0:
        logging.warning("Voltage is zero, check sensor wiring.")
        return 0, 0

    # Calculate Rs and ratio
    Rs = ((supply_voltage - voltage) / voltage) * known_resistor_value
    ratio = Rs / R0

    # Using logarithmic formula for PPM calculation
    m = -0.38  # Example value from the MQ-135 datasheet
    b = 1.2  # Example value from the MQ-135 datasheet
    PPM = math.pow(10, ((math.log10(ratio) - b) / m))

    logging.info(f"Calculated PPM: {PPM}")
    return PPM, voltage


# Function to calculate AQI based on PPM
def calculate_aqi(ppm):
    # EPA AQI formula for CO2 (adjust based on pollutant)
    AQI_low = 0
    AQI_high = 50
    PPM_low = 0
    PPM_high = 500

    if ppm > PPM_high:
        logging.warning("PPM exceeds standard limits, cannot calculate AQI.")
        return 500  # Maximum AQI (hazardous)

    AQI = ((AQI_high - AQI_low) / (PPM_high - PPM_low)) * (ppm - PPM_low) + AQI_low
    logging.info(f"Calculated AQI: {AQI}")
    return AQI


# Function to predict AQI using Intel AI Toolkit
def predict_aqi_with_ai(analog_value):
    # Prepare input for the AI model (normalize or preprocess as required by your model)
    input_data = [[analog_value]]  # Adjust input shape as required by the model
    result = compiled_model([input_data])[output_layer]
    predicted_aqi = result[0]  # Extract predicted AQI from model output
    logging.info(f"Predicted AQI using AI: {predicted_aqi}")
    return predicted_aqi


# Function to send data to ThingsBoard via HTTP
def send_data_to_thingsboard(data):
    url = f"https://{THINGSBOARD_HOST}/api/v1/{ACCESS_TOKEN}/telemetry"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        logging.info("Data sent successfully to ThingsBoard")
    else:
        logging.error(f"Error sending data to ThingsBoard: {response.text}")


# Main loop to read sensor data, calculate AQI, and send to ThingsBoard
try:
    while True:
        # Read analog value from MQ-135 via MCP3008
        analog_value = read_adc(MQ135_CHANNEL)
        ppm, voltage = calculate_ppm(analog_value)
        aqi = calculate_aqi(ppm)

        # Predict AQI using AI
        predicted_aqi = predict_aqi_with_ai(analog_value)

        # Prepare the payload to send to ThingsBoard
        payload = {
            "airQualityIndex": aqi,
            "predictedAqi": predicted_aqi,
            "mq135_analog_values": analog_value,
            "mq135_voltage": voltage,
            "ppm": ppm
        }
        logging.info(f"Sending data: {payload}")

        # Send data to ThingsBoard
        send_data_to_thingsboard(payload)

        # Delay before sending the next reading
        time.sleep(5)

except KeyboardInterrupt:
    logging.info("Program interrupted.")
except Exception as e:
    logging.error(f"An error occurred: {e}")
finally:
    logging.info("Disconnected from ThingsBoard.")
