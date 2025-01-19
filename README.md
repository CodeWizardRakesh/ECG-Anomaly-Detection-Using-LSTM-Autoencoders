# ECG-Anomaly-Detection-Using-LSTM-Autoencoders
This project implements an anomaly detection system for ECG signals using an LSTM-based Autoencoder. The system is capable of identifying irregularities in heart rhythms by training on normal ECG data and detecting anomalies as reconstruction errors. Additionally, the project includes a web interface built with Flask for easy interaction and real-time predictions
## Features
- Preprocessing of raw ECG signals.
- Training an LSTM Autoencoder to learn normal ECG patterns.
- Detecting anomalies based on reconstruction error thresholds.
- Flask web application for uploading ECG data and visualizing predictions.
## Flask Application
- Users can upload ECG files in CSV format.
- The app preprocesses the data and feeds it into the trained model.
- Results, including reconstructed signals and anomaly scores, are visualized on the web interface.
## To start the Application
- Navigate to the correct directory
- And run this command
``` bash
python app.py
```
## Screeshots
![img1](https://github.com/CodeWizardRakesh/ECG-Anomaly-Detection-Using-LSTM-Autoencoders/blob/main/images/Screenshot%202025-01-19%20112932.png)
