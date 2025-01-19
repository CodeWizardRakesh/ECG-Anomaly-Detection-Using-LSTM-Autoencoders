import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory
import os

import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
#from arff2pandas import a2p
from scipy.io import arff
# Constants
class Encoder(nn.Module): 

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

#Decoder
class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

#Encoder + Decoder
class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

THRESHOLD = 26
CLASS_NORMAL = 1
RANDOM_SEED = 42

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and Data Paths
model_path = './Model/Heart.pth'
data_path = './Data-20241013T093531Z-001/EGC.csv'
input_plot_dir = "./history/input_plot"
output_plot_dir = "./history/output_plot"

# Load the pre-trained model
model = torch.load(model_path, map_location=device)

# Load the dataset
df = pd.read_csv(data_path)

# Ensure necessary directories exist
os.makedirs(input_plot_dir, exist_ok=True)
os.makedirs(output_plot_dir, exist_ok=True)


THRESHOLD = 26
CLASS_NORMAL = 1
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class Encoder(nn.Module): 

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

#Decoder
class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

#Encoder + Decoder
class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x
# Define the Flask app
app = Flask(__name__)




# Prediction functions

def predict_single(model, data):
    model.eval()  # Set the model to evaluation mode
    data = torch.tensor(data).float().unsqueeze(1).to(device)  # Prepare data for input
    with torch.no_grad():
        seq_pred = model(data)
        criterion = torch.nn.L1Loss(reduction='sum').to(device)
        loss = criterion(seq_pred, data).item()  # Compute the loss
    return seq_pred.cpu().numpy().flatten(), loss  # Return prediction and the loss


def classify_and_plot(data, model, threshold=THRESHOLD):
    prediction, loss = predict_single(model, data)
    result = "Normal" if loss <= threshold else "Abnormal"
    
    # Plot and save input data
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Input ECG', color='blue')
    plt.title("Input ECG Data")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    input_plot_path = f"{input_plot_dir}/input_data_plot.png"
    plt.savefig(input_plot_path)
    plt.close()

    # Plot and save reconstructed data
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='True ECG', color='blue')
    plt.plot(prediction, label=f'Reconstructed ECG ({result})', color='red')
    plt.title(f"ECG Classification: {result} (Loss: {loss:.2f})")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    output_plot_path = f"{output_plot_dir}/output_data_plot.png"
    plt.savefig(output_plot_path)
    plt.close()
    
    return result, input_plot_path, output_plot_path


# Flask routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        patient_number = int(request.form['patient_number'])
        ecg_data = df.iloc[patient_number].drop('target').values  # Fetch the data for the given patient number
        result, input_plot, output_plot = classify_and_plot(ecg_data, model)
        return render_template('result.html', patient_number=patient_number, result=result)


@app.route('/input_plot/<filename>')
def input_plot(filename):
    return send_from_directory(input_plot_dir, filename)


@app.route('/output_plot/<filename>')
def output_plot(filename):
    return send_from_directory(output_plot_dir, filename)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
