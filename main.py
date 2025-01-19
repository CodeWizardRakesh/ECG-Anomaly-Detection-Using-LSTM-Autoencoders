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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('.\Model\Heart.pth', map_location=torch.device('cuda'))

df = pd.read_csv(".\Data-20241013T093531Z-001\EGC.csv")
print(df.shape)

normal_df = df[df.target == CLASS_NORMAL].drop(labels='target', axis=1)
normal_df.shape

anomaly_df = df[df.target != CLASS_NORMAL].drop(labels='target', axis=1)
anomaly_df.shape

train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
  val_df,
  test_size=0.33,
  random_state=RANDOM_SEED
)

def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features


train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)
test_df.to_csv('ECG_normal.csv')
anomaly_df.to_csv('ECG_anomaly.csv')



def predict_single(model, data):
    """
    Predicts whether a single ECG sequence is normal or abnormal.
    """
    model.eval()  # Set the model to evaluation mode
    data = torch.tensor(data).float().unsqueeze(1).to(device)  # Prepare data for input

    with torch.no_grad():
        seq_pred = model(data)
        criterion = nn.L1Loss(reduction='sum').to(device)  # L1 Loss    to CALCULATING LOSS
        loss = criterion(seq_pred, data).item()  # Compute the loss
        
    return seq_pred.cpu().numpy().flatten(), loss  # Return prediction and the loss

def classify_and_plot(data, model, threshold=THRESHOLD):
    """
    Classifies the ECG data as normal or abnormal, and plots the input and reconstruction.
    """
    # Predict the reconstruction and loss for the given ECG data
    prediction, loss = predict_single(model, data)
    
    # Classify based on the loss
    if loss <= threshold:
        result = "Normal"
    else:
        result = "Abnormal"
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Input ECG', color='blue')
    plt.title("Input ECG Data")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(".\history\input_plot\input_data_plot.png")  # Save the input data plot
    plt.close()
    
    
    
    # Plot the original ECG and the reconstruction
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='True ECG', color='green')
    plt.plot(prediction, label=f'Reconstructed ECG ({result})', color='red')
    plt.title(f"ECG Classification: {result} (Loss: {loss:.2f})")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    
    
    plt.savefig(".\history\output_plot\output_data_plot.png")
    plt.show()
    plt.close()
    
    print(f'Classification: {result}')


# def plot_prediction(data, model, title, ax):
#   predictions, pred_losses = predict(model, [data])

#   ax.plot(data, label='true')
#   ax.plot(predictions[0], label='reconstructed')
#   ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
#   ax.legend()
  
  

# fig, axs = plt.subplots(
#   nrows=2,
#   ncols=6,
#   sharey=True,
#   sharex=True,
#   figsize=(22, 8)
# )

# for i, data in enumerate(test_normal_dataset[:6]):
#   plot_prediction(data, model, title='Normal', ax=axs[0, i])
# for i, data in enumerate(test_anomaly_dataset[:6]):
#   plot_prediction(data, model, title='Anomaly', ax=axs[1, i])

# fig.tight_layout()

# # Replacing sns.distplot with plt.hist + kde plot (using seaborn's kdeplot for consistency)
# _, losses = predict(model, train_dataset)
# plt.figure(figsize=(10, 5))
# plt.hist(losses, bins=50, density=True, alpha=0.6, color='g')
# sns.kdeplot(losses, color='blue')
# plt.title('Training Loss Distribution')
# plt.xlabel('Loss')
# plt.ylabel('Density')
# plt.show()

# # Normal Heart Prediction
# predictions, pred_losses = predict(model, test_normal_dataset)
# plt.figure(figsize=(10, 5))
# plt.hist(pred_losses, bins=50, density=True, alpha=0.6, color='g')
# sns.kdeplot(pred_losses, color='blue')
# plt.title('Test Normal Loss Distribution')
# plt.xlabel('Loss')
# plt.ylabel('Density')
# plt.show()

# correct = sum(l <= THRESHOLD for l in pred_losses)
# print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')

# # Anomaly Heart Prediction
# anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
# predictions, pred_losses = predict(model, anomaly_dataset)
# plt.figure(figsize=(10, 5))
# plt.hist(pred_losses, bins=50, density=True, alpha=0.6, color='g')
# sns.kdeplot(pred_losses, color='blue')
# plt.title('Test Anomaly Loss Distribution')
# plt.xlabel('Loss')
# plt.ylabel('Density')
# plt.show()

# correct = sum(l > THRESHOLD for l in pred_losses)
# print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')


sample_ecg = test_normal_dataset[105].cpu().numpy().flatten()
classify_and_plot(sample_ecg, model)

sample_anomaly = test_anomaly_dataset[200].cpu().numpy().flatten()
classify_and_plot(sample_anomaly, model)
