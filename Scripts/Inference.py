import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from torch import nn
import Res_Unet as resunet
from Res_Unet import Config
from Data_prep import Data_Handler



class inf_model:
    def __init__(self):
        self.model = resunet.smp.Unet(
            encoder_name=resunet.Config.encoder_name,
            encoder_weights=resunet.Config.encoder_weights,
            in_channels=resunet.Config.in_channels,
            classes=resunet.Config.classes
        )
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            # X comes in as (Batch, Height, Width, Channels)
            # We must permute to (Batch, Channels, Height, Width) for PyTorch!
            X_tensor = torch.from_numpy(X).float().permute(0, 3, 1, 2).to(self.device)
            predictions = self.model(X_tensor)
            # Permute back to (Batch, Height, Width, Channels) for Matplotlib
            return predictions.cpu().permute(0, 2, 3, 1).numpy()
        
inf = inf_model()
inf.load_weights("resunet34_model_mri.pth")

X = np.load("Outputs/X_data_mri.npy")
print("Loaded X_data_mri.npy, Data shape: ", X.shape)
y = np.load("Outputs/y_data_mri.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=Config.train_test_ratio)

test_val = X_val[:3]  # Take the first 3 samples from the validation set for inference
predictions = inf.predict(test_val)

print(f"Test Validation shape: {test_val.shape}, X_val Shape: {X_val.shape}")

Data_Handler.plot_grid(X=test_val, y=y_val[:3], type="PREDICT", predictions=predictions)