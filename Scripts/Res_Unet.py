import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.models as models
import segmentation_models_pytorch as smp


class Config:
    # Architecture
    encoder_name = 'resnet34' 
    encoder_weights = 'imagenet'
    pretrained = True
    in_channels = 3
    classes = 1
    
    # Training
    batch_size = 4
    learning_rate = 1e-4
    epochs = 40
    train_test_ratio = 0.4

    @classmethod
    def plotting_loss(cls, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, cls.epochs + 1), train_losses, label='Training Loss (MAE)', color='blue')
        plt.plot(range(1, cls.epochs + 1), val_losses, label='Validation Loss (MAE)', color='red', linestyle='--')

        plt.title('ResUNet MRI Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_validation_loss.png')
        plt.show()

class scan_Dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.x = X_data
        self.y = y_data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        # 3. Grab the specific image and label requested by the 'index'
        img = self.x[index]
        label = self.y[index]
        
        
        # 4. Convert them to PyTorch 'Tensors' and fix the channel order!
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) # Moves HWC to CHW for a single image
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0) # Adds the 1 channel to Make it (1, H, W)
        
        return img_tensor, label_tensor

class model_build:
    def __init__(self):
        self.model = smp.Unet(
            encoder_name=Config.encoder_name,        
            encoder_weights=Config.encoder_weights,     
            in_channels=Config.in_channels,                  
            classes=Config.classes,                      
        )
    def compile(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        self.criterion = torch.nn.L1Loss()  # L1 Loss for regression-like segmentation

    def train(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        for epoch in range(Config.epochs):
            self.model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * images.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

             # Validation loop
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)

            print(f'Epoch [{epoch+1}/{Config.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

        torch.save(self.model.state_dict(), f"{Config.encoder_name}_model.pth")
        return train_losses, val_losses

if __name__ == "__main__":
            
    X = np.load("Outputs/X_data_mri.npy")
    y = np.load("Outputs/y_data_mri.npy")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=Config.train_test_ratio)

    train_dataset = scan_Dataset(X_train, y_train)
    val_dataset = scan_Dataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    model = model_build()
    model.compile()
    train_losses, val_losses = model.train(train_loader, val_loader)

    # Plot the training and validation loss
    Config.plotting_loss(train_losses, val_losses)

