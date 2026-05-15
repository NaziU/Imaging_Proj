import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Load the trained model
model = load_model('resunet_mri_model.keras')

#Load the data
X_val = np.load("X_data_mri.npy")
y_val = np.load("y_data_mri.npy")

#predictions
predictions = model.predict(X_val)

num_rows = X_val.shape[0]  # Number of samples in the dataset

print (f"X_val Shape: {X_val.shape}, y_val Shape: {y_val.shape}, predictions Shape: {predictions.shape}")

# Set up a plot with 1 row and 5 columns
fig, axes = plt.subplots(num_rows, 5, figsize=(8, num_rows * 1.5))

for i in range(num_rows):
    filters_img = X_val[i]
    original_img = y_val[i]
    
    # Plot Original Image 1 and Filters
    axes[i][0].imshow(original_img, cmap='gray')
    axes[i][0].set_xticks([])
    axes[i][0].set_yticks([])

    # Canny (Channel 0)
    axes[i][1].imshow(filters_img[:, :, 0], cmap='gray')
    axes[i][1].set_xticks([])
    axes[i][1].set_yticks([])

    # Plot Sobel (Channel 1)
    axes[i][2].imshow(filters_img[:, :, 1], cmap='gray')
    axes[i][2].set_xticks([])
    axes[i][2].set_yticks([])

    # Plot Laplacian (Channel 2)
    axes[i][3].imshow(filters_img[:, :, 2], cmap='gray')
    axes[i][3].set_xticks([])
    axes[i][3].set_yticks([])

    # plot the predicted mask
    axes[i][4].imshow(predictions[i, :, :, 0], cmap='gray')
    axes[i][4].set_xticks([])
    axes[i][4].set_yticks([])

plt.savefig('MRI_Grid_predictions.png')
plt.show()
plt.close()