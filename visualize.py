from Data_prep import X_mri, y_mri
import matplotlib.pyplot as plt
import numpy as np


# Choose a random index
random_idx = np.random.randint(0, len(X_mri))
random_idx_2 = np.random.randint(0, len(X_mri))
random_idx_3 = np.random.randint(0, len(X_mri))

# Extract the data for that single image
original_img_1, original_img_2, original_img_3 = y_mri[random_idx], y_mri[random_idx_2], y_mri[random_idx_3]
filters_img_1, filters_img_2, filters_img_3 = X_mri[random_idx], X_mri[random_idx_2], X_mri[random_idx_3]

# Set up a plot with 1 row and 4 columns
fig, axes = plt.subplots(3, 4, figsize=(16, 4))


# Plot Original Image 1 and Filters
axes[0][0].imshow(original_img_1, cmap='gray')
axes[0][0].set_title("Original Image (y)")
axes[0][1].imshow(filters_img_1[:, :, 0], cmap='gray')
axes[0][1].set_title("Canny Filter")

# Plot Sobel (Channel 1)
axes[0][2].imshow(filters_img_1[:, :, 1], cmap='gray')
axes[0][2].set_title("Sobel Filter")

# Plot Laplacian (Channel 2)
axes[0][3].imshow(filters_img_1[:, :, 2], cmap='gray')
axes[0][3].set_title("Laplacian Filter")

# Plot Original Image 2 and Filters
axes[1][0].imshow(original_img_2, cmap='gray')
axes[1][0].set_title("Original Image (y)")
axes[1][1].imshow(filters_img_2[:, :, 0], cmap='gray')
axes[1][1].set_title("Canny Filter")

# Plot Sobel (Channel 1)
axes[1][2].imshow(filters_img_2[:, :, 1], cmap='gray')
axes[1][2].set_title("Sobel Filter")

# Plot Laplacian (Channel 2)
axes[1][3].imshow(filters_img_2[:, :, 2], cmap='gray')
axes[1][3].set_title("Laplacian Filter")

# Plot Original Image 3 and Filters
axes[2][0].imshow(original_img_3, cmap='gray')
axes[2][0].set_title("Original Image (y)")
axes[2][1].imshow(filters_img_3[:, :, 0], cmap='gray')
axes[2][1].set_title("Canny Filter")

# Plot Sobel (Channel 1)
axes[2][2].imshow(filters_img_3[:, :, 1], cmap='gray')
axes[2][2].set_title("Sobel Filter")

# Plot Laplacian (Channel 2)
axes[2][3].imshow(filters_img_3[:, :, 2], cmap='gray')
axes[2][3].set_title("Laplacian Filter")

plt.show()