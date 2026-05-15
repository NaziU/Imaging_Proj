from calendar import c
from tabnanny import verbose

import bottleneck
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation , Add
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from torch import Tensor, mode



def residual_block(x, filters, stride=1):
    # Save the original input for the skip connection expressway
    shortcut = x

    # --- The Main Path ---
    # First Convolution (Heavy Lifting)
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Convolution (Refining)
    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # --- The Expressway (Skip Connection) ---
    # If we changed the spatial dimensions (stride > 1) OR the depth (filters),
    # we mathematically cannot add the shortcut to 'x'. 
    # We must project the shortcut using a 1x1 convolution to match the new shape.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add the expressway back to the main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


# Convert lists to numpy arrays and add channel dimension
X = np.array(np.load("X_data_mri.npy"))
y = np.array(np.load("y_data_mri.npy"))

size = X.shape[1]  # Assuming X has shape (num_samples, height, width, channels)

y = np.expand_dims(y, axis=-1)  # Add channel dimension to y

print(f"X Shape: {X.shape}, y Shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

input_layer = Input(shape=(size, size, 3))
# Encoder
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
conv1 = residual_block(conv1, filters=64)
conv1 = residual_block(conv1, filters=64)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = residual_block(pool1, filters=128)
conv2 = residual_block(conv2, filters=128)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = residual_block(pool2, filters=256)
conv3 = residual_block(conv3, filters=256)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = residual_block(pool3, filters=512)
conv4 = residual_block(conv4, filters=512)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#--- Bottleneck ---

bottleneck = Conv2D(1024,(3,3),activation='relu',padding='same')(pool4)
bottleneck = Conv2D(1024,(3,3),activation='relu',padding='same')(bottleneck)

# Decoder
upconv1 =Conv2DTranspose(512,(2,2),strides=(2, 2),padding='same')(bottleneck)
concat1 = concatenate([upconv1, conv4])
conv5 = Conv2D(512,(3,3),activation='relu',padding='same')(concat1)
conv5 = residual_block(conv5, filters=512)
conv5 = residual_block(conv5, filters=512)

upconv2 = Conv2DTranspose(256,(2,2),strides=(2, 2),padding='same')(conv5)
concat2 = concatenate([upconv2, conv3])
conv6 = Conv2D(256,(3,3),activation='relu',padding='same')(concat2)
conv6 = residual_block(conv6, filters=256)
conv6 = residual_block(conv6, filters=256)

upconv3 = Conv2DTranspose(128,(2,2),strides=(2, 2),padding='same')(conv6)
concat3 = concatenate([upconv3, conv2])
conv7 = Conv2D(128,(3,3),activation='relu',padding='same')(concat3)
conv7 = residual_block(conv7, filters=128)
conv7 = residual_block(conv7, filters=128)

upconv4 = Conv2DTranspose(64,(2,2),strides=(2, 2),padding='same')(conv7)
concat4 = concatenate([upconv4, conv1])
conv8 = Conv2D(64,(3,3),activation='relu',padding='same')(concat4)
conv8 = residual_block(conv8, filters=64)
conv8 = residual_block(conv8, filters=64)

# output layer
output_layer = Conv2D(1,(1,1),activation='sigmoid',padding='same')(conv8)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_absolute_error', metrics=['mse'])

# 1. Save the training process to a variable
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=16, verbose=1)

# 2. Extract the loss data
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs_range = range(1, len(training_loss) + 1)

# 3. Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, training_loss, label='Training Loss (MAE)', color='blue')
plt.plot(epochs_range, validation_loss, label='Validation Loss (MAE)', color='red', linestyle='--')

plt.title('ResUNet MRI Reconstruction Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss.png')
plt.show()

model.save("resunet_mri_model.keras")