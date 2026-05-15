from glob import glob
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

#data 


def Data_prep(path, size = (256, 256)):
    #define Lists to store the images and filters
   
    images_mri = []
    filter_3_mri = []
    images_xray = []
    filter_3_xray = []

    #Loop through both folders 
    for folder_path in path:
        if "MRI" in folder_path:
            # Match common image extensions
            search_pattern = os.path.join(folder_path, "*.*")
            for file_path in sorted(glob(search_pattern)):
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                #Read the image and mask
                img = cv2.imread(file_path)
                
                #Resize the image and mask to the desired size
                img = cv2.resize(img, size)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                img = img / 255.0  # Normalize the image to [0, 1]

                #append image and filters to the lists
                images_mri.append(img)

                # create Canny / Sobel / Laplacian filters
                f1_normalised = cv2.Canny((img*255).astype(np.uint8), 100, 200) / 255.0  
                f2 = cv2.Sobel((img*255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=5)
                f2_normalised = ( f2 - np.min(f2) ) / ( np.max(f2) - np.min(f2) + 1e-8 )  # Normalize to [0, 1]
                f3 = cv2.Laplacian((img*255).astype(np.uint8), cv2.CV_64F)
                f3_normalised = ( f3 - np.min(f3) ) / ( np.max(f3) - np.min(f3) + 1e-8 )  # Normalize to [0, 1]

                # Stack the filters into a single 3-channel list
                list_3 = [f1_normalised, f2_normalised, f3_normalised]
                # make a combined list of the 3 filters and append it to the filter_3 list
                # the combined list will have the shape (256, 256, 3) , this is the 3 channel input image
                combined_filters = np.stack(list_3, axis=-1)
                filter_3_mri.append(combined_filters)

        if "Xray" in folder_path:
            search_pattern = os.path.join(folder_path, "*.*")
            for file_path in sorted(glob(search_pattern)):
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                #Read the image and mask
                img = cv2.imread(file_path)
                
                #Resize the image and mask to the desired size
                img = cv2.resize(img, size)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                img = img / 255.0  # Normalize the image to [0, 1]

                #append image and filters to the lists
                images_xray.append(img)

                # create Canny / Sobel / Laplacian filters
                f1_normalised = cv2.Canny((img*255).astype(np.uint8), 100, 200) / 255.0  
                f2 = cv2.Sobel((img*255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=5)
                f2_normalised = ( f2 - np.min(f2) ) / ( np.max(f2) - np.min(f2) + 1e-8  )  # Normalize to [0, 1]
                f3 = cv2.Laplacian((img*255).astype(np.uint8), cv2.CV_64F)
                f3_normalised = ( f3 - np.min(f3) ) / ( np.max(f3) - np.min(f3) + 1e-8 )  # Normalize to [0, 1]

                # Stack the filters into a single 3-channel list
                list_3 = [f1_normalised, f2_normalised, f3_normalised]
                # make a combined list of the 3 filters and append it to the filter_3 list
                # the combined list will have the shape (256, 256, 3) , this is the 3 channel input image
                combined_filters = np.stack(list_3, axis=-1)
                filter_3_xray.append(combined_filters)
    
    X_mri = np.array(filter_3_mri)
    y_mri = np.array(images_mri)
    np.save("Outputs/X_data_mri.npy", X_mri)
    print("saved Outputs/X_data_mri.npy, Data shape: ", X_mri.shape)
    np.save("Outputs/y_data_mri.npy", y_mri)
    print("saved Outputs/y_data_mri.npy, Data shape: ", y_mri.shape)

    X_xray = np.array(filter_3_xray)
    y_xray = np.array(images_xray)
    np.save("Outputs/X_data_xray.npy", X_xray)
    print("saved Outputs/X_data_xray.npy, Data shape: ", X_xray.shape)
    np.save("Outputs/y_data_xray.npy", y_xray)
    print("saved Outputs/y_data_xray.npy, Data shape: ", y_xray.shape)

    return X_mri, y_mri, X_xray, y_xray

def plot_grid(X, y, type = "MRI" ):

    type = type.upper()
    num_rows = X.shape[0]  # Number of samples in the dataset

    # Set up a plot with 1 row and 4 columns
    fig, axes = plt.subplots(num_rows, 4, figsize=(5, num_rows))

    for i in range(num_rows):
        filters_img = X[i]
        original_img = y[i]

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
    if type == "MRI":
      plt.savefig("Outputs/MRI_Plots.png")
    elif type == "XRAY":
      plt.savefig("Outputs/Xray_Plots.png")
    else :
      plt.savefig("Outputs/Grid.png")
    plt.show()
    plt.close()


def main():
    path = [r"C:\Users\nazeh\Downloads\Imaging_Proj\Group 9\MRI", 
            r"C:\Users\nazeh\Downloads\Imaging_Proj\Group 9\Xray"]
    X_mri, y_mri, X_xray, y_xray = Data_prep(path)
    plot_grid(X_mri, y_mri, type="MRI")
    plot_grid(X_xray, y_xray, type="XRAY")

if __name__ == "__main__":
    main()
