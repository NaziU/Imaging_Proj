from glob import glob
import cv2
import numpy as np

#data 
path = [r"Group 9\MRI",
        r"Group 9\Xray"]

size = (256,256)

images = []
filter_3 = []


#Loop through both folders
for folder_path in path:
    if folder_path.endswith("MRI"):
        for file_path in sorted(glob(folder_path + r"\*.jpg")):
            #Read the image and mask
            img = cv2.imread(file_path)
            
            #Resize the image and mask to the desired size
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            img = img / 255.0  # Normalize the image to [0, 1]

            #append image and filters to the lists
            images.append(img)

            # create Canny / Sobel / Laplacian filters
            f1_normalised = cv2.Canny((img*255).astype(np.uint8), 100, 200) / 255.0  
            f2 = cv2.Sobel((img*255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=5)
            f2_normalised = ( f2 - np.min(f2) ) / ( np.max(f2) - np.min(f2) )  # Normalize to [0, 1]
            f3 = cv2.Laplacian((img*255).astype(np.uint8), cv2.CV_64F)
            f3_normalised = ( f3 - np.min(f3) ) / ( np.max(f3) - np.min(f3) + 1e-8 )  # Normalize to [0, 1]

            # Stack the filters into a single 3-channel list
            list_3 = [f1_normalised, f2_normalised, f3_normalised]
            # make a combined list of the 3 filters and append it to the filter_3 list
            # the combined list will have the shape (256, 256, 3) , this is the 3 channel input image
            combined_filters = np.stack(list_3, axis=-1)
            filter_3.append(combined_filters)

X_mri = np.array(filter_3)
y_mri = np.array(images)
np.save("X_data_mri.npy", X_mri)
np.save("y_data_mri.npy", y_mri)
