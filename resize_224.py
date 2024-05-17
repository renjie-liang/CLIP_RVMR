import os
import cv2
from tqdm import tqdm

def resize_and_save_images(input_dir, output_dir, size=(224, 224)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Iterate over all subdirectories in the input directory
    for subdir in tqdm(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir)
        output_subdir_path = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir_path, exist_ok=True)
        
        # Iterate over all files in the subdirectory
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            
            # Read the image
            image = cv2.imread(file_path)
            
            # Resize the image
            resized_image = cv2.resize(image, size)
            
            # Save the resized image to the output directory
            output_file_path = os.path.join(output_subdir_path, file_name)
            cv2.imwrite(output_file_path, resized_image)

# Specify the input and output directories
input_directory = "/home/share/rjliang/Dataset/TVR/frames"
output_directory = "/home/share/rjliang/Dataset/TVR/frame_224"

# Call the function
resize_and_save_images(input_directory, output_directory)
