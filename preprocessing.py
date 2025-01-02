

import pickle
import pandas as pd 
import os
from PIL import Image

# Save patient_data as a pickle file
def save_patient_data_pickle(patient_data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(patient_data, f)


#from transformers import TextStreamer

df = pd.read_csv("data.csv")
df = df.drop(axis=0,columns=['ET Outcome','No of SACs -Fresh','Miscarriage -Fresh','Final Birth Fresh','No of Embryo Fresh transferred ','Miscarriage -Frozen' ])
print(df.shape)

import re

def condense_spaces(input_string):
    # Use regex to replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', input_string).strip()


def string_match(name, filepath):
    
    name = condense_spaces(name)
    filepath = condense_spaces(filepath)
    # Replace underscores and hyphens with spaces
    output_string = filepath.replace("_", " ").replace("-", " ")

    # Print the result
    print("Output strinf of impath here : ",output_string.lower(), " Name :", name.lower())

    if name.lower() in output_string.lower():
        print("True")
        return True
    else:
        print("False")
        return False

def string_match(name, filepath):
    
    name = condense_spaces(name)
    filepath = condense_spaces(filepath)
    # Replace underscores and hyphens with spaces
    output_string = filepath.replace("_", " ").replace("-", " ")

    # Print the result
    print("Output strinf of impath here : ",output_string.lower(), " Name :", name.lower())

    if name.lower() in output_string.lower():
        print("True")
        return True
    else:
        print("False")
        return False

# Placeholder for the patient_data dictionary
patient_data = {}

# Function to append patient info and images
def append_patient_image(patient_id, image_obj, caption, df):
    # Check if the patient exists in the data
    if patient_id not in patient_data:
        # If patient is not in the data, add both the patient info and images
        print(f"Adding new patient {patient_id}")
        
        # Extract only the specific columns you want for the "output"
        #output_columns = ['Final Birth Fresh', 'Final Birth Frozen', 'Miscarriage -Frozen', 'Miscarriage -Fresh','FET Outcome','ET Outcome']
        output_columns = ['Final Birth Frozen','FET Outcome']
        patient_info = df[df['Patient_Name'] == patient_id].drop(labels=output_columns, axis=1).iloc[0].to_dict()
        patient_output = df[df['Patient_Name'] == patient_id][output_columns].iloc[0].to_dict()

        patient_data[patient_id] = {
            "patient_info": patient_info,  # Add patient info without the output columns
            "patient_images": [{"image": image_obj, "caption": caption}],  # Add the first image with caption
            "output": patient_output  # Add the specific output columns
        }
    else:
        # If the patient already exists, append images, regardless of whether info exists
        print(f"Appending image for patient {patient_id}")
        # Always append image even if the info is empty or the images list is empty
        patient_data[patient_id]["patient_images"].append({
            "image": image_obj,  # Store the PIL image object
            "caption": caption  # Store the filename as caption
        })

# Function to load patient info and images from the directory
def load_patient_data_from_directory(df, directory_path):
    for i in range(df.shape[0]):  # Iterate over each row in the dataframe
        patient_name = df['Patient_Name'].iloc[i]
        
        # Loop through the directory and match patient name with image filenames
        for impath in os.listdir(directory_path):
            #name = impath.split("_")[0]  # Assuming patient name is at the start of the filename
            if string_match(patient_name, impath):
                print(f"patient_name: {patient_name}, impath : {impath}")
                try:
                    # Open the image file and create a PIL image object
                    image_path = os.path.join(directory_path, impath)
                    image_obj = Image.open(image_path)

                    # Use the full filename as the caption
                    caption = impath  # Full filename as caption
                    append_patient_image(patient_name, image_obj, caption, df)
                except Exception as e:
                    print(f"Error processing image {impath}: {e}")

# Load and process patient data and images
df = df  # Ensure df is your dataframe object
directory_path = "uploaded_images/AI TRAIL DATA"  # Directory containing images

# Load and process patient data and images
load_patient_data_from_directory(df, directory_path)

save_patient_data_pickle(patient_data, "patient_data")