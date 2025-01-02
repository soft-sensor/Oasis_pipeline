import pandas as pd 

#from transformers import TextStreamer


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

import pickle

# Save patient_data as a pickle file
def save_patient_data_pickle(patient_data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(patient_data, f)

def load_patient_data_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

##Adding output

import os
from PIL import Image


# Path to your pickle file
pickle_file_path = "patient_data"

# Check if the file exists
if os.path.exists(pickle_file_path):
      
      print(f"The file '{pickle_file_path}' exists.")
      patient_data = load_patient_data_pickle(pickle_file_path)


else:
        df = pd.read_csv("data.csv.csv")
        df = df.drop(axis=0,columns=['ET Outcome','No of SACs -Fresh','Miscarriage -Fresh','Final Birth Fresh','No of Embryo Fresh transferred ','Miscarriage -Frozen' ])
        print(df.shape)

        print(f"The file '{pickle_file_path}' does not exist.")


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
        directory_path = "AI TRAIL DATA"  # Directory containing images

        # Load and process patient data and images
        load_patient_data_from_directory(df, directory_path)

        save_patient_data_pickle(patient_data, "patient_data")

        # Print the resulting dictionary to see patient data and images
for patient_id, data in patient_data.items():
    print(f"Patient ID: {patient_id}")
    print("Patient Info:", data["patient_info"])
    print("Patient Output:", data["output"])  # Output columns
    for img in data["patient_images"]:
        print(f"Image Caption: {img['caption']} | Image Object: {img['image']}")



if True:
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "oasis_model_frozen", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 100,
        #device_map='cuda:1',
        #dtype = dtype,
        load_in_4bit = False,
    )
    FastVisionModel.for_inference(model) # Enable native 2x faster inference
pass


def generate_dynamic_patient_message(patient_info, captions, num_pairs):
    """
    Generate a dynamic patient message given the patient info, captions, and number of text-image pairs.
    
    Args:
        patient_info (dict): Dictionary containing the patient's details.
        captions (list): List of captions corresponding to each stage.
        num_pairs (int): Number of text-image pairs.
    
    Returns:
        dict: Structured message for dynamic patient message generation.
    """
    base_message = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',

                                        'text': f"""Given a patient's Day 5 Blastocyst Images at 3 different stages:
                                 {', '.join(captions)}, where Frozen embryos will be implanted and pregnancy can result from either of them.
                                 Image captions represent which stage they are it can be either in Thaw, ET or Frozen,.
                                 You are also provided with the patient's info as {patient_info}.
                                 Please observe the images and related information very carefully and determine these outcomes:
                                 1) FET Outcome 2) Final Birth Frozen """
                }
            ]
        }
    ]

    # Append text-image pairs dynamically
    for caption in enumerate(captions, start=1):
        base_message[0]['content'].append({'type': 'image', 'text': caption})

    return base_message

#def process_multiple_patients(patient_data):
"""
Process multiple patients' data by separating their info, captions, and images.

Args:
    patient_data (list): List of dictionaries containing patient info, images, and outcomes.

Returns:
    list: A list of structured data for each patient.
"""
#processed_data = []

Output=[]
patients=[]

for key,patient in enumerate(patient_data):
    patient_info = patient_data[patient].get('patient_info', {})
    captions = [entry['caption'] for entry in patient_data[patient].get('patient_images', [])]
    images = [entry['image'] for entry in patient_data[patient].get('patient_images', [])]
    outcomes = patient_data[patient].get('output', {})

    patients.append(patient)

    messages = generate_dynamic_patient_message(patient_info, captions, len(captions))

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        images,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    print("\nAfter training:\n  Patient no : ",key," and name ", patient )

    # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    # _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
    #                 use_cache = True, temperature = 0.5, min_p = 0.1)
    out = model.generate(**inputs,  max_new_tokens = 128, use_cache = True, temperature = 0.5, min_p = 0.1)

            # Example string
    #input_str = "Some text before >assistant<|end_header_id|>\n\nDesired text here and more text: keyword1, , keyword2\n\nkeyword3, keyword4\n\nkeyword5"
    input_str = tokenizer.batch_decode(out)[0]

    #print(f"Raw output : {input_str}")
    # Key to search
    key = ">assistant<|end_header_id|>"

    # Find the part after the key
    if key in input_str:
        result = input_str.split(key, 1)[1]  # Split once at the key and take the second part
        result=result.split("<|eot_id|>")[0]
        #result = result.split('[{')[0]
        #result = result.split('}]')
        print("res :",result)

        Output.append(result)

    else:
        print("Key not found")
        result = ""


import pandas as pd
import ast

# Example input: multiple lists represented as strings
data_strings = Output

# Initialize an empty list to hold all parsed dictionaries
all_data = []

# Process each string
for data_str in data_strings:
    # Clean the string and parse it into a Python list
    data_list = ast.literal_eval(data_str.strip())
    # Append the parsed list to the combined data
    all_data.extend(data_list)

# Convert the combined list of dictionaries to a DataFrame
df = pd.DataFrame(all_data)
df.insert(0, 'Patient_Name', patients)

#df['Patient_Name']=patients

# Display the DataFrame
#import ace_tools as tools; tools.display_dataframe_to_user(name="Combined Patient Outcomes DataFrame", dataframe=df)
print(df)

df.to_csv('output.csv',index=False)
