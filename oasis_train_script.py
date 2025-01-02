import pandas as pd 
df = pd.read_csv("train_df.csv")
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

##Adding output

import os
from PIL import Image

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
df = df.drop(axis=0,columns=['ET Outcome','No of SACs -Fresh','Miscarriage -Fresh','Final Birth Fresh','No of Embryo Fresh transferred '])

##Replace label variable with 0 or 1 for clear outputs 
##not tested, newly added
df['FET Outcome'] = df['FET Outcome'].replace({"NEGATIVE":0,"Negative":0,"Positive":1,"POSITIVE":1,"Cancel":0,
                  "2":1,"Cancellation":0}                )

df['Final Birth Frozen']=df['Final Birth Frozen'].replace({"No Birth":0,"No Birth":0,"NO BIRTH":0,"0.0":0, "Unknown":0,"Live birth" : 1})

print("After delete shape... ",df.shape)
# Load and process patient data and images
load_patient_data_from_directory(df, directory_path)

# Print the resulting dictionary to see patient data and images
for patient_id, data in patient_data.items():
    print(f"Patient ID: {patient_id}")
    print("Patient Info:", data["patient_info"])
    print("Patient Output:", data["output"])  # Output columns
    for img in data["patient_images"]:
        print(f"Image Caption: {img['caption']} | Image Object: {img['image']}")


##Finetuning of vision llama model

import os
from unsloth import FastVisionModel
import torch
#from datasets import load_dataset
#from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 1. Load the model

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = False,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules      = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_to_conversation(info, captions,images, patient_output):
    conversation = [
        { 
            "role": "user",
            "content": [
                {
                    "type": "text",  
                    "text": f"""Given a patient's Day 5 Blastocyst Images at 3 different stages:
                     1) Thaw, 2) ET, 3) Frozen, where Frozen embryos will be implanted and pregnancy may result from them.
                     Image captions represent which stage they are.
                     You are also provided with the patient's info as {info}. Please observe the images and related information very carefully and determine these outcomes:
                     output format:
                     1) FET Outcome 2) 'Final Birth Frozen''"""
                }
            ] + [
                # Add each caption and image sequentially
                item
                for caption,image in zip(captions,images)
                for item in [{"type": "text", "text": caption}, {"type": "image", "image" : image}]
            ]
        },
        { 
            "role": "assistant",
            "content": [
                {"type": "text", "text": patient_output}
            ]
        }
    ]
    return { "messages": conversation }

#converted_dataset = convert_to_conversation(info, captions, images, patient_output)

info = []
images = []
captions = []
patient_output = []  # List to store patient_output
converted_dataset = []

for patient_id, data in patient_data.items():
       #if patient_id == "Rani Puranik":
        print(f"Patient ID: {patient_id}")
        info.append(data["patient_info"])  # Store patient info
        patient_output.append(data["output"])  # Store patient output
        
        # Collect images and captions
        for img in data["patient_images"]:
            captions.append(img['caption'])
            images.append(img['image'])
        converted_dataset.append(convert_to_conversation(info, captions, images, patient_output))
        info=[]
        captions=[]
        patient_output=[]
        images=[]

print(len(converted_dataset)) # List to store patient_output




# 4. Training

FastVisionModel.for_training(model)

epochs = 50

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = -1,
        num_train_epochs=epochs,
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



model.save_pretrained("oasis_model_frozen")
tokenizer.save_pretrained("oasis_model_frozen")
print(f"Model saved at {epochs} epochs")

