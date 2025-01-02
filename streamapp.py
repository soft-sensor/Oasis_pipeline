import streamlit as st
import pandas as pd
import os
import zipfile
import pickle
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import subprocess
from eval import eval_og_pd


def load_patient_data_pickle(file_path):
    """Load patient data from a pickle file."""

    # Define the script path and arguments
    script_path = "preprocessing.py"

    # Run the script
    try:
        subprocess.run(["python", script_path], check=True)
        print("External script executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error while running the script: {e}")

    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_patient_data_pickle(data, file_path):
    """Save patient data to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def extract_images_to_pil(image_dir, zip_file, patient_names):
    """Extract images from ZIP and store them as PIL objects in memory."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(image_dir)

    patient_images = {name: [] for name in patient_names}

    # Match images to patients based on file prefixes
    for img_file in os.listdir(image_dir):
        for patient_name in patient_names:
            if img_file.startswith(patient_name):
                img_path = os.path.join(image_dir, img_file)
                try:
                    img_obj = Image.open(img_path)
                    patient_images[patient_name].append({"image": img_obj, "caption": img_file})
                except UnidentifiedImageError:
                    st.warning(f"Skipping invalid image file: {img_file}")
    return patient_images


# Define directories and files
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)

processed_data_file = "patient_data"
save_path="data.csv"

# Initialize session state
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "view_data_clicked" not in st.session_state:
    st.session_state.view_data_clicked = False
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False
if "evaluate_clicked" not in st.session_state:
    st.session_state.evaluate_clicked = False

st.title("Patient Data Uploader and Viewer")

# Step 1: Upload CSV
st.subheader("Upload a CSV File with Patient Data")
uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])


if uploaded_csv:
    try:
        # Save the uploaded file as "data.csv" in the specified directory
        with open(save_path, "wb") as f:
            f.write(uploaded_csv.getbuffer())

        
        st.success(f"CSV file saved as: {save_path}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Step 2: Upload ZIP
st.subheader("Upload a ZIP File Containing Patient Images")
uploaded_zip = st.file_uploader("Upload a ZIP file of images", type=["zip"])


if uploaded_zip:
    try:
        # Wrap the uploaded file in BytesIO
        with zipfile.ZipFile(BytesIO(uploaded_zip.read()), 'r') as zip_ref:
            # Extract all contents into the directory
            zip_ref.extractall(image_directory)
        
        # Display success message
        st.success(f"Files have been extracted to: {image_directory}")
        
        # List extracted files
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP archive.")
    except Exception as e:
        st.error(f"An error occurred: {e}")



print(f"Files have been extracted to '{image_directory}'")

        # Create patient data

# Process uploaded files
if uploaded_csv and uploaded_zip:
    try:
        # Process CSV
        #patient_data =load_patient_data_pickle(processed_data_file)
        st.session_state.patient_data = load_patient_data_pickle(processed_data_file)
        patient_names =  st.session_state.patient_data.keys()
        st.success("Patient data loaded successfully from the pickle file.")
        # Extract ZIP file and create PIL.Image objects


        # Save processed data


    except Exception as e:
        st.error(f"Error processing files: {e}")

# Step 3: View Processed Data
if st.session_state.patient_data:
    if st.button("View Processed Data"):
        st.session_state.view_data_clicked = True

# Step 4: Select and View Patient Data
if st.session_state.view_data_clicked:
    st.subheader("Select a Patient")
    patient_name = st.selectbox(
        "Choose a Patient Name",
        options=list(st.session_state.patient_data.keys()),
        index=0,
        key="patient_select"
    )

    if patient_name:
        st.session_state.selected_patient = patient_name

if st.session_state.selected_patient:
    patient_name = st.session_state.selected_patient
    patient_info = st.session_state.patient_data[patient_name]["patient_info"]
    patient_output = st.session_state.patient_data[patient_name]["output"]
    patient_images = st.session_state.patient_data[patient_name]["patient_images"]

    # Display patient info
    st.subheader(f"Patient Info: {patient_name}")
    st.write(patient_info)

    # Display patient output
    st.subheader("Patient Output")
    st.write(patient_output)

    st.subheader("Patient Images")
    for idx, img_data in enumerate(patient_images):
        image = img_data["image"]
        caption=img_data["caption"]
        st.image(image,caption)

if st.session_state.view_data_clicked:
    if st.button("Predict"):
        st.session_state.predict_clicked = True
        #st.session_state.view_data_clicked = False  # Hide previous data

# Step 6: Run Prediction Script and Display DataFrame
if st.session_state.predict_clicked:
    st.subheader("Prediction Results")
    try:
        # Run the external script for predictions
        prediction_script = "oasis_infer_script.py"
        subprocess.run(["python", prediction_script], check=True)
        st.success("Prediction script executed successfully!")

        # Load and display the DataFrame generated by the prediction script
        result_df = pd.read_csv("output.csv")  # Assume the script saves predictions here
        st.dataframe(result_df)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running prediction script: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


if st.session_state.predict_clicked:
        if st.button("Evaluate"):
            st.session_state.evaluate_clicked=True
            #st.session_state.predict_clicked = False
            #st.session_state.view_data_clicked = False

if st.session_state.evaluate_clicked:
    st.subheader("Evaluation Results")
    try:
        output_csv_path = "output.csv"
        original_csv_path = "data.csv"
        accuracy_0, precision_0, recall_0,accuracy_1, precision_1, recall_1 = eval_og_pd(output_csv_path, original_csv_path)

        st.write("-- FET Outcome --")
        st.text(f"Accuracy : {accuracy_0}, Precision : {precision_0}, Recall : {recall_0}")

        
        st.write("-- Final Birth Outcome --")
        st.text(f"Accuracy : {accuracy_1}, Precision : {precision_1}, Recall : {recall_1}")

    except:
        print("error")




