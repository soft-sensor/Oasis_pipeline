import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score

def eval_cal(y_true, y_pred):

    # Calculate precision
    precision = precision_score(y_true, y_pred, average='binary')  # For binary classification

    # Calculate recall
    recall = recall_score(y_true, y_pred, average='binary')

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, precision, recall

    # Print results
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")
    # print(f"Accuracy: {accuracy:.2f}")


def eval_og_pd(output_csv_path, original_csv_path):


   df = pd.read_csv(output_csv_path)
   test_df = pd.read_csv(original_csv_path)

   # Find rows in test_df where 'Patient_Name' is not in df
   missing_names = test_df[~test_df['Patient_Name'].isin(df['Patient_Name'])]

   # Print the names of rows being deleted
   print("Deleted rows with names:")
   print(missing_names['Patient_Name'].tolist())

   # Remove the rows from test_df
   test_df = test_df[test_df['Patient_Name'].isin(df['Patient_Name'])]


   test_df['FET Outcome'] = test_df['FET Outcome'].replace({"NEGATIVE":0,"Negative":0,"Positive":1,"POSITIVE":1,
                  "2":1}                )
   
   test_df['Final Birth Frozen']=test_df['Final Birth Frozen'].replace({"No Birth":0,"Unknown":0,"Live birth" : 1})

   df['FET Outcome'] = df['FET Outcome'].replace({"NEGATIVE":0,"Negative":0,"Positive":1,"POSITIVE":1,"Cancel":0,
                  "2":1,"Cancellation":0}                )

   df['Final Birth Frozen']=df['Final Birth Frozen'].replace({"No Birth":0,"No Birth":0,"NO BIRTH":0,"0.0":0, "Unknown":0,"Live birth" : 1})


   print("FET Evaluation :")

   accuracy_0, precision_0, recall_0 = eval_cal(test_df['FET Outcome'], df['FET Outcome'] )
   
   print("Final Birth Evaluation :")
      
   accuracy_1, precision_1, recall_1 =  eval_cal(test_df['Final Birth Frozen'], df['Final Birth Frozen'])

   return accuracy_0, precision_0, recall_0,accuracy_1, precision_1, recall_1
      

