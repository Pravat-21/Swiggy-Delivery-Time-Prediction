import pandas as pd
import requests
from pathlib import Path

# path for data
root_path = Path(__file__).parent
data_path = root_path / "data" / "raw" / "train.csv"

# prediction endpoint
predict_url = "http://127.0.0.1:8000//predict"

# sample row for testing the endpoint
sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
# remove the target column
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()
print(data)

# get the response from API
response = requests.post(url=predict_url,json=data)

print("The status code for response is", response.status_code)

def is_float(res_text):
    try:
        float(res_text)
        return True
    except Exception:
        return False

if response.status_code == 200:
    if is_float(response.text)==False :
        print(response.text)
    else:
        print(f"The prediction value by the API is {float(response.text):.2f} min")
else:
    print("Error:", response.status_code)