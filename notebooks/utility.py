import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import missingno as msn
import haversine as hs
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2_contingency, f_oneway,jarque_bera,probplot
from sklearn.preprocessing import PowerTransformer



def time_diff(row):

    order_picked = pd.to_timedelta(row['Time_Order_picked'])
    time_order =  pd.to_timedelta(row['Time_Orderd'])

    x = order_picked - time_order
    return x

def haversine_row(row):
    loc1 = (row['Restaurant_latitude'], row['Restaurant_longitude'])
    loc2 = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    return hs.haversine(loc1, loc2, unit=hs.Unit.KILOMETERS)


def basic_data_cleaning(df):

    df.drop(columns=['ID'],inplace=True)
    df['Restaurant_latitude'] = abs(df['Restaurant_latitude'].replace(0,np.nan))
    df['Restaurant_longitude'] = abs(df['Restaurant_longitude'].replace(0,np.nan))
    cols = ["Delivery_location_latitude","Delivery_location_longitude"]
    df.loc[df['Restaurant_latitude'].isnull(), cols] = np.nan

    columns = ["Delivery_person_Age",  "Delivery_person_Ratings","Time_Orderd", "Weatherconditions", "Road_traffic_density",
    "multiple_deliveries", "Festival", "City"]
    for i in columns:
        df[i] = df[i].replace('NaN ',np.nan)
    df['Weatherconditions'] = df['Weatherconditions'].replace("conditions NaN",np.nan)
    df['Weatherconditions'] = df['Weatherconditions'].apply(lambda x: x.split()[-1] if isinstance(x, str) else x)
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: x[-2:]).astype("float")

    # changing datatype 
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype("float")
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype("float")
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['multiple_deliveries'] = df['multiple_deliveries'].astype("float")

    # applying harversine distance
    df['Distance_res_to_loc_KM'] = df[['Restaurant_latitude', 'Restaurant_longitude','Delivery_location_latitude', 'Delivery_location_longitude']].apply(haversine_row,axis=1)
    df.insert(7,"Distance_res_loc_KM",df.pop('Distance_res_to_loc_KM'))

    # finding the time taken between restrurant & order pickup
    df["Time_res_pickup"] = (df[['Time_Order_picked','Time_Orderd']].apply(time_diff,axis=1).dt.total_seconds()/60).replace({-1425.:15., -1430.:10., -1435.:5.})
    df.insert(11,"Time_res_pickup",df.pop("Time_res_pickup"))
    df["order_time_hr"] = pd.to_datetime(df['Time_Orderd']).dt.hour
    df.insert(10,"order_time_hr",df.pop("order_time_hr"))
    df["time_of_day"] = pd.cut(
    df['order_time_hr'],
    bins=[0, 6, 12, 17, 20, 24],
    labels=['after midnight', 'morning', 'afternoon', 'evening', 'night'],
    right=False
    )
    df.insert(11,"time_of_day",df.pop("time_of_day"))

    # fixing values in columns:
    df['Road_traffic_density'] = df['Road_traffic_density'].str.strip().str.lower()
    df['Weatherconditions'] = df['Weatherconditions'].str.strip().str.lower()
    df['Type_of_order'] = df['Type_of_order'].str.strip().str.lower()
    df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip().str.lower()
    df['Festival'] = df['Festival'].str.strip().str.lower()
    df['City'] = df['City'].str.strip().str.lower()

    # dropping some rows
    df.drop(index=df[df['Delivery_person_Age'] <18].index,inplace=True)
    df.drop(index=df[df['Delivery_person_Ratings']>5].index,inplace=True)
    
    # creating rider city info
    rename_city = {"INDO":"Indore","BANG":"Bengaluru","COIMB":"Coimbatore","CHEN":"Chennai","HYD":"Hyderabad",
               "RANCHI":"Ranchi","MYS":"Mysore","DEH":"Dehradun","KOC":"Kochi","PUNE":"Pune","LUD":"Ludhiana",
               "LUDH":"Ludhiana","KNP":"Kanpur","MUM":"Mumbai","KOL":"Kolkata","JAP":"Jamshedpur","SUR":"Surat",
               "GOA":"Goa","AURG":"Aurangabad","AGR":"Agra","VAD":"Vadodara","ALH":"Allahabad","BHP":"Bhopal"}
    
    df['rider_city'] = df['Delivery_person_ID'].str.split("RES").str.get(0).replace(rename_city)
    df.insert(1,'rider_city',df.pop('rider_city'))
    df.insert(11,'order_dayname',df['Order_Date'].dt.day_name())
    df.insert(12,'order_in_weekend',np.where(df['order_dayname'].isin(['Saturday','Sunday']),1,0))
    

    df.drop(columns=['Delivery_person_ID'],inplace=True)

    # after doing EDA these columns are selected to be removed
    df.drop(columns=['rider_city','Type_of_order','order_in_weekend','order_time_hr','Time_res_pickup','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked'],inplace=True)
    
    return df

