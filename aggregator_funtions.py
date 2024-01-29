import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def get_input_form_user(path: str):
    """ Gets the path to the raw data from users and filter out the occupancy sesnor data.
    
    Args:
    - path: the path to the raw data
    if not valid the code will break and report an error
    
    Returns:
    - files: list of all the raw data files
    """
    def is_valid_path(path):
        return os.path.isfile(path) or os.path.isdir(path)
    
    if not is_valid_path(path):
        print(f"'{path_to_check}' is not a valid file or directory path.")
        return None
    else:
        files= os.listdir(old_path)
        os.chdir(old_path)
        return files
    
    
def filter_sensor_data(files: list): #todo: function can read both csv and parquet files
    """ Filter occupancy data.
    
    Args:
    - files: list of all the parquet files 
    
    Returns:
    - df_total: dataframe for all houses with the following columns (date_time, Identifier, day, hour, sensors with Occ data)
    """
    for file in files:
        df= pd.read_parquet(file)
        df_total= pd.DataFrame()
        for house in df.Identifier.unique():
            df_1 = df[df.Identifier == house]
            nan_columns = df_1.isna().all()
            active_columns = nan_columns[nan_columns == False].index.tolist()
            occ_colu= [item for item in active_columns if 'Occ' in item]
            occ= occ_colu.copy()
            if not occ==[]:
                column_extra= ["date_time", "Identifier"]
                occ_colu.extend(column_extra)
                df2= df_1[occ_colu]
                df2[occ] = df2[occ].astype(float)
                df2[occ] = df2[occ].replace({True: 1, False: 0})
                df2['date_time'] = pd.to_datetime(df2['date_time'])
                df2['hour'] = df2['date_time'].dt.hour
                df2['date']= df2['date_time'].dt.date 
                df_total= pd.concat([df_total, df2], ignore_index=True)
                
    print("Occupancy Data is filtered and now the aggregation process will start")
    return df_total
        
def occupancy_hourly_average(df_total: pd.DataFrame()):
    """ Starts the aggregation process with calculating the average reading for each day.
    
     Args:
    - df_total: filtered dataframe with only occupancy data
    
    Returns:
    - df_houses: dataframe for all houses with the following columns (date, hour, average_occ, Identifier, number_sensors)
    """
    df_house= df_total.groupby(['Identifier', 'date', 'hour']).mean()
    houses = [value[0] for value in df_house.index]
    houses_unqiue= list(set(houses))
    df_houses= pd.DataFrame()
    for house in houses_unqiue:
        df1= df_house.loc[house,:]
        df_1= df1.dropna(how='all', axis=1)
        df2= pd.DataFrame(df_1.mean(axis=1))
        df2['Identifier']= [house]*len(df2[0])
        df2['number_sensors']= [len(df_1.columns)]*len(df2[0])
        df2['average_occ']= df2[0]
        df2= df2.drop([0], axis=1)
        df_houses= pd.concat([df_houses, df2])
    df_houses= df_houses.reset_index()
    print("First level of Aggregation is done")
    return df_houses

def occupancy_status_profile(df_houses: pd.DataFrame(), metric_working: int, metric_nonworking: int, metric_weekend:int):
    """ convert the average occ to 0,1 depend on the selected metrics 

    
     Args:
    - df_houses: aggregated dataframe with average occupancy values
    - metric_working: int ranges from 0.1 to 1, refer to the quartile percentage used for comparing working hours (9 to 5)
    - metric_nonworking: int ranges from 0.1 to 1, refer to the quartile percentage used for comparing nonworking hours during the weekdays
    - metric_weekend: int ranges from 0.1 to 1, refer to the quartile percentage used for comparing hours during weekends.
    
    Returns:
    - df_final: dataframe for all houses with the following columns (date_time, Identifier, day, hour, sensors with Occ data)
    """
    #claculate the metrics
    df_houses['date'] = pd.to_datetime(df_houses['date'])
    df_houses['weekday'] = df_houses['date'].dt.weekday
    df_quantile= pd.DataFrame()
    quantile_data = []
    for house in df_houses['Identifier'].unique():
        df_h = df_houses[df_houses['Identifier'] == house]
        # For working hours on weekdays
        for hour in range(9, 18):  # From 9 to 17 inclusive
            working_data = df_h[(df_h['hour'] == hour) & (df_h['weekday'] < 5)]
            quantile = working_data['average_occ'].quantile(metric_working)
            quantile_data.append({'Identifier': house, 'hour': hour, 'quantile': quantile, 'type': 'working_hours'})

        # For non-working hours on weekdays
        for hour in list(range(0, 9)) + list(range(18, 24)):  # Before 9 and after 17
            nonworking_data = df_h[(df_h['hour'] == hour) & (df_h['weekday'] < 5)]
            quantile = nonworking_data['average_occ'].quantile(metric_nonworking)
            quantile_data.append({'Identifier': house, 'hour': hour, 'quantile': quantile, 'type': 'nonworking_hours'})

        # For hours during the weekend
        for hour in range(24):  # All day for weekends
            weekend_data = df_h[(df_h['hour'] == hour) & (df_h['weekday'] >= 5)]
            quantile = weekend_data['average_occ'].quantile(metric_weekend)
            quantile_data.append({'Identifier': house, 'hour': hour, 'quantile': quantile, 'type': 'weekend_hours'})
        dz= pd.DataFrame(quantile_data)
        df_quantile= pd.concat([df_quantile, dz], ignore_index=True)
    #compare and convert
    df_final= df_houses.merge(df_quantile, on=['Identifier', 'hour'])
    df_final['Occupancy']= df_comp['average_occ']>= df_comp['quantile']
    print("The aggregation is done")
    return df_final
