import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
import matplotlib.pyplot as plt

def get_initial_input_form_user(path: str):
    """ Gets the path to the raw data from users and generate list of the files
    
    Args:
    - path: the path to the raw data
    if not valid the code will break and report an error
    
    Returns:
    - files: list of all the raw data files.
    
    """
    def is_valid_path(path):
        return os.path.isfile(path) or os.path.isdir(path)
    
    if not is_valid_path(path):
        print(f"'{path_to_check}' is not a valid file or directory path.")
        return None
    else:
        files= os.listdir(path)
        os.chdir(path)
        print(f"Total number of files= {len(files)}")
        print("The filtering process is starting now")
        return files
    
    
def filter_sensor_data(files: list):
    """ Filter occupancy data.
    
    Args:
    - files: list of all the parquet files 
    
    Returns:
    - df_total: dataframe for all houses with the following columns (date_time, Identifier, day, hour, sensors with Occ data)
    """
    for file in files:
        file_name, ext= os.path.splitext(file)
        if ext.lower() == '.parquet':
            df= pd.read_parquet(file)
        else:
            df= pd.read_csv(file)
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


def get_quantile_inputs_from_users():
    message_label = widgets.Label('Please choose different values for the 3 hours types')

    # Existing widgets for working, nonworking, and weekend hours
    dropdown_working = widgets.Dropdown(options=['working hours', 'nonworking hours', 'weekends hours'], description='Metric 1:')
    slider_working = widgets.IntSlider(value=10, min=0, max=100, step=5, description='Value 1:')

    dropdown_nonworking = widgets.Dropdown(options=['working hours', 'nonworking hours', 'weekends hours'], description='Metric 2:')
    slider_nonworking = widgets.IntSlider(value=10, min=0, max=100, step=5, description='Value 2:')

    dropdown_weekend = widgets.Dropdown(options=['working hours', 'nonworking hours', 'weekends hours'], description='Metric 3:')
    slider_weekend = widgets.IntSlider(value=10, min=0, max=100, step=5, description='Value 3:')

    # New widgets for night definition
    dropdown_night_start = widgets.Dropdown(options=list(range(24)), description='Night Start:')
    dropdown_night_end = widgets.Dropdown(options=list(range(24)), description='Night End:')

    def update_message(*args):
        selected_values = [dropdown_working.value, dropdown_nonworking.value, dropdown_weekend.value]
        if len(set(selected_values)) == 3:
            message_label.value = 'Selection complete. Please also define night hours.'
        else:
            message_label.value = 'Please choose different values for the 3 hours types'

    # Register the event
    dropdown_working.observe(update_message, names='value')
    dropdown_nonworking.observe(update_message, names='value')
    dropdown_weekend.observe(update_message, names='value')

    # Display the widgets
    display(message_label)
    display(dropdown_working, slider_working)
    display(dropdown_nonworking, slider_nonworking)
    display(dropdown_weekend, slider_weekend)
    display(widgets.HBox([dropdown_night_start, dropdown_night_end]))  # Display night start and end dropdowns side by side

    # You can return the widget objects if you need to access their values later
    return (dropdown_working, slider_working, dropdown_nonworking, slider_nonworking, dropdown_weekend, slider_weekend, dropdown_night_start, dropdown_night_end)
    
    
    
def occupancy_status_profile(df_houses: pd.DataFrame(), wd):
    """ convert the average occ to 0,1 depend on the selected metrics 

    
     Args:
    - df_houses: aggregated dataframe with average occupancy values
    - wd: the results chosen by users from the sliders 
    
    Returns:
    - df_final: dataframe for all houses with the following columns (date_time, Identifier, day, hour, sensors with Occ data)
    """
    #extract the metrics
    metric_values = {
    "working hours": wd[2 * [wd[i].value for i in range(0, 6, 2)].index("working hours") + 1].value,
    "nonworking hours": wd[2 * [wd[i].value for i in range(0, 6, 2)].index("nonworking hours") + 1].value,
    "weekends hours": wd[2 * [wd[i].value for i in range(0, 6, 2)].index("weekends hours") + 1].value
    }
    metric_working= metric_values['working hours']/100
    metric_nonworking= metric_values['nonworking hours']/100
    metric_weekend= metric_values['weekends hours']/100
    
    #calculate the mertics
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
    df_final['Occupancy']= df_final['average_occ']>= df_final['quantile']
    # Additional step for night hours
    night_start, night_end = night
    for index, row in df_final.iterrows():
        if night_start <= row['hour'] <= night_end or (night_start > night_end and not (night_end <= row['hour'] < night_start)):
            if row['average_occ'] > 0:
                df_final.at[index, 'Occupancy'] = True
    print("The aggregation is done")
    return df_final


def display_results(df_final:pd.DataFrame()):
    
    # Calculate the percentage of True and False values for each hour
    percentages = df_final.groupby('hour')['Occupancy'].value_counts(normalize=True).unstack().fillna(0)

    # Plotting
    percentages[True].plot(kind='line', figsize=(12, 6))
    plt.xlabel('Hour of the Day')
    plt.ylabel('Aggregated Occupied Probability')

    plt.xticks(rotation=0)
    plt.show()
    print(f"Occupied hours= {round(percentages[True].mean()*100,0)}%")
