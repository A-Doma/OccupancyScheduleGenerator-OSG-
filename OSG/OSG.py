import pandas as pd
import os
import pyarrow.parquet as pq
import pyarrow as pa
import warnings
warnings.filterwarnings('ignore')
import ipywidgets as widgets
from IPython.display import display, clear_output

def get_initial_input_form_user(path: str, df_metadata: pd.DataFrame):
    """ Gets the path to the raw data from users and generate list of the files
    
    Args:
    - path: the path to the raw data
    - df_metadata: The metadata descripting the household characteristics.
    if not valid the code will break and report an error
    
    Returns:
    - filtered_files: list of all the raw data files with thermostat models includes built-in motion sensor.
    
    """
    def is_valid_path(path):
        return os.path.isfile(path) or os.path.isdir(path)
    
    if not is_valid_path(path):
        print(f"'{path}' is not a valid file or directory path.")
        return None, None
    else:
        model = ['ecobee4', 'ESTWVC', 'ecobee3', 'SmartSi']
        df_ilg = df_metadata[df_metadata['model'].isin(model)]
        files = list(df_ilg['identifier'])
        if not files:
            print("No illegible houses")
            return None, None
        files_path = [os.path.join(path, file) for file in os.listdir(path)]
        print(f"Total number of files= {len(files_path)}")
        print("The filtering process is starting now")
        return files_path, files

def filter_sensor_data(files_path: list, files, output_file="filtered_data.parquet"):
    """Filter occupancy data and save intermediate results to disk in chunks."""
    
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=len(files_path),
        description='Processing:',
        bar_style='',
        style={'bar_color': 'blue'},
        orientation='horizontal')
    display(progress_bar)
    
    output_path = os.path.join(os.getcwd(), output_file)
    
    # Delete existing file before writing
    if os.path.exists(output_path):
        os.remove(output_path)
    
    all_data = []
    
    for file in files_path:
        file_name, ext = os.path.splitext(file)
        if ext.lower() == '.parquet':
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file, dtype={'Identifier': 'category'})  # Reduce memory usage
        
        for house in df.Identifier.unique():
            if house in files:
                df_1 = df[df.Identifier == house].copy()
                occ_columns = [col for col in df_1.columns if 'Occ' in col]
                if occ_columns:
                    df_1 = df_1[['date_time', 'Identifier'] + occ_columns]
                    df_1[occ_columns] = df_1[occ_columns].astype(float)
                    df_1['date_time'] = pd.to_datetime(df_1['date_time'])
                    df_1['hour'] = df_1['date_time'].dt.hour
                    df_1['date'] = df_1['date_time'].dt.date 
                    all_data.append(df_1)

        # Process in chunks of 50 files
        if len(all_data) >= 50:
            df_batch = pd.concat(all_data, ignore_index=True)
            save_to_parquet(df_batch, output_path)
            all_data = []  # Free memory
            
        progress_bar.value += 1

    # Save remaining data
    if all_data:
        df_batch = pd.concat(all_data, ignore_index=True)
        save_to_parquet(df_batch, output_path)

    print("Occupancy Data is filtered and saved as a Parquet file.")
    return output_path  # Return file path instead of DataFrame


def save_to_parquet(df, output_path):
    """Helper function to append data to a Parquet file correctly."""
    table = pa.Table.from_pandas(df)

    if not os.path.exists(output_path):
        pq.write_table(table, output_path, compression="snappy")  # Create new file
    else:
        existing_table = pq.read_table(output_path)
        combined_table = pa.concat_tables([existing_table, table])  # Append new data
        pq.write_table(combined_table, output_path, compression="snappy")  # Overwrite file
        
def occupancy_hourly_average(df_total: pd.DataFrame):
    """ Starts the aggregation process with calculating the average reading for each day.
    
     Args:
    - df_total: filtered dataframe with only occupancy data
    
    Returns:
    - df_houses: dataframe for all houses with the following columns (date, hour, average_occ, Identifier, number_sensors)
    """
    df_total['number_sensors'] = df_total.filter(like='Occ').notna().sum(axis=1)
    df_total['average_occ'] = df_total.filter(like='Occ').mean(axis=1)
    
    df_houses = df_total.groupby(['Identifier', 'date', 'hour']).agg({
        'average_occ': 'mean',
        'number_sensors': 'max'
    }).reset_index()
    
    print("First level of Aggregation is done")
    return df_houses

def get_quantile_inputs_from_users():
    message_label = widgets.Label('Please choose different values for the 3 hours types')

    # Existing widgets for working, nonworking, and weekend hours
    dropdown_working = widgets.Dropdown(options=['working hours', 'nonworking hours', 'weekends hours'], description='Metric (1)')
    slider_working = widgets.IntSlider(value=10, min=0, max=100, step=5, description='Percentile (%)')
    slider_working.style.font_weight = 'bold'
    slider_working.style.font_size = '22px'
    
    dropdown_nonworking = widgets.Dropdown(options=['nonworking hours','working hours', 'weekends hours'], description='Metric (2):')
    slider_nonworking = widgets.IntSlider(value=10, min=0, max=100, step=5, description='Percentile (%)')
    slider_nonworking.style.font_weight = 'bold'
    slider_nonworking.style.font_size = '22px'
    
    dropdown_weekend = widgets.Dropdown(options=['weekends hours','working hours', 'nonworking hours'], description='Metric (3)')
    slider_weekend = widgets.IntSlider(value=10, min=0, max=100, step=5, description='Percentile (%)')
    slider_weekend.style.font_weight = 'bold'
    slider_weekend.style.font_size = '22px'
    
    # New widgets for night definition
    dropdown_night_start = widgets.Dropdown(options=list(range(24)), description='Night Start:')
    dropdown_night_end = widgets.Dropdown(options=list(range(24)), description='Night End:')
    dropdown_night_start.style.font_weight = 'bold'
    dropdown_night_start.style.font_size = '22px'
    dropdown_night_end.style.font_weight = 'bold'
    dropdown_night_end.style.font_size = '22px'

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

    # You can return the widget objects if you need to access their values later
    return (dropdown_working, slider_working, dropdown_nonworking, slider_nonworking, dropdown_weekend, slider_weekend, dropdown_night_start, dropdown_night_end)

def occupancy_status_profile(df_houses: pd.DataFrame, wd):
    """ Convert the average occ to 0,1 depend on the selected metrics 

     Args:
    - df_houses: aggregated dataframe with average occupancy values
    - wd: the results chosen by users from the sliders 
    
    Returns:
    - df_final: dataframe for all houses with the following columns (date_time, Identifier, day, hour, sensors with Occ data)
    """
    print("The final aggregation step just started")
    # Extract the metrics
    metric_values = {
        "working hours": wd[2 * [wd[i].value for i in range(0, 6, 2)].index("working hours") + 1].value,
        "nonworking hours": wd[2 * [wd[i].value for i in range(0, 6, 2)].index("nonworking hours") + 1].value,
        "weekends hours": wd[2 * [wd[i].value for i in range(0, 6, 2)].index("weekends hours") + 1].value
    }
    metric_working = metric_values['working hours'] / 100
    metric_nonworking = metric_values['nonworking hours'] / 100
    metric_weekend = metric_values['weekends hours'] / 100
    night = [wd[-2].value, wd[-1].value]
    
    # Calculate the metrics
    df_houses['date'] = pd.to_datetime(df_houses['date'])
    df_houses['weekday'] = df_houses['date'].dt.weekday
    df_houses['day_type'] = df_houses['weekday'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')
    quantile_data = []
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=len(df_houses['Identifier'].unique()),
        description='Generating Occupancy Schedules:',
        bar_style='',
        style={'bar_color': 'blue'},
        orientation='horizontal')
    progress_bar.style.font_weight = 'bold'
    progress_bar.style.font_size = '22px'
    display(progress_bar)
    
    for house in df_houses['Identifier'].unique():
        df_h = df_houses[df_houses['Identifier'] == house]
        # For working hours on weekdays
        for hour in range(9, 18):  # From 9 to 17 inclusive
            working_data = df_h[(df_h['hour'] == hour) & (df_h['weekday'] < 5)]
            quantile = working_data['average_occ'].quantile(metric_working)
            quantile_data.append({'Identifier': house, 'hour': hour, 'day_type': 'weekday', 'quantile': quantile, 'type': 'working_hours'})

        # For non-working hours on weekdays
        for hour in list(range(0, 9)) + list(range(18, 24)):  # Before 9 and after 17
            nonworking_data = df_h[(df_h['hour'] == hour) & (df_h['weekday'] < 5)]
            quantile = nonworking_data['average_occ'].quantile(metric_nonworking)
            quantile_data.append({'Identifier': house, 'hour': hour, 'day_type': 'weekday', 'quantile': quantile, 'type': 'nonworking_hours'})

        # For hours during the weekend
        for hour in range(24):  # All day for weekends
            weekend_data = df_h[(df_h['hour'] == hour) & (df_h['weekday'] >= 5)]
            quantile = weekend_data['average_occ'].quantile(metric_weekend)
            quantile_data.append({'Identifier': house, 'hour': hour, 'day_type': 'weekend', 'quantile': quantile, 'type': 'weekend_hours'})
        progress_bar.value += 1
    
    df_quantile = pd.DataFrame(quantile_data)
    
    # Compare and convert
    df_final = df_houses.merge(df_quantile, on=['Identifier', 'hour', 'day_type'])
    df_final['Occupancy'] = (df_final['average_occ'] >= df_final['quantile']).astype(int)
    
    # Additional step for night hours
    night_start, night_end = night
    if night_start < night_end:
        night_mask = (df_final['hour'] >= night_start) & (df_final['hour'] <= night_end)
    else:
        night_mask = (df_final['hour'] >= night_start) | (df_final['hour'] <= night_end)
    
    night_occupied_mask = df_final.groupby('date')['average_occ'].transform(lambda x: (x > 0).any())
    df_final.loc[night_mask & night_occupied_mask, 'Occupancy'] = 1

    df_final['date_time'] = df_final['date'] + pd.to_timedelta(df_final['hour'], unit='h')
    df_final = df_final.drop(['date', 'hour'], axis=1)
    df_final = df_final.sort_values(by=['Identifier', 'date_time'])
    df_final_leg = df_final[df_final['number_sensors'] >= 2]
    columns_order = ['date_time', 'Identifier'] + [col for col in df_final_leg.columns if col not in ['date_time', 'Identifier']]
    data_reordered = df_final_leg[columns_order]
    data_reordered.reset_index(inplace=True)
    data_reordered = data_reordered.drop(['index', 'weekday', 'type', 'average_occ'], axis=1)
    data_reordered = data_reordered.drop_duplicates()
    
    print("The aggregation is done")
    
    return data_reordered

import matplotlib.pyplot as plt

def display_results(df_final_leg: pd.DataFrame):
    # Calculate the percentage of 1s (occupied) for each hour
    percentages = df_final_leg.groupby(df_final_leg['date_time'].dt.hour)['Occupancy'].mean()

    # Plotting the percentage of occupied (1) values
    percentages.plot(kind='line', figsize=(10, 6))
    plt.xlabel('Hour of the Day')
    plt.ylabel('Aggregated Occupied Probability')
    plt.title('Aggregated Occupied Probability by Hour of the Day')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()
    
    print(f"Number of houses: {len(df_final_leg.Identifier.unique())} houses")
    print(f"Occupied hours: {round(percentages.mean() * 100, 0)}%")

def save_csv(df_final_leg):
    # Get the path to the main user environment (home directory)
    user_home_path = os.path.expanduser("~")
    # Create the "Occupancy_OSG" folder if it doesn't exist
    folder_path = os.path.join(user_home_path, "Occupancy_OSG")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Define the save path for the CSV file
    save_path = os.path.join(folder_path, "Final_profiles.csv")
    # Save the DataFrame to the specified path
    df_final_leg.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

def start(path: str, df_metadata: pd.DataFrame):
    """ Gets the path to the raw data and the metadata file for removing inadequate houses
    
    Args:
    - path: the path to the raw data
    - df_metadata: The metadata describing the household characteristics.
    if not valid the code will break and report an error
    
    Returns:
    - df_final: dataframe with the generated profiles.
    
    """
    global wd, output_area
    files, filter_files = get_initial_input_form_user(path, df_metadata)
    if files is None:
        return
    df_total = filter_sensor_data(files, filter_files)
    if df_total is None:
        return
    df_houses = occupancy_hourly_average(df_total)
    wd = get_quantile_inputs_from_users()
    
    # Extract individual widgets from the returned structure if needed
    dropdown_working, slider_working, dropdown_nonworking, slider_nonworking, dropdown_weekend, slider_weekend, dropdown_night_start, dropdown_night_end = wd
    
    # Define the update_results function
    def update_results(button=None):
        with output_area:
            clear_output(wait=True)
            df_final = occupancy_status_profile(df_houses, wd)
            display_results(df_final)
            save_button = widgets.Button(description="Save CSV", button_style='success')
            save_button.on_click(lambda b: save_csv(df_final))
            display(save_button)

    # Create a button that when clicked will run the update_results function
    start_button = widgets.Button(description="Start Analysis")
    start_button.on_click(update_results)
    start_button.style.font_weight = 'bold'
    start_button.style.font_size = '16px'

    # Create an output area for the results if it doesn't exist
    output_area = widgets.Output()

    # Display the message label and widgets
    display(dropdown_working, slider_working)
    display(dropdown_nonworking, slider_nonworking)
    display(dropdown_weekend, slider_weekend)
    display(widgets.HBox([dropdown_night_start, dropdown_night_end]))
    display(start_button, output_area)
