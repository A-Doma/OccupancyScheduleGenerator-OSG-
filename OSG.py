import aggregator_funtions as ag_f
import ipywidgets as widgets
from IPython.display import display, clear_output

def start(path:str):
    global wd, output_area
    files = ag_f.get_initial_input_form_user(path)
    df_total = ag_f.filter_sensor_data(files)
    df_houses = ag_f.occupancy_hourly_average(df_total)
    wd = ag_f.get_quantile_inputs_from_users()
    
    # Extract individual widgets from the returned structure if needed
    dropdown_working, slider_working, dropdown_nonworking, slider_nonworking, dropdown_weekend, slider_weekend, dropdown_night_start, dropdown_night_end = wd
    
    # Define the update_results function
    def update_results(button=None):
        with output_area:
            clear_output(wait=True)
            df_final = ag_f.occupancy_status_profile(df_houses, wd)
            ag_f.display_results(df_final)

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
