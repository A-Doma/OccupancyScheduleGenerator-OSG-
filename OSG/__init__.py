"""
Your package's docstring with a description of its purpose, usage, etc.
"""

# Import the main entry point function or class from main.py
from .main import main

# Import relevant functions/classes from aggregator_functions.py that should be available to the user
from .aggregator_functions import get_initial_input_form_user,filter_sensor_data, occupancy_hourly_average, get_quantile_inputs_from_users, occupancy_status_profile, display_results

# Define your package's version
__version__ = '0.1.0'

# You can also include any necessary setup code for your package here
def _setup():
    # Setup code here, if necessary
    pass

# Run the setup code
_setup()
