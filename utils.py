import pandas as pd
import datetime

def convert_to_years(value):
    # First, attempt to interpret the value as an integer directly
    try:
        int_value = round(value)
        if int_value > 365:
            # Treat as days and convert to years, rounded to the nearest integer
            return round(int_value / 365.25)
        else:
            # Treat as years directly
            return int_value
    except (ValueError, TypeError):
        # If conversion to integer fails, continue with other checks
        pass

    # Check if the value is a pandas Timedelta
    if isinstance(value, pd.Timedelta):
        # Convert Timedelta to years and round to the nearest integer
        return round(value.days / 365.25)

    # Check if the value is a standard Python timedelta
    elif isinstance(value, datetime.timedelta):
        # Convert Timedelta to years and round to the nearest integer
        return round(value.days / 365.25)
    
    # Check if the value is a string
    elif isinstance(value, str):
        try:
            # Attempt to interpret the string as a pandas Timedelta
            timedelta_value = pd.to_timedelta(value)
            return round(timedelta_value.days / 365.25)
        except ValueError:
            print(f"Invalid format for string: {value}")
            return None

    # If the type is unsupported, print a message and return None
    else:
        print(f"Unsupported format: {type(value)}")
        return None
