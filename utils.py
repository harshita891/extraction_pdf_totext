import re

def extract_roll_number(filename):
   
    match = re.match(r"(\d+)", filename)
    return match.group(1) if match else "Unknown"
