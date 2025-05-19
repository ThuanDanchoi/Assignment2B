"""
Prepare unique locations for geocoding

This script extracts all unique traffic sensor locations from the cleaned SCATS dataset
and saves them to a CSV file for batch geocoding.
"""

import pandas as pd

df = pd.read_csv("../data/processed/cleaned_scats_data.csv")

unique_locations = df["Location"].dropna().unique()

location_df = pd.DataFrame({"location": unique_locations})

location_df.to_csv("data/processed/locations_to_geocode.csv", index=False)

print("Created file locations_to_geocode.csv với", len(location_df), "locations")
