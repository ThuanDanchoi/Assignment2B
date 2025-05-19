"""
Geocoding script using OpenCage API

This script loads raw location names, normalizes them, and retrieves their
latitude and longitude using the OpenCage geocoding API. Results are saved to CSV.
"""

import pandas as pd
import requests
from tqdm import tqdm

# ==== API Key (replace with your own key if needed) ====
API_KEY = "42a0fa02e8b74101a28159cd6d79cb0b"

# ==== Load raw locations from CSV ====
df = pd.read_csv("../data/processed/locations_to_geocode.csv")

# ==== Normalize location string for geocoding ====
def normalize_address(addr):
    """
    Clean and standardize address format before API query.

    Args:
        addr (str): Raw address string.

    Returns:
        str: Normalized address string for geocoding.
    """
    addr = str(addr).upper().strip().replace("_", " ")
    addr = addr.replace(" OF ", " & ")
    return addr + ", Victoria, Australia"

# ==== Call OpenCage API ====
def geocode_opencage(address):
    """
    Query OpenCage Geocoding API to get coordinates for a given address.

    Args:
        address (str): Normalized address string.

    Returns:
        pd.Series: [latitude, longitude] or [None, None] if failed.
    """
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={API_KEY}"
    try:
        resp = requests.get(url)
        data = resp.json()
        if data["results"]:
            lat = data["results"][0]["geometry"]["lat"]
            lng = data["results"][0]["geometry"]["lng"]
            return pd.Series([lat, lng])
    except Exception as e:
        print("[ERROR] Geocoding failed:", e)
    return pd.Series([None, None])

# ==== Geocode all locations ====
tqdm.pandas(desc="Geocoding location")
df[["latitude", "longitude"]] = df["location"].progress_apply(
    lambda loc: geocode_opencage(normalize_address(loc))
)

# ==== Save results to CSV ====
df.to_csv("data/processed/locations_with_latlon.csv", index=False)

print(f"[INFO] File created: locations_with_latlon.csv")
print(f"[INFO] Total valid coordinates: {df['latitude'].notnull().sum()} / {len(df)}")
