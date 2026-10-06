import requests
import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 14.82,
    "longitude": 78.28,
    "start_date": "2020-05-15",
    "end_date": "2020-06-17",
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["hourly"])

print(df.head())
print("\nRows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

df.to_csv("data/plant1_openmeteo.csv", index=False)
print("\nSaved: data/plant1_openmeteo.csv")

plant = pd.read_csv("data/plant1_hourly.csv")
plant["datetime"] = pd.to_datetime(plant["datetime"])

df["datetime"] = pd.to_datetime(df["time"])

weather = df.rename(columns={
    "shortwave_radiation": "public_sw_radiation",
    "temperature_2m": "public_temp2m",
    "cloud_cover": "cloud_cover"
})

weather = weather[
    ["datetime", "public_sw_radiation", "public_temp2m", "cloud_cover"]
]

merged = pd.merge(
    plant,
    weather,
    on="datetime",
    how="left"
)

print("\nMerged data:")
print(merged.head())
print("\nMerged rows:", len(merged))
print("\nMissing values:")
print(merged.isna().sum())

merged.to_csv("data/plant1_merged.csv", index=False)
print("\nSaved: data/plant1_merged.csv")

print("\nUnit comparison:")
print("On-site irradiation unit: kW/m²")
print("Open-Meteo shortwave radiation unit: W/m²")

print("\nSample values:")
print(merged[
    ["datetime", "irradiation", "public_sw_radiation"]
].head(10).to_string(index=False))

three_days = merged[
    (merged["datetime"] >= "2020-05-15") &
    (merged["datetime"] < "2020-05-18")
].copy()

plt.figure(figsize=(10, 5))
plt.plot(
    three_days["datetime"],
    three_days["irradiation"],
    label="On-site Irradiation"
)
plt.plot(
    three_days["datetime"],
    three_days["public_sw_radiation"] / 1000,
    label="Open-Meteo Radiation / 1000"
)

plt.xlabel("Date and Time")
plt.ylabel("Radiation (kW/m²)")
plt.title("Three-Day Radiation Comparison")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/three_day_radiation_comparison.png", dpi=300)
plt.show()

print("\nPeak-hour verification:")

for date in ["2020-05-15", "2020-05-16", "2020-05-17"]:
    day = merged[merged["datetime"].dt.date.astype(str) == date]

    on_site_peak = day.loc[day["irradiation"].idxmax()]
    public_peak = day.loc[day["public_sw_radiation"].idxmax()]

    print(f"\nDate: {date}")
    print("On-site peak:", on_site_peak["datetime"],
          "Irradiation =", on_site_peak["irradiation"])
    print("Open-Meteo peak:", public_peak["datetime"],
          "Radiation =", public_peak["public_sw_radiation"])