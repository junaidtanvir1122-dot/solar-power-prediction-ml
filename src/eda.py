import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/plant1_hourly.csv")

df["datetime"] = pd.to_datetime(df["datetime"])

plt.figure(figsize=(10, 6))

plt.scatter(
    df["irradiation"],
    df["ac_power"],
    alpha=0.5
)

plt.xlabel("Irradiation (kW/m²)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Irradiation")
plt.grid(True)

plt.savefig(
    "results/ac_power_vs_irradiation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(10, 6))

plt.scatter(
    df["ambient_temp"],
    df["module_temp"],
    c=df["irradiation"],
    alpha=0.6
)

plt.xlabel("Ambient Temperature (°C)")
plt.ylabel("Module Temperature (°C)")
plt.title("Module Temperature vs Ambient Temperature")
plt.colorbar(label="Irradiation (kW/m²)")
plt.grid(True)

plt.savefig(
    "results/module_vs_ambient.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(10, 6))

plt.scatter(
    df["dc_power"],
    df["ac_power"],
    alpha=0.5
)

plt.xlabel("DC Power (kW)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs DC Power")
plt.grid(True)

plt.savefig(
    "results/ac_vs_dc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


df["hour"] = df["datetime"].dt.hour

hourly_average = df.groupby("hour")["ac_power"].mean()

plt.figure(figsize=(10, 6))

plt.plot(
    hourly_average.index,
    hourly_average.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power (kW)")
plt.title("Average AC Power for Each Hour")
plt.xticks(range(24))
plt.grid(True)

plt.savefig(
    "results/average_ac_by_hour.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()