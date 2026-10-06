import pandas as pd
from load_data import load_raw


def parse_datetime(df):
    df = df.copy()

    print("Original DATE_TIME examples:")
    print(df["DATE_TIME"].head())

    df["datetime"] = pd.to_datetime(df["DATE_TIME"])

    return df


def main():

    raw = load_raw()

    gen = parse_datetime(raw["gen1"])
    sensor = parse_datetime(raw["sensor1"])

    print("\nGeneration first timestamp:")
    print(gen["datetime"].min())

    print("Generation last timestamp:")
    print(gen["datetime"].max())

    print("\nSensor first timestamp:")
    print(sensor["datetime"].min())

    print("Sensor last timestamp:")
    print(sensor["datetime"].max())

    generation_hourly = (
        gen.groupby("datetime")[["AC_POWER", "DC_POWER"]]
        .sum()
        .reset_index()
    )

    sensor_clean = sensor[
        [
            "datetime",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]
    ].copy()

    merged = pd.merge(
        generation_hourly,
        sensor_clean,
        on="datetime",
        how="outer",
        indicator=True
    )

    only_generation = (merged["_merge"] == "left_only").sum()
    only_sensor = (merged["_merge"] == "right_only").sum()

    print("\nTimestamps only in generation:", only_generation)
    print("Timestamps only in sensor:", only_sensor)

    merged = merged.drop(columns=["_merge"])

    merged = merged.sort_values("datetime")
    merged = merged.set_index("datetime")

    hourly = merged.resample("1h").mean()

    hourly = hourly.reset_index()

    hourly = hourly.rename(
        columns={
            "AC_POWER": "ac_power",
            "DC_POWER": "dc_power",
            "AMBIENT_TEMPERATURE": "ambient_temp",
            "MODULE_TEMPERATURE": "module_temp",
            "IRRADIATION": "irradiation"
        }
    )

    output_columns = [
        "datetime",
        "ac_power",
        "dc_power",
        "ambient_temp",
        "module_temp",
        "irradiation"
    ]

    hourly = hourly[output_columns]

    missing_rows = hourly.isna().any(axis=1).sum()

    print("\nHourly rows:", len(hourly))
    print("Hourly rows with missing values:", missing_rows)

    hourly = hourly.dropna()

    hourly.to_csv(
        "data/plant1_hourly.csv",
        index=False
    )

    print("\nSaved:")
    print("data/plant1_hourly.csv")

    print("\nFinal data:")
    print(hourly.head())

    print("\nFinal shape:")
    print(hourly.shape)


if __name__ == "__main__":
    main()