import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
base_path = "./visualization/cpu"
file_path = f"{base_path}/cpu.csv"
data = pd.read_csv(file_path)

# Clean the Timestamp column by removing text within parentheses
data["Timestamp"] = data["Timestamp"].str.replace(r"\s\([^)]*\)", "", regex=True)

# Parse the cleaned timestamps
data["Timestamp"] = pd.to_datetime(
    data["Timestamp"], format="%a %b %d %Y %H:%M:%S GMT%z", errors="coerce"
)

# Drop rows with invalid timestamps
data_cleaned = data.dropna(subset=["Timestamp"])

# Convert CPU utilization columns to numeric
pod_columns = ["Worker2", "Worker1", "Worker0", "Master"]
for column in pod_columns:
    data_cleaned[column] = pd.to_numeric(data_cleaned[column], errors="coerce")

# Calculate the average CPU utilization
overall_average = data_cleaned[pod_columns].mean().mean()

# Plotting CPU utilization for each pod over time with the average line
plt.figure(figsize=(12, 6))
for pod in pod_columns:
    plt.plot(data_cleaned["Timestamp"], data_cleaned[pod], label=pod)

# Adding a horizontal line for the overall average
plt.axhline(y=overall_average, color="red", linestyle="--", label=f"Avg utilization: {overall_average:.2f}%")

# Formatting the x-axis to show only time (HH:MM:SS)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M:%S"))

# Adding labels and legend
plt.xlabel("Time")
plt.ylabel("CPU Utilization (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{base_path}/plot.png")