import pandas as pd
import matplotlib.pyplot as plt

base_path = "./visualization/memory"
file_path = f"{base_path}/memory.csv"
data = pd.read_csv(file_path)

# Clean the Timestamp column by removing text within parentheses
data["Timestamp"] = data["Timestamp"].str.replace(r"\s\([^)]*\)", "", regex=True)

# Parse the cleaned timestamps
data["Timestamp"] = pd.to_datetime(
    data["Timestamp"], format="%a %b %d %Y %H:%M:%S GMT%z", errors="coerce"
)

# Drop rows with invalid timestamps
data_cleaned = data.dropna(subset=["Timestamp"])

# Calculate the overall average memory utilization across all pods
pod_columns = ["Worker2", "Worker1", "Worker0", "Master"]
overall_average_memory = data_cleaned[pod_columns].mean().mean()

# Plot memory utilization for each pod over time
plt.figure(figsize=(12, 6))
for pod in pod_columns:
    plt.plot(data_cleaned["Timestamp"], data_cleaned[pod], label=pod)

# Add a horizontal line for the overall average memory utilization
plt.axhline(y=overall_average_memory, color="red", linestyle="--", label=f"Avg utilization: {overall_average_memory:.2f}%")

# Set the y-axis range to 0-100%
plt.ylim(0, 100)

# Format the x-axis to show only time (HH:MM:SS)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M:%S"))

# Add labels and legend
plt.xlabel("Time")
plt.ylabel("Memory Utilization (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{base_path}/plot.png")
