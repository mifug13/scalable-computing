import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Load the cleaned data
base_path = "./visualization/nodes"
file_path = f"{base_path}/nodes.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Clean and parse the Timestamp column
data["Timestamp"] = data["Timestamp"].str.replace(r"\s\([^)]*\)", "", regex=True)
data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%a %b %d %Y %H:%M:%S GMT%z", errors="coerce")

# Convert Node_Count to numeric
data["Node_Count"] = pd.to_numeric(data["Node_Count"], errors="coerce")

# Add 1 hour to shift the timeline
data["Timestamp"] += timedelta(hours=1)

# Define scale events
scale_events = [
    "2024-12-05T14:06:50Z",
    "2024-12-05T14:20:36Z",
    "2024-12-05T14:28:52Z",
    "2024-12-05T15:05:26Z",
]
scale_event_labels = ["Scale Down", "Scale Up", "Scale Up", "Scale Down"]

# Add the cluster creation event
cluster_creation_event = pd.to_datetime("2024-12-05T13:57:00") + timedelta(hours=1)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(
    data["Timestamp"],
    data["Node_Count"],
    label="Number of Nodes",
    linestyle="-",  # Simple line
    color="black"
)

# Adding vertical lines with different colors for scale events
for event, label in zip(scale_events, scale_event_labels):
    event_time = pd.to_datetime(event) + timedelta(hours=1)
    color = "blue" if "Scale Up" in label else "red"
    plt.axvline(x=event_time, color=color, linestyle="--", label=f"Event: {label}")

# Adding the cluster creation event
plt.axvline(x=cluster_creation_event, color="green", linestyle="--", label="Event: Cluster Created")

# Setting x-axis range
plt.xlim(data["Timestamp"].min(), data["Timestamp"].max())

# Formatting the x-axis to show only time (HH:MM:SS)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M:%S"))

# Adding labels with Times New Roman font
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.legend()
plt.grid()

# Setting Times New Roman font for tick labels
plt.xticks()
plt.yticks()

plt.tight_layout()
plt.savefig(f"{base_path}/plot.png")