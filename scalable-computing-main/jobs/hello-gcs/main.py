# Import necessary modules
import os

# Get the GCS mount point from the environment variable
gcs_mount_point = os.getenv('GCS_MOUNT_POINT')

# Define the file path
file_path = os.path.join(gcs_mount_point, 'hello_world.txt')

# Write "Hello World!" to the file
try:
    with open(file_path, 'w') as file:
        file.write("Hello World!")
    print(f"'Hello World!' has been written to {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
