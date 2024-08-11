import os
import glob

# Define the directory where your .db files are stored.
# If the databases are in the current directory, you can use os.getcwd()
db_directory = os.getcwd()

# Pattern to match all .db files
db_pattern = os.path.join(db_directory, "*.db")

# List all .db files in the directory
db_files = glob.glob(db_pattern)

# Loop through the list and remove each file
for db_file in db_files:
    try:
        os.remove(db_file)
        print(f"Deleted: {db_file}")
    except Exception as e:
        print(f"Error deleting {db_file}: {e}")

print("All .db files have been removed.")
