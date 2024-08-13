#!/bin/bash

# Array of source directories to search for .json files
folders=("./data/json_2.1.0/tests_seen" "./data/json_2.1.0/tests_unseen" "./data/json_2.1.0/train" "./data/json_2.1.0/valid_seen" "./data/json_2.1.0/valid_unseen")

# Destination directory where .json files will be copied
destination="./json_files"

# Check if the destination directory exists, create if it doesn't
if [ ! -d "$destination" ]; then
    echo "Creating destination directory: $destination"
    mkdir -p "$destination"
fi

# Initialize a counter for renaming
counter=1

# Iterate over each folder
for folder in "${folders[@]}"; do
    # Find all .json files in the current folder and its subfolders
    find "$folder" -type f -name "*.json" | while read -r file; do
        # Construct a new file name with the counter
        new_file_name="file_$counter.json"
        # Copy the file to the destination with the new name
        cp "$file" "$destination/$new_file_name"
		# Set the file permissions to 644 (rw-r--r--)
        chmod 644 "$destination/$new_file_name"
        # Increment the counter
        ((counter++))
    done
done

echo "Copying complete. Total files copied: $((counter - 1))"