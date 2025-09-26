#!/bin/bash

# Target directory where .txt files will be found
folder="./withoutconds" 

# Destination directory where .json files will be copied
destination="./txt_files"

# Check if the destination directory exists, create if it doesn't
if [ ! -d "$destination" ]; then
    echo "Creating destination directory: $destination"
    mkdir -p "$destination"
fi

# Initialize a counter for renaming
counter=1

# Find all .txt files in the current folder and its subfolders
find "$folder" -type f -name "*.txt" | while read -r file; do
	# Construct a new file name with the counter
	new_file_name="task_$counter.txt"
	# Copy the file to the destination with the new name
	cp "$file" "$destination/$new_file_name"
	# Set the file permissions to 644 (rw-r--r--)
	chmod 644 "$destination/$new_file_name"
	# Increment the counter
	((counter++))
done

echo "Copying complete. Total files copied: $((counter - 1))"