#!/bin/bash

# Check if LIBRILIGHT_PATH is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <LIBRILIGHT_PATH>"
    exit 1
fi

# List of archives
archives=("small.tar" "medium.tar" "large.tar")

# File containing filenames to extract
file_list="$1"

# Loop over each archive
for archive in "${archives[@]}"; do

    echo "Processing $archive..."
    tar -tf "$archive" | grep -Ff "$file_list" | grep -F .flac | tar -xf "$archive" -T -
done