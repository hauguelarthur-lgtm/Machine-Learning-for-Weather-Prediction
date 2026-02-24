#!/usr/bin/env sh

cd era5_france_raw/ || exit 1

for archive in *.nc; do
    if file "$archive" | grep -q "Zip archive data"; then
        echo "Extracting archive: $archive"
        
        mkdir -p tmp_extract
        unzip -q "$archive" -d tmp_extract/
        
        base_name="${archive%.nc}"
        
        # Load the extracted files into the shell's positional parameters
        set -- tmp_extract/*
        
        # $# is the built-in variable defining the total number of positional parameters
        file_count=$#
        
        if [ "$file_count" -eq 1 ]; then
            # $1 calls the first positional parameter (the single extracted file)
            mv "$1" "$archive"
            echo " -> Replaced with 1 valid NetCDF."
        else
            counter=1
            # Iterate through all positional parameters seamlessly
            for extracted in "$@"; do
                new_name="${base_name}_part${counter}.nc"
                mv "$extracted" "$new_name"
                echo " -> Extracted fragmented dataset to: $new_name"
                counter=$((counter + 1))
            done
            rm "$archive"
        fi
        
        rm -r tmp_extract/
    else
        echo "Skipping $archive (Valid NetCDF detected)."
    fi
done