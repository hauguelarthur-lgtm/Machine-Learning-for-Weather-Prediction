cd era5_land_ca_t2m/

for file in *.nc; do
    # Verify if the file is actually a ZIP archive to prevent corrupting valid NetCDFs
    if file "$file" | grep -q "Zip archive data"; then
        echo "Processing $file..."
        
        # -p extracts the content to standard output, avoiding the internal generic name
        unzip -p "$file" > "tmp_extract.nc"
        
        # Replace the ZIP file with the valid NetCDF file
        mv "tmp_extract.nc" "$file"
    else
        echo "Skipping $file (Already a valid NetCDF or different format)."
    fi
done
