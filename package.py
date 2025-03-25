import os
import zipfile
import shutil
import sys
from datetime import datetime

def create_addon_zip():
    """Create a zip file of the addon for easy installation in Blender"""
    # Define the addon name and create zip filename with timestamp
    addon_name = "super_empty"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{addon_name}_{timestamp}.zip"
    
    # Define the addon directory (this will be the root in the zip)
    addon_dir = "super_empty"
    
    # Check if the addon directory exists
    if not os.path.exists(addon_dir):
        print(f"Error: Addon directory '{addon_dir}' not found.")
        return None
    
    # Create the zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from the addon directory
        for root, dirs, files in os.walk(addon_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Add the file to the zip with the correct structure
                # This preserves the directory structure inside the zip
                zipf.write(file_path)
    
    print(f"Addon packaged as {zip_filename}")
    print(f"The structure of files in the zip will be:")
    print(f"- {addon_dir}/")
    print(f"  - __init__.py")
    print(f"  - templates/")
    print(f"    - super_empty_template.blend")
    
    return zip_filename

if __name__ == "__main__":
    create_addon_zip() 