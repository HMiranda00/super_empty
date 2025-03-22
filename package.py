import os
import zipfile
import json
import shutil
import sys
from datetime import datetime

def create_addon_zip():
    """Create a zip file of the addon for easy installation in Blender"""
    # Load the manifest
    with open('manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Create zip filename with version
    zip_filename = f"{manifest['name'].replace(' ', '_')}_{manifest['version']}.zip"
    
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