# Super Empty Rig

A Blender addon that loads a custom rig and parents selected objects to it.

## Features

- Automatically loads the super empty rig from a template
- Parents selected objects to the rig using bone parenting
- Adjusts the rig to match the size and position of the selected objects
- Organizes objects in collections for better scene management

## Installation

1. Download the addon zip file
2. In Blender, go to Edit > Preferences > Add-ons
3. Click "Install..." and select the downloaded zip file
4. Enable the addon by checking the box next to "Rigging: Super Empty Rig"

## Usage

1. Select one or more objects in your scene
2. Open the sidebar in the 3D View (press N if it's not visible)
3. Go to the "Super Empty" tab
4. Click "Load Super Empty Rig"
5. The rig will be loaded and the objects will be parented to it

## Notes

- The root bone will be positioned at the bottom of the selected objects
- The squash_top bone will be positioned at the top of the selected objects
- The bone shapes will be scaled to match the size of the selected objects
- A "Stretch To" constraint will be added to the main bone, targeting the squash_top bone

## Requirements

- Blender 3.0 or newer 