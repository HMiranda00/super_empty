# Super Empty Rig

A Blender addon that creates a simple but powerful rig for animating objects.

## Features

- Automatically creates a rig for selected objects, adapting to their size and orientation
- Works with both single objects and multiple objects
- Handles hierarchical relationships between objects (parent-child relationships)
- Provides intuitive controls for squash and stretch animations
- Preserves object transformations during rigging

## Installation

1. Download the latest release zip file
2. In Blender, go to Edit > Preferences > Add-ons
3. Click "Install..." and select the zip file
4. Enable the addon by checking the box next to "Rigging: Super Empty Rig"

## Usage

1. Select one or more mesh objects in the 3D viewport
2. Go to the "Super Empty" tab in the sidebar (press N if the sidebar is hidden)
3. Click the "Load Rig" button
4. The addon will create a rig that fits the selected objects and parent them to it

## Rig Controls

The Super Empty rig provides the following controls:

- **root**: The main control at the base of the rig. Use this to move the entire rig.
- **main**: The central control. Use this to move the object(s).
- **squash_top**: The control at the top. Move this up or down to create squash and stretch effects.

## Scenarios

The addon handles various scenarios intelligently:

1. **Single Object**: The rig adapts to the object's dimensions and orientation.
2. **Multiple Objects**:
   - **Scenario 2.1**: Multiple unrelated objects - The rig encompasses all objects.
   - **Scenario 2.2**: Parent object with children - The rig uses the parent's orientation and includes all children.
   - **Scenario 2.3**: Child object with parent - The rig uses the parent's orientation and includes siblings.

## Settings

You can customize the addon's behavior through the preferences panel:

1. Click the "Settings" button in the Super Empty panel
2. Adjust the following options:

### General Settings:
- **Adjust Shape Sizes**: Automatically scales the control shapes to match the object size
- **Use Direct Orientation Copy**: Applies the object's orientation directly to the rig
- **Height Factor**: Adjusts the height of the squash_top controller relative to the object height
- **Position via Root**: Keeps the rig at the origin and positions via the root bone (useful for animation)

### Orientation Settings:
- **Use Pose Space Correction**: Enables conversion between Blender's object space (Z-up) and bone space (Y-up)

### Debug Settings:
- **Enable Debug Logs**: Shows detailed debug messages in the Blender console

## New Features

Recent updates have added several significant improvements:

1. **Better Multi-Object Handling**: The rig now correctly adapts to multiple objects, considering their combined dimensions, positions, and orientations.

2. **Hierarchical Object Support**: The addon now intelligently handles parent-child relationships between objects, maintaining proper scaling and orientation.

3. **Local Space Dimensions**: For hierarchical scenarios, the rig calculates dimensions in the parent object's local space, ensuring correct sizing.

4. **Root Positioning Mode**: Option to keep the rig at the origin and use the root bone for positioning, which can be better for certain animation workflows.

5. **Rest Pose Application**: The addon now applies the pose as rest pose for the squash_top bone, making the rig more stable.

6. **Euler Rotation Mode**: All bones are converted to Euler rotation mode for better compatibility and predictability.

## Troubleshooting

If you encounter any issues:

1. Enable debug logs in the addon preferences
2. Open the Blender System Console (Window > Toggle System Console)
3. Perform the action that's causing problems
4. Check the console for detailed debug messages

## License

This addon is released under the MIT License. 