import bpy
import os
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import FloatProperty, BoolProperty

bl_info = {
    "name": "Super Empty Rig",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Super Empty",
    "description": "Load and parent objects to the super empty rig",
    "warning": "",
    "doc_url": "",
    "category": "Rigging",
}

# Constants to define rig properties
RIG_BASE_HEIGHT = 2.0  # Base height of the rig in meters (from root to squash_top when zeroed)

def log_debug(message):
    """Log debug message to console"""
    # Check if logs are enabled in addon preferences
    try:
        prefs = bpy.context.preferences.addons["super_empty"].preferences
        if prefs.enable_debug_logs:
            print(f"SUPER_EMPTY_DEBUG: {message}")
    except (AttributeError, KeyError):
        # If preferences are not available, we don't log anything
        pass

def get_template_path():
    """Returns the path to the template blend file"""
    addon_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(addon_path, "templates", "super_empty_template.blend")

def adjust_orientation_for_bone_space(orientation):
    """Adjust the orientation to account for bone space vs object space difference
    In Blender bones, the default orientation has Y as the primary axis (pointing up)
    while in object space Z is the up axis."""
    log_debug(f"Adjusting orientation for bone space, input: {orientation}")
    
    # Check if pose space correction is enabled in preferences
    prefs = bpy.context.preferences.addons["super_empty"].preferences
    if not prefs.use_pose_space_correction:
        log_debug("Pose space correction disabled by user preference")
        return orientation
    
    # COMPLETELY NEW REVISED APPROACH
    # Create a rotation matrix that maps object space to bone space
    
    # Convert quaternion to matrix
    obj_matrix = orientation.to_matrix()
    obj_matrix.normalize()  # Ensure there's no scale
    
    # Create a matrix that represents a 90-degree rotation in X
    # which converts Z-up (object space) to Y-up (bone space)
    conversion = Matrix.Rotation(math.radians(-90), 3, 'X')
    
    # Apply the conversion to the object's orientation matrix
    bone_matrix = conversion @ obj_matrix
    
    # Convert back to quaternion
    bone_quat = bone_matrix.to_quaternion()
    
    log_debug(f"Bone space orientation after conversion: {bone_quat}")
    
    return bone_quat

def create_orientation_from_normal(normal):
    """Create an orientation matrix from a normal vector"""
    normal = normal.normalized()
    
    log_debug(f"Creating orientation from normal: {normal}")
    
    # Find the axis with the smallest component
    min_ind = 0
    min_axis = abs(normal.x)
    
    if abs(normal.y) < min_axis:
        min_ind = 1
        min_axis = abs(normal.y)
    if abs(normal.z) < min_axis:
        min_ind = 2
    
    log_debug(f"Smallest component axis index: {min_ind}")
    
    # Create a vector perpendicular to the normal
    right = Vector((0, 0, 0))
    if min_ind == 0:
        right = Vector((normal.x, -normal.z, normal.y))
    elif min_ind == 1:
        right = Vector((-normal.z, normal.y, normal.x))
    elif min_ind == 2:
        right = Vector((-normal.y, normal.x, normal.z))
    
    right = right.normalized()
    
    # Create an up vector
    up = normal.cross(right)
    up = up.normalized()
    
    log_debug(f"Created basis vectors - Right: {right}, Up: {up}, Normal: {normal}")
    
    # Create the rotation matrix
    mat = Matrix.Identity(3)
    mat.col[0] = right
    mat.col[1] = up
    mat.col[2] = normal
    
    log_debug(f"Created orientation matrix: {mat}")
    
    return mat

def get_object_oriented_dimensions(obj):
    """
    Calculates the dimensions of the object considering its rotation.
    Returns the dimensions in world space, with orientation.
    """
    # Get the object's transformation matrix
    matrix_world = obj.matrix_world
    
    # Calculate dimensions using the oriented bounding box
    corners = [matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Extract min/max on each axis
    min_x = min(corner.x for corner in corners)
    max_x = max(corner.x for corner in corners)
    min_y = min(corner.y for corner in corners)
    max_y = max(corner.y for corner in corners)
    min_z = min(corner.z for corner in corners)
    max_z = max(corner.z for corner in corners)
    
    # Calculate dimensions
    dimensions = Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    
    # Calculate center
    center = Vector(((max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2))
    
    # FIXED: Revised method to calculate the lowest point
    # Extract the object's transformation to determine the "down" direction
    loc, rot, scale = obj.matrix_world.decompose()
    
    # Create a matrix with just the rotation (no scale or translation)
    rot_matrix = rot.to_matrix().to_4x4()
    
    # Get the "down" vector in world coordinates using the object's rotation
    # IMPORTANT: We're using (0,0,1) and not (0,0,-1) as before, and we'll invert it later
    # to ensure we're looking in the right direction
    world_down = rot_matrix @ Vector((0, 0, 1, 0))
    world_down = Vector((world_down.x, world_down.y, world_down.z)).normalized()
    
    # INVERT the vector to actually point downward
    world_down = -world_down
    
    log_debug(f"Object {obj.name} world down vector (corrected): {world_down}")
    
    # Find the lowest point in the world_down direction
    # IMPORTANT: Now we look for the MAXIMUM projection value, not the minimum
    # because the higher the projection on the "down" vector, the lower the point is
    lowest_point = None
    lowest_value = float('-inf')  # Start with -infinity to find maximum
    
    for corner in corners:
        # Project the corner in the down direction to find the "lowest"
        # The higher the value, the "lower" the point is
        proj_value = corner.dot(world_down)
        if proj_value > lowest_value:  # CHANGE: now we look for the MAXIMUM value
            lowest_value = proj_value
            lowest_point = corner
    
    if lowest_point:
        # Adjust the lowest point to be exactly at ground level
        # Calculate the center of the vertices that are on the bottom face
        tolerance = 0.001
        bottom_face_points = [p for p in corners if abs(p.dot(world_down) - lowest_value) < tolerance]
        
        if bottom_face_points:
            # Calculate the center of the bottom face
            bottom_x = sum(p.x for p in bottom_face_points) / len(bottom_face_points)
            bottom_y = sum(p.y for p in bottom_face_points) / len(bottom_face_points)
            bottom_z = sum(p.z for p in bottom_face_points) / len(bottom_face_points)
            bottom = Vector((bottom_x, bottom_y, bottom_z))
        else:
            # If we don't find enough points on the bottom face, use the lowest point
            bottom = lowest_point
    else:
        # Fallback to center with minimum Z
        bottom = Vector((center.x, center.y, min_z))
    
    log_debug(f"Object {obj.name} oriented dimensions: {dimensions}")
    log_debug(f"Object {obj.name} center: {center}")
    log_debug(f"Object {obj.name} bottom: {bottom}")
    
    return dimensions, center, bottom

def get_selection_bounds(context):
    """Get the bounds of selected objects using Blender's built-in functionality"""
    log_debug("Starting get_selection_bounds")
    
    # Store original cursor location and pivot point
    original_cursor = context.scene.cursor.location.copy()
    original_pivot_point = context.scene.tool_settings.transform_pivot_point
    
    log_debug(f"Original cursor: {original_cursor}, Original pivot: {original_pivot_point}")
    
    # Temporarily change the pivot point to Bounding Box Center
    context.scene.tool_settings.transform_pivot_point = 'BOUNDING_BOX_CENTER'
    
    # Store original selection
    original_selected = context.selected_objects.copy()
    original_active = context.view_layer.objects.active
    
    log_debug(f"Selected objects: {[obj.name for obj in original_selected]}")
    log_debug(f"Active object: {original_active.name if original_active else 'None'}")
    
    # Store original mode
    original_mode = None
    if original_active:
        original_mode = original_active.mode
        if original_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    
    # Execute cursor to selected - this places the cursor at the calculated center
    bpy.ops.view3d.snap_cursor_to_selected()
    
    # Get cursor position - this is our center
    center = context.scene.cursor.location.copy()
    log_debug(f"Center from cursor: {center}")
    
    # MODIFIED: Calculate dimensions considering object rotation
    dimensions = Vector((0, 0, 0))
    
    # Get dimensions of each selected object considering rotation
    all_dimensions = []
    min_point = Vector((float('inf'), float('inf'), float('inf')))
    max_point = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    # Calculate actual dimensions and collect face normals for orientation
    face_normals = []
    
    for obj in context.selected_objects:
        if obj.type == 'MESH':
            log_debug(f"Processing mesh object: {obj.name}")
            
            # Get dimensions considering rotation
            obj_dim, obj_center, obj_bottom = get_object_oriented_dimensions(obj)
            all_dimensions.append(obj_dim)
            
            # Update min/max points
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                
                min_point.x = min(min_point.x, world_corner.x)
                min_point.y = min(min_point.y, world_corner.y)
                min_point.z = min(min_point.z, world_corner.z)
                
                max_point.x = max(max_point.x, world_corner.x)
                max_point.y = max(max_point.y, world_corner.y)
                max_point.z = max(max_point.z, world_corner.z)
            
            # Collect face normals (for orientation calculation)
            if obj.data.polygons:
                log_debug(f"Object has {len(obj.data.polygons)} polygons")
                for poly in obj.data.polygons:
                    # Convert normal to world space
                    local_normal = poly.normal
                    world_normal = obj.matrix_world.to_3x3() @ local_normal
                    world_normal.normalize()
                    face_normals.append(world_normal)
                    log_debug(f"Polygon {poly.index} - Local normal: {local_normal}, World normal: {world_normal}")
    
    # Use bbox dimensions as fallback
    bbox_dimensions = max_point - min_point
    
    # For multiple objects, use the largest dimension on each axis
    if len(all_dimensions) > 1:
        dimensions.x = max(dim.x for dim in all_dimensions)
        dimensions.y = max(dim.y for dim in all_dimensions)
        dimensions.z = max(dim.z for dim in all_dimensions)
    elif len(all_dimensions) == 1:
        dimensions = all_dimensions[0]
    else:
        dimensions = bbox_dimensions
    
    # MODIFIED: Completely new calculation of the lowest point for multiple objects
    # We use the same projection logic as in the get_object_oriented_dimensions function
    if original_active and original_active.type == 'MESH':
        # Extract the active object's rotation to determine the "down" vector
        loc, rot, scale = original_active.matrix_world.decompose()
        rot_matrix = rot.to_matrix().to_4x4()
        
        # FIXED: Use the same corrected logic from get_object_oriented_dimensions
        world_down = rot_matrix @ Vector((0, 0, 1, 0))
        world_down = Vector((world_down.x, world_down.y, world_down.z)).normalized()
        world_down = -world_down  # Invert to point downward
        
        log_debug(f"Selection world down vector based on active object (corrected): {world_down}")
        
        # Find the lowest point in the world_down direction
        lowest_point = None
        lowest_value = float('-inf')  # Start with -inf to find maximum
        all_corners = []
        
        # Collect all corners from all objects
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                obj_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                all_corners.extend(obj_corners)
                
                # Find the lowest point
                for corner in obj_corners:
                    proj_value = corner.dot(world_down)
                    if proj_value > lowest_value:  # FIXED: Look for the MAXIMUM value
                        lowest_value = proj_value
                        lowest_point = corner
        
        if lowest_point and all_corners:
            # Find all points that are on the bottom face
            tolerance = 0.001
            bottom_face_points = [p for p in all_corners if abs(p.dot(world_down) - lowest_value) < tolerance]
            
            if bottom_face_points:
                # Calculate the center of the bottom face
                bottom_x = sum(p.x for p in bottom_face_points) / len(bottom_face_points)
                bottom_y = sum(p.y for p in bottom_face_points) / len(bottom_face_points)
                bottom_z = sum(p.z for p in bottom_face_points) / len(bottom_face_points)
                bottom = Vector((bottom_x, bottom_y, bottom_z))
                log_debug(f"Using average of {len(bottom_face_points)} bottom face points: {bottom}")
            else:
                # If we don't find enough points on the bottom face, use the lowest point
                bottom = lowest_point
                log_debug(f"Using lowest point: {bottom}")
        else:
            # Fallback to center with minimum Z
            bottom = Vector((center.x, center.y, min_point.z))
            log_debug(f"Using center-XY with min_z for bottom: {bottom}")
    else:
        # If we don't have an active object, use center with minimum Z
        bottom = Vector((center.x, center.y, min_point.z))
        log_debug(f"No active object - Using center-XY with min_z for bottom: {bottom}")
    
    log_debug(f"Calculated bounds - Min: {min_point}, Max: {max_point}")
    log_debug(f"Final Dimensions: {dimensions}")
    log_debug(f"Bottom: {bottom}")
    
    # Determine orientation based on average normal
    orientation = Quaternion((1, 0, 0, 0))  # Default identity orientation
    
    # MODIFIED: use the object's orientation directly instead of calculating from normals
    if original_active and original_active.type == 'MESH':
        log_debug("Using active object's orientation directly")
        
        # Get the object's rotation matrix
        rotation_matrix = original_active.matrix_world.to_3x3()
        
        # Remove scale from the matrix
        rotation_matrix.normalize()
        
        # Convert to quaternion
        orientation = rotation_matrix.to_quaternion()
        
        log_debug(f"Object rotation quaternion: {orientation}")
        
        # Also get and log Euler values for debugging
        if original_active.rotation_mode == 'QUATERNION':
            euler_rotation = original_active.rotation_quaternion.to_euler('XYZ')
        else:
            euler_rotation = original_active.rotation_euler
            
        log_debug(f"Object rotation euler (XYZ): {[math.degrees(angle) for angle in euler_rotation]}")
    
    elif face_normals:
        log_debug(f"No direct orientation, calculating from {len(face_normals)} face normals")
        
        # Calculate average normal
        avg_normal = Vector((0, 0, 0))
        for normal in face_normals:
            avg_normal += normal
        
        if avg_normal.length > 0:
            avg_normal.normalize()
            log_debug(f"Average normal: {avg_normal}")
            
            # Create orientation matrix from average normal
            orientation_matrix = create_orientation_from_normal(avg_normal)
            
            # Convert to quaternion
            orientation = orientation_matrix.to_quaternion()
            log_debug(f"Calculated orientation quaternion: {orientation}")
    
    # Test orientation with a simple vector
    test_vector = Vector((0, 0, 1))
    rotated_vector = orientation @ test_vector
    log_debug(f"Test - Vector (0,0,1) rotated by orientation becomes: {rotated_vector}")
    
    # Restore original cursor location and pivot point
    context.scene.cursor.location = original_cursor
    context.scene.tool_settings.transform_pivot_point = original_pivot_point
    
    # Restore original selection
    for obj in context.selected_objects:
        obj.select_set(False)
    
    for obj in original_selected:
        obj.select_set(True)
    
    context.view_layer.objects.active = original_active
    
    log_debug("Finished get_selection_bounds")
    return dimensions, center, bottom, orientation

def get_objects_hierarchy_info(context, selected_objects):
    """
    Analyzes the hierarchy relationships between selected objects
    and returns useful information for rig scaling and orientation.
    
    Now finds the topmost parent in the hierarchy.
    """
    # If there are no objects, return default values
    if not selected_objects:
        return None, False, [], None
    
    active_obj = context.view_layer.objects.active
    active_has_children = False
    objects_to_consider = []
    root_parent = None  # New variable to store the topmost parent
    
    # SCENARIO 1: If there are multiple objects, check if the active one has children
    if len(selected_objects) > 1 and active_obj:
        # Check if the active object has children
        active_has_children = any(obj.parent == active_obj for obj in bpy.data.objects)
        log_debug(f"Active object {active_obj.name} has children: {active_has_children}")
        
        # Find the topmost parent of the active object
        if active_obj.parent:
            # Trace back to find the topmost parent
            current_parent = active_obj.parent
            while current_parent.parent:
                current_parent = current_parent.parent
            root_parent = current_parent
            log_debug(f"Found topmost parent for active object: {root_parent.name}")
        
        # If the active object doesn't have children, we'll discard its orientation
        # and consider all selected objects for sizing and positioning
        if not active_has_children:
            objects_to_consider = selected_objects
            log_debug(f"Multiple objects without hierarchy - considering all {len(selected_objects)} selected objects")
    
    # SCENARIO 2: If we have only 1 selected object with children
    elif len(selected_objects) == 1 and active_obj:
        # Check if the object has children
        active_has_children = any(obj.parent == active_obj for obj in bpy.data.objects)
        
        if active_has_children:
            # The active object itself is the topmost parent in this case
            root_parent = active_obj
            log_debug(f"Active object is the topmost parent: {root_parent.name}")
            
            # Collect all descendants recursively
            descendants = []
            
            def collect_descendants(obj):
                children = [o for o in bpy.data.objects if o.parent == obj]
                for child in children:
                    descendants.append(child)
                    collect_descendants(child)
            
            collect_descendants(active_obj)
            objects_to_consider = [active_obj] + descendants
            log_debug(f"Single object with hierarchy: {active_obj.name}, found {len(descendants)} descendants")
        else:
            # SCENARIO 3: If the object has no children, check if it has a parent
            if active_obj.parent:
                # Trace back to find the topmost parent
                current_parent = active_obj.parent
                while current_parent.parent:
                    current_parent = current_parent.parent
                root_parent = current_parent
                log_debug(f"Found topmost parent: {root_parent.name}")
                
                # Collect the parent and all its descendants
                descendants = []
                
                def collect_descendants(obj):
                    children = [o for o in bpy.data.objects if o.parent == obj]
                    for child in children:
                        descendants.append(child)
                        collect_descendants(child)
                
                collect_descendants(root_parent)
                objects_to_consider = [root_parent] + descendants
                log_debug(f"Object with parent hierarchy: topmost parent: {root_parent.name}, found {len(descendants)} descendants")
                
                # Use the topmost parent as reference for orientation
                active_obj = root_parent
                active_has_children = True
            else:
                # Single object with no parent or children
                objects_to_consider = selected_objects
                root_parent = active_obj
    
    # For any other case, use the selected objects
    if not objects_to_consider:
        objects_to_consider = selected_objects
    
    # Filter to only mesh objects
    mesh_objects = [obj for obj in objects_to_consider if obj.type == 'MESH']
    log_debug(f"Final objects to consider: {len(mesh_objects)} mesh objects")
    
    return active_obj, active_has_children, mesh_objects, root_parent

def get_objects_combined_dimensions(mesh_objects):
    """
    Calculates the combined dimensions of all mesh objects considered
    to properly scale the rig.
    
    This function computes the bounding box that encompasses all objects together,
    regardless of their positions in space.
    """
    if not mesh_objects:
        return Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))
    
    # Initialize min/max points with extreme values
    min_point = Vector((float('inf'), float('inf'), float('inf')))
    max_point = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    # Collect all vertices from all objects in world space
    for obj in mesh_objects:
        # Use the object's bounding box transformed to world space
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            
            # Update min/max
            min_point.x = min(min_point.x, world_corner.x)
            min_point.y = min(min_point.y, world_corner.y)
            min_point.z = min(min_point.z, world_corner.z)
            
            max_point.x = max(max_point.x, world_corner.x)
            max_point.y = max(max_point.y, world_corner.y)
            max_point.z = max(max_point.z, world_corner.z)
    
    # Calculate dimensions
    dimensions = max_point - min_point
    
    # Calculate center
    center = (max_point + min_point) / 2
    
    # Bottom center (lowest point)
    bottom = Vector((center.x, center.y, min_point.z))
    
    log_debug(f"Combined boundingbox min: {min_point}, max: {max_point}")
    log_debug(f"Combined dimensions: {dimensions}")
    log_debug(f"Combined center: {center}")
    log_debug(f"Combined bottom: {bottom}")
    
    return dimensions, center, bottom

def get_objects_local_dimensions(parent_obj, child_objects):
    """
    Calculates the dimensions of a group of objects in the local space of the parent object.
    This is critical for object hierarchies, where we need to discount the parent's rotation
    to scale the rig correctly.
    
    Args:
        parent_obj: The parent object whose rotation should be discounted
        child_objects: List of objects (including the parent) to calculate dimensions for
    
    Returns:
        dimensions, center, bottom in the parent's local space
    """
    if not parent_obj or not child_objects:
        return Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))
    
    log_debug(f"Calculating local dimensions for parent: {parent_obj.name} with {len(child_objects)} objects")
    
    # Get the parent object's transformation matrix
    parent_matrix_world = parent_obj.matrix_world
    
    # Calculate the inverse matrix to transform to the parent's local space
    parent_matrix_world_inv = parent_matrix_world.inverted()
    
    # Initialize min/max points with extreme values
    min_point = Vector((float('inf'), float('inf'), float('inf')))
    max_point = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    # Collect all vertices from all objects transformed to the parent's local space
    for obj in child_objects:
        if obj.type == 'MESH':
            # For each vertex, transform to world space and then to the parent's local space
            for corner in obj.bound_box:
                # First transform to world space
                world_corner = obj.matrix_world @ Vector(corner)
                
                # Then transform to the parent's local space
                local_corner = parent_matrix_world_inv @ world_corner
                
                # Update min/max in the parent's local space
                min_point.x = min(min_point.x, local_corner.x)
                min_point.y = min(min_point.y, local_corner.y)
                min_point.z = min(min_point.z, local_corner.z)
                
                max_point.x = max(max_point.x, local_corner.x)
                max_point.y = max(max_point.y, local_corner.y)
                max_point.z = max(max_point.z, local_corner.z)
    
    # Calculate dimensions in the parent's local space
    dimensions = max_point - min_point
    
    # Calculate center in the parent's local space
    local_center = (max_point + min_point) / 2
    
    # Bottom center (lowest point) in local space
    local_bottom = Vector((local_center.x, local_center.y, min_point.z))
    
    # Transform the center and bottom back to world space
    world_center = parent_matrix_world @ local_center
    world_bottom = parent_matrix_world @ local_bottom
    
    log_debug(f"Local dimensions: {dimensions}")
    log_debug(f"Local center: {local_center}, World center: {world_center}")
    log_debug(f"Local bottom: {local_bottom}, World bottom: {world_bottom}")
    
    return dimensions, world_center, world_bottom

class SUPEREMPTY_OT_load_rig(Operator):
    """Load the Super Empty rig and parent selected objects to it"""
    bl_idname = "superempty.load_rig"
    bl_label = "Load Super Empty Rig"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Properties are no longer directly visible in the operator
    # but we still need to keep them to access preference values
    adjust_shapes: BoolProperty(
        name="Adjust Shape Sizes",
        description="Automatically adjust the size of bone shapes to match the object",
        default=True,
        options={'HIDDEN'}
    )
    
    use_direct_copy: BoolProperty(
        name="Use Direct Orientation Copy",
        description="Use a direct approach to copy the object's orientation to the rig",
        default=True,
        options={'HIDDEN'}
    )
    
    height_factor: FloatProperty(
        name="Height Factor",
        description="Adjusts the height of the squash_top controller (smaller values make the controller lower)",
        default=1.0,  # Default value is now 1
        min=0.1,
        max=2.0,
        step=0.1,
        options={'HIDDEN'}
    )
    
    # New property for the operator
    use_root_positioning: BoolProperty(
        name="Position via Root",
        description="Keeps the rig at the origin and does all positioning through the root bone in pose mode",
        default=False,
        options={'HIDDEN'}
    )
    
    def invoke(self, context, event):
        # Use addon preferences instead of scene properties
        addon_prefs = context.preferences.addons["super_empty"].preferences
        self.adjust_shapes = addon_prefs.adjust_shapes
        self.use_direct_copy = addon_prefs.use_direct_copy
        self.height_factor = addon_prefs.height_factor
        self.use_root_positioning = addon_prefs.use_root_positioning  # New property
        return self.execute(context)
    
    def execute(self, context):
        # We no longer need to save preferences to the scene
        
        log_debug("\n\n==== STARTING SUPER EMPTY RIG OPERATOR ====")
        
        # Check if there are selected objects
        if len(context.selected_objects) == 0:
            self.report({'ERROR'}, "No objects selected")
            return {'CANCELLED'}
        
        # Filter mesh objects
        original_mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not original_mesh_objects:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}
        
        # Store original selection
        original_selected = context.selected_objects.copy()
        original_active = context.view_layer.objects.active
        
        log_debug(f"Selected objects: {[obj.name for obj in original_selected]}")
        
        # NEW: Analyze the hierarchy of objects and determine the best scenario
        # Now also returns the topmost parent if exists
        active_obj, active_has_children, mesh_objects, root_parent = get_objects_hierarchy_info(context, original_selected)
        
        # Keep track of the root parent to parent it to the rig instead of selected objects
        objects_to_parent = []
        if root_parent:
            log_debug(f"Will parent topmost parent to rig: {root_parent.name}")
            objects_to_parent = [root_parent]
        else:
            objects_to_parent = original_selected
            log_debug(f"No topmost parent found, will parent selected objects to rig")
        
        # Check if we have mesh objects after hierarchy analysis
        if not mesh_objects:
            self.report({'ERROR'}, "No suitable mesh objects found after hierarchy analysis")
            return {'CANCELLED'}
        
        # Logs to help with diagnostics
        log_debug(f"Number of mesh_objects to consider: {len(mesh_objects)}")
        log_debug(f"Mesh objects to consider: {[obj.name for obj in mesh_objects]}")
        
        # Decide whether to consider orientation based on scenarios
        consider_orientation = True
        if len(original_selected) > 1 and not active_has_children:
            # Scenario 2.1: Multiple objects, but the active one has no children
            log_debug("Multiple objects selected but active has no children - ignoring orientation")
            consider_orientation = False
        
        # IMPORTANT: For multiple objects, use our specialized function
        # to calculate the combined dimensions of all objects
        multi_object_scenario = len(mesh_objects) > 1
        
        # Check if we're in a hierarchical scenario (2.2 or 2.3)
        hierarchical_scenario = active_has_children or (len(mesh_objects) == 1 and active_obj and active_obj.parent)
        
        # Use the root parent for orientation and dimensions in hierarchical scenarios
        if hierarchical_scenario and root_parent:
            # For hierarchical scenarios, calculate dimensions in the parent's local space
            log_debug(f"Using local space dimensions calculation for hierarchical scenario with root parent: {root_parent.name}")
            dimensions, center, bottom = get_objects_local_dimensions(root_parent, mesh_objects)
            
            # Use the root parent's orientation
            rotation_matrix = root_parent.matrix_world.to_3x3()
            rotation_matrix.normalize()
            orientation = rotation_matrix.to_quaternion()
            log_debug(f"Using orientation from root parent: {orientation}")
        elif multi_object_scenario:
            # Calculate combined dimensions directly - this is more precise for multiple objects
            combined_dimensions, combined_center, combined_bottom = get_objects_combined_dimensions(mesh_objects)
            log_debug("Using combined dimensions calculation for multiple objects")
            dimensions = combined_dimensions
            center = combined_center
            bottom = combined_bottom
            
            # Get the active object's orientation or identity if we're not considering orientation
            if consider_orientation and active_obj and active_obj.type == 'MESH':
                # Get the object's rotation matrix
                rotation_matrix = active_obj.matrix_world.to_3x3()
                # Remove scale from the matrix
                rotation_matrix.normalize()
                # Convert to quaternion
                orientation = rotation_matrix.to_quaternion()
                log_debug(f"Using orientation from active object: {orientation}")
            else:
                orientation = Quaternion((1, 0, 0, 0))  # Identity (no rotation)
                log_debug("Using identity orientation (no rotation)")
        else:
            # For a single object, continue using the existing logic
            # which works perfectly
            log_debug("Using standard calculation for single object")
            
            # Save original selection to restore later
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            
            # Select the objects determined by hierarchy analysis
            for obj in mesh_objects:
                obj.select_set(True)
                log_debug(f"Selected for computation: {obj.name}")
            
            # Ensure we have a valid active object
            if consider_orientation and active_obj and active_obj.type == 'MESH':
                context.view_layer.objects.active = active_obj
                log_debug(f"Set active object for orientation: {active_obj.name}")
            elif mesh_objects:
                context.view_layer.objects.active = mesh_objects[0]
                log_debug(f"Set first mesh object as active: {mesh_objects[0].name}")
            
            # Get dimensions, center, bottom and orientation using Blender's built-in tools
            dimensions, center, bottom, orientation = get_selection_bounds(context)
            
            # If we're not considering orientation, use identity quaternion
            if not consider_orientation:
                orientation = Quaternion((1, 0, 0, 0))
                log_debug("Using identity orientation (no rotation)")
            
            # Restore original selection
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            
            for obj in original_selected:
                obj.select_set(True)
            
            context.view_layer.objects.active = original_active
        
        # Log the values before modifying anything
        log_debug(f"BEFORE MODIFICATION - Dimensions: {dimensions}, Center: {center}, Bottom: {bottom}")
        log_debug(f"BEFORE MODIFICATION - Orientation: {orientation}")
        log_debug(f"Consider orientation: {consider_orientation}")
        
        # Load the template rig
        template_path = get_template_path()
        
        # Check if the template file exists
        if not os.path.exists(template_path):
            self.report({'ERROR'}, f"Template file not found at {template_path}")
            return {'CANCELLED'}
        
        log_debug(f"Loading template from: {template_path}")
        
        # Load the collection from the template
        with bpy.data.libraries.load(template_path, link=False) as (data_from, data_to):
            data_to.collections = [c for c in data_from.collections if c == "super_empty_template"]

        # Link the collection to the scene with a unique name
        rig_col = None
        for col in data_to.collections:
            if col is not None:
                # Generate a unique name for the collection
                base_name = col.name
                if base_name in bpy.context.scene.collection.children:
                    # Find a unique name by adding a numeric suffix
                    counter = 1
                    while f"{base_name}.{counter:03d}" in bpy.context.scene.collection.children:
                        counter += 1
                    # Rename the collection
                    col.name = f"{base_name}.{counter:03d}"
                    log_debug(f"Collection renamed to {col.name}")
                
                # Now it's safe to link the collection to the scene
                bpy.context.scene.collection.children.link(col)
                rig_col = col
                log_debug(f"Linked collection: {col.name}")

        # Find the armature
        armature = None
        for obj in rig_col.objects:
            if obj.type == 'ARMATURE':
                # Also create a unique name for the armature
                base_name = obj.name
                if base_name in bpy.data.objects:
                    # Find a unique name by adding a numeric suffix
                    counter = 1
                    while f"{base_name}.{counter:03d}" in bpy.data.objects:
                        counter += 1
                    # Rename the armature
                    obj.name = f"{base_name}.{counter:03d}"
                    log_debug(f"Armature renamed to {obj.name}")
                
                armature = obj
                break
        
        if not armature:
            self.report({'ERROR'}, "Armature not found in the imported collection")
            return {'CANCELLED'}
        
        log_debug(f"Found armature: {armature.name}")
        
        # Create the selected_objects collection if it doesn't exist
        selected_objects_col_name = f"selected_objects_{armature.name.split('.')[-1]}" if '.' in armature.name else "selected_objects"
        if selected_objects_col_name not in rig_col.children:
            selected_objects_col = bpy.data.collections.new(selected_objects_col_name)
            rig_col.children.link(selected_objects_col)
        else:
            selected_objects_col = rig_col.children[selected_objects_col_name]
        
        # Store original mode
        original_armature_mode = armature.mode
        
        # Step 1: Position the armature at the bottom position
        log_debug(f"STEP 1 - Positioning armature")
        
        if self.use_root_positioning:
            # New method: Keep armature at origin
            log_debug("Using root bone for positioning instead of moving armature")
            armature.location = Vector((0, 0, 0))
            
            if self.use_direct_copy:
                # Reset armature rotation to default if using root for positioning
                armature.rotation_mode = 'QUATERNION'
                armature.rotation_quaternion = Quaternion((1, 0, 0, 0))
                log_debug("Reset armature rotation to identity (using root for orientation)")
        else:
            # Original method: Position the armature directly
            # REFINEMENT: Precise armature positioning considering rotation
            # First, we set the armature orientation (before positioning it)
            if self.use_direct_copy:
                log_debug("Using direct copy of orientation to armature object")
                armature.rotation_mode = 'QUATERNION'
                armature.rotation_quaternion = orientation

            # Now, we calculate the offset needed to ensure that the rig's origin
            # is precisely at the lowest point
            # IMPORTANT: This advanced calculation takes into account the object's rotation
            if original_active and original_active.type == 'MESH':
                # Extract just the armature's rotation to calculate the offset
                arm_rot_matrix = armature.rotation_quaternion.to_matrix().to_4x4()
                
                # Get the local "down" vector of the rig (in armature space)
                # In the rig's local coordinates, it's (0, 0, 0) -> (0, 0, -1)
                # We need to compensate for this vector to align with the lowest point
                rig_origin = Vector((0, 0, 0))
                rig_down_local = Vector((0, 0, -1))
                
                # Calculate what would be the position of the origin point after rotation is applied
                # This is the offset we need to compensate for in positioning
                rig_origin_world = armature.location
                rig_down_world = armature.location + (arm_rot_matrix @ Vector((0, 0, -1, 0))).to_3d()
                
                log_debug(f"Rig origin world position before adjustment: {rig_origin_world}")
                log_debug(f"Rig down direction in world space: {rig_down_world - rig_origin_world}")
                
                # Set the final armature position
                armature.location = bottom
                
                log_debug(f"Armature final position: {armature.location}")
            else:
                # Simple method if we don't have an active object
                armature.location = bottom
        
        # Step 2: Enter pose mode to adjust bones
        bpy.context.view_layer.objects.active = armature
        original_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode='POSE')
        
        # Get required bones
        root_bone = armature.pose.bones.get('root')
        sub_root_bone = armature.pose.bones.get('sub_root')
        squash_top_bone = armature.pose.bones.get('squash_top')
        main_bone = armature.pose.bones.get('main')
        
        if not all([root_bone, sub_root_bone, squash_top_bone, main_bone]):
            self.report({'ERROR'}, "Required bones not found in the rig")
            bpy.ops.object.mode_set(mode=original_mode)
            return {'CANCELLED'}
        
        log_debug(f"Found required bones: root, sub_root, squash_top, main")
        
        # Position and orient via root bone if the option is enabled
        if self.use_root_positioning:
            log_debug("STEP 3 - Setting root bone position and orientation")
            
            # Position the root bone at the bottom
            root_bone.location = bottom
            
            # Apply rotation to the root bone
            root_bone.rotation_mode = 'QUATERNION'
            root_bone.rotation_quaternion = orientation
            
            log_debug(f"Root bone position set to: {root_bone.location}")
            log_debug(f"Root bone orientation set to: {root_bone.rotation_quaternion}")
        
        # Step 3: Reset transformations only for specific bones
        log_debug("STEP 3 - Resetting specific bone transformations")
        for bone_name in ['sub_root', 'squash_top']:
            bone = armature.pose.bones.get(bone_name)
            if bone:
                bone.location = Vector((0, 0, 0))
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler = Vector((0, 0, 0))
                bone.scale = Vector((1, 1, 1))
        
        # Step 4: Apply orientation to the sub_root bone only if not using direct copy
        if not self.use_root_positioning and not self.use_direct_copy:
            log_debug(f"STEP 4 - Setting sub_root orientation")
            # Adjust the orientation to account for bone space vs object space difference
            adjusted_orientation = adjust_orientation_for_bone_space(orientation)
            
            sub_root_bone.rotation_mode = 'QUATERNION'
            sub_root_bone.rotation_quaternion = adjusted_orientation
            
            # Log the actual value that was set
            log_debug(f"sub_root_bone rotation after setting adjusted orientation: {sub_root_bone.rotation_quaternion}")
        else:
            log_debug("Skipping sub_root orientation - using direct copy to armature instead")
        
        # Step 5: Position the squash_top to adjust the object height - Y AXIS
        # SIMPLIFIED: Use dimensions already calculated for multiple objects
        height = dimensions.z * self.height_factor
        
        # Ensure a minimum size
        if height < 0.001:
            height = 1.0
        
        log_debug(f"Using height for squash_top: {height}")
        
        # Adjust the squash_top position to match the object height
        # Regardless of whether the object is taller or shorter than the rig's base height
        height_adjustment = height - RIG_BASE_HEIGHT
        
        # We adjust the squash_top location to reflect the difference between the object height
        # and the rig's base height - allowing it to be negative for smaller objects
        log_debug(f"STEP 5 - Object height: {height}, Rig base height: {RIG_BASE_HEIGHT}")
        log_debug(f"Setting squash_top height on Y axis: {height_adjustment}")
        squash_top_bone.location = Vector((0, height_adjustment, 0))
        
        # Step 6: Scale the bone shapes to match the object dimensions - NOW NON-UNIFORM
        if self.adjust_shapes:
            log_debug(f"STEP 6 - Scaling bone shapes to match object dimensions")
            
            # SIMPLIFIED: Use calculated dimensions directly
            # Dimensions have already been correctly calculated for multiple objects
            x_scale = max(dimensions.x / 2.0, 0.1)  # X axis
            z_scale = max(dimensions.y / 2.0, 0.1)  # Y axis in obj space = Z in bone space
            y_scale = max(dimensions.z / 2.0, 0.1)  # Z axis in obj space = Y in bone space
            
            log_debug(f"Non-uniform shape scale: X={x_scale}, Y={y_scale}, Z={z_scale}")
            
            # Apply non-uniform scale on the main bone to cover the entire selection
            main_bone.custom_shape_scale_xyz = Vector((x_scale, y_scale, z_scale))
            
            # For other bones, use a uniform scale based on the largest dimension
            max_dim = max(dimensions.x, dimensions.y, dimensions.z)
            log_debug(f"Max single dimension for uniform scaling: {max_dim}")
                
            uniform_scale = max(max_dim / 2.0, 0.1)
            
            # Apply uniform scale to other bones
            for bone in armature.pose.bones:
                if bone.custom_shape and bone != main_bone:
                    bone.custom_shape_scale_xyz = Vector((uniform_scale, uniform_scale, uniform_scale))
        
        # Exit pose mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Step 7: Create collection structure and organize objects
        log_debug("STEP 7 - Creating new collection structure")
        
        # Get the name for the main collection (using the root parent's name or the first selected object)
        main_collection_name = "SuperEmpty"
        if root_parent:
            main_collection_name = root_parent.name
        elif len(original_selected) > 0:
            main_collection_name = original_selected[0].name
        
        log_debug(f"Using main collection name: {main_collection_name}")
        
        # Create the main collection if it doesn't exist or find a unique name
        if main_collection_name in bpy.context.scene.collection.children:
            # Find a unique name by adding a numeric suffix
            counter = 1
            while f"{main_collection_name}.{counter:03d}" in bpy.context.scene.collection.children:
                counter += 1
            main_collection_name = f"{main_collection_name}.{counter:03d}"
        
        main_collection = bpy.data.collections.new(main_collection_name)
        bpy.context.scene.collection.children.link(main_collection)
        
        # Create the _GEO subcollection
        geo_collection_name = f"{main_collection_name}_GEO"
        geo_collection = bpy.data.collections.new(geo_collection_name)
        main_collection.children.link(geo_collection)
        
        # Rename the armature to match the collection name structure
        armature_name = f"{main_collection_name}_RIG"
        armature.name = armature_name
        log_debug(f"Renamed armature to: {armature.name}")
        
        # Move the armature to the main collection
        for col in list(armature.users_collection):
            col.objects.unlink(armature)
        main_collection.objects.link(armature)
        log_debug(f"Moved armature to main collection: {main_collection.name}")
        
        # Get all objects to move to the GEO collection
        # This includes original selected objects and ALL of their descendants
        all_objects_to_move = set()
        
        # Helper function to collect object and all its descendants
        def collect_object_and_descendants(obj, collection):
            collection.add(obj)
            for child in bpy.data.objects:
                if child.parent == obj:
                    collect_object_and_descendants(child, collection)
        
        # Collect all selected objects and their descendants
        for obj in original_selected:
            collect_object_and_descendants(obj, all_objects_to_move)
        
        log_debug(f"Total objects to move to GEO collection: {len(all_objects_to_move)}")
        
        # Move all collected objects to the _GEO collection
        for obj in all_objects_to_move:
            # Remove from current collections
            for col in list(obj.users_collection):
                col.objects.unlink(obj)
            # Add to our GEO collection
            geo_collection.objects.link(obj)
            log_debug(f"Moved object to GEO collection: {obj.name}")
        
        # Store the name of the template collection before removing it
        template_col_name = ""
        if rig_col:
            template_col_name = rig_col.name
            
            # First make sure no objects are left in the collection
            for obj in list(rig_col.objects):
                rig_col.objects.unlink(obj)
            
            # Also unlink any child collections
            for child_col in list(rig_col.children):
                rig_col.children.unlink(child_col)
            
            # Then remove the collection from the scene
            bpy.context.scene.collection.children.unlink(rig_col)
            
            # Finally remove from Blender's data if it's not used elsewhere
            if rig_col.users == 0:
                bpy.data.collections.remove(rig_col)
                
            log_debug(f"Removed template collection from scene: {template_col_name}")
        
        # Step 8: Add the stretch constraint
        log_debug("STEP 8 - Adding stretch constraint")
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        
        # Remove any existing stretch constraint
        for c in main_bone.constraints:
            if c.type == 'STRETCH_TO':
                main_bone.constraints.remove(c)
        
        # Add a new stretch constraint
        stretch_const = main_bone.constraints.new('STRETCH_TO')
        stretch_const.target = armature
        stretch_const.subtarget = 'squash_top'
        
        # Always use the actual object height for rest_length
        stretch_const.rest_length = height
        
        stretch_const.volume = 'NO_VOLUME'
        
        # Exit pose mode back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Step 9 (FINAL STEP): Parent the objects to the main bone
        log_debug("STEP 9 (FINAL) - Parenting objects to main bone")
        # Ensure all objects are deselected
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
            
        # Select the armature first
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        
        # Enter pose mode
        bpy.ops.object.mode_set(mode='POSE')
        
        # Clear all bone selections
        for bone in armature.pose.bones:
            bone.bone.select = False
        
        # Select the main bone for parenting
        main_bone.bone.select = True
        armature.data.bones.active = main_bone.bone
        
        # Exit to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Now select the objects to be parented (either the root parent or original selected)
        objects_parented = False
        for obj in objects_to_parent:
            obj.select_set(True)
            objects_parented = True
        
        if objects_parented:
            # Parent the objects to the main bone as the very last step
            bpy.ops.object.parent_set(type='BONE', keep_transform=True)
            
            # Log the objects' transformations after parenting
            log_debug("After parenting:")
            for obj in objects_to_parent:
                log_debug(f"Object {obj.name} - Location: {obj.location}, Rotation: {obj.rotation_euler if obj.rotation_mode == 'XYZ' else obj.rotation_quaternion}")
        else:
            log_debug("No objects to parent - skipping parenting step")
        
        # Select the armature at the end
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        
        # Log final state of important bones
        log_debug("Final bone states:")
        bpy.ops.object.mode_set(mode='POSE')
        log_debug(f"Root bone - Location: {root_bone.location}, Rotation: {root_bone.rotation_quaternion if root_bone.rotation_mode == 'QUATERNION' else root_bone.rotation_euler}")
        log_debug(f"Sub_root bone - Location: {sub_root_bone.location}, Rotation: {sub_root_bone.rotation_quaternion if sub_root_bone.rotation_mode == 'QUATERNION' else sub_root_bone.rotation_euler}")
        log_debug(f"Main bone - Location: {main_bone.location}, Rotation: {main_bone.rotation_quaternion if main_bone.rotation_mode == 'QUATERNION' else main_bone.rotation_euler}")
        log_debug(f"Squash_top bone - Location: {squash_top_bone.location}, Rotation: {squash_top_bone.rotation_quaternion if squash_top_bone.rotation_mode == 'QUATERNION' else squash_top_bone.rotation_euler}")
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # New step at the end: Apply pose as rest pose for the squash_top
        log_debug("Applying pose as rest pose for squash_top bone")
        
        # First save selected objects
        selected_objects = context.selected_objects.copy()
        active_object = context.view_layer.objects.active
        
        # Ensure the armature is selected and active
        for obj in context.selected_objects:
            obj.select_set(False)
            
        armature.select_set(True)
        context.view_layer.objects.active = armature
        
        # Enter pose mode and select only the squash_top bone
        bpy.ops.object.mode_set(mode='POSE')
        
        # Deselect all bones
        for bone in armature.pose.bones:
            bone.bone.select = False
        
        # Select only squash_top
        squash_top_bone.bone.select = True
        armature.data.bones.active = squash_top_bone.bone
        
        # Apply pose as rest pose (only for squash_top)
        bpy.ops.pose.armature_apply(selected=True)
        
        # Convert all rotations to Euler
        for bone in armature.pose.bones:
            if bone.rotation_mode != 'XYZ':
                # Save current rotation
                current_rotation = bone.rotation_quaternion if bone.rotation_mode == 'QUATERNION' else bone.rotation_axis_angle
                
                # Change mode to Euler
                bone.rotation_mode = 'XYZ'
                
                # Conversion happens automatically when changing the mode
                log_debug(f"Converted bone {bone.name} rotation from {bone.rotation_mode} to XYZ Euler")
        
        # Return to object mode and restore selection
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Restore original selection
        for obj in context.selected_objects:
            obj.select_set(False)
            
        for obj in selected_objects:
            obj.select_set(True)
            
        context.view_layer.objects.active = active_object
        
        log_debug("==== COMPLETED SUPER EMPTY RIG OPERATOR ====\n")
        
        self.report({'INFO'}, "Rig loaded and objects parented successfully")
        return {'FINISHED'}


class SUPEREMPTY_PT_panel(Panel):
    """Panel for the Super Empty Rig addon"""
    bl_label = "Super Empty Rig"
    bl_idname = "SUPEREMPTY_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Super Empty"
    
    def draw(self, context):
        layout = self.layout
        
        # Main section with addon information
        box = layout.box()
        box.label(text="Super Empty Rig")
        box.label(text="Load a rig and parent objects")
        
        # Just the button to load the rig
        row = layout.row()
        row.scale_y = 1.5  # Slightly larger button for emphasis
        row.operator(SUPEREMPTY_OT_load_rig.bl_idname, text="Load Rig")
        
        # Button to access addon preferences
        prefs_row = layout.row()
        prefs_row.operator("preferences.addon_show", text="Settings").module="super_empty"

# We no longer need to store properties in the scene
def register_scene_properties():
    # Keep this function empty for compatibility
    pass

def unregister_scene_properties():
    # Keep this function empty for compatibility
    pass

class SuperEmptyPreferences(AddonPreferences):
    """Addon preferences for Super Empty Rig"""
    bl_idname = "super_empty"
    
    # Transferring operator options to preferences
    adjust_shapes: BoolProperty(
        name="Adjust Shape Sizes",
        description="Automatically adjust the size of bone shapes to match the object",
        default=True
    )
    
    use_direct_copy: BoolProperty(
        name="Use Direct Orientation Copy",
        description="Apply orientation directly to the armature instead of the sub_root bone",
        default=True
    )
    
    height_factor: FloatProperty(
        name="Height Factor",
        description="Adjusts the height of the squash_top controller (smaller values make the controller lower)",
        default=1.0,  # Default value is now 1
        min=0.1,
        max=2.0,
        step=0.1
    )
    
    # New option for root positioning
    use_root_positioning: BoolProperty(
        name="Position via Root",
        description="Keeps the rig at the origin and does all positioning through the root bone in pose mode",
        default=False
    )
    
    # Original options
    use_pose_space_correction: BoolProperty(
        name="Use Pose Space Correction",
        description="Enable orientation correction for Blender's bone space (Y-up vs Z-up)",
        default=False  # MODIFIED: now off by default
    )
    
    enable_debug_logs: BoolProperty(
        name="Enable Debug Logs",
        description="Show detailed debug messages in the Blender console",
        default=False
    )
    
    def draw(self, context):
        layout = self.layout
        
        # New general settings section
        box = layout.box()
        box.label(text="General Settings:")
        box.prop(self, "adjust_shapes")
        box.prop(self, "use_direct_copy")
        box.prop(self, "height_factor")
        box.prop(self, "use_root_positioning")
        
        # Orientation settings section
        box = layout.box()
        box.label(text="Orientation Settings:")
        box.prop(self, "use_pose_space_correction")
        box.label(text="Off: Preserves the original object rotation")
        box.label(text="On: Attempts to convert between coordinate spaces (Y-up vs Z-up)")
        
        # Debug section
        box = layout.box()
        box.label(text="Debug Settings:")
        box.prop(self, "enable_debug_logs")
        box.label(text="On: Shows detailed messages in the console for troubleshooting")
        
        # Informational note about viewing logs
        if self.enable_debug_logs:
            info_box = layout.box()
            info_box.label(text="To view logs: Window > Toggle System Console")


classes = (
    SUPEREMPTY_OT_load_rig,
    SUPEREMPTY_PT_panel,
    SuperEmptyPreferences,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Store preferences also in the scene for panel use
    register_scene_properties()

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Remove preferences from the scene
    unregister_scene_properties()

if __name__ == "__main__":
    register() 