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

# Constantes para definir propriedades do rig
RIG_BASE_HEIGHT = 2.0  # Altura base do rig em metros (do root até squash_top quando zerado)

def log_debug(message):
    """Log debug message to console"""
    # Verificar se os logs estão ativados nas preferências do addon
    try:
        prefs = bpy.context.preferences.addons["super_empty"].preferences
        if prefs.enable_debug_logs:
            print(f"SUPER_EMPTY_DEBUG: {message}")
    except (AttributeError, KeyError):
        # Se as preferências não estiverem disponíveis, não logamos nada
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
    
    # Verificar se a configuração de correção de pose space está ativada
    prefs = bpy.context.preferences.addons["super_empty"].preferences
    if not prefs.use_pose_space_correction:
        log_debug("Pose space correction disabled by user preference")
        return orientation
    
    # ABORDAGEM COMPLETAMENTE NOVA REVISADA
    # Criar uma matriz de rotação que mapeia o espaço do objeto para o espaço do bone
    
    # Converter quaternion para matriz
    obj_matrix = orientation.to_matrix()
    obj_matrix.normalize()  # Garantir que não tenha escala
    
    # Criar uma matriz que representa a rotação de 90 graus em X
    # que converte Z-up (object space) para Y-up (bone space)
    conversion = Matrix.Rotation(math.radians(-90), 3, 'X')
    
    # Aplicar a conversão à matriz de orientação do objeto
    bone_matrix = conversion @ obj_matrix
    
    # Converter de volta para quaternion
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
    Calcula as dimensões do objeto considerando sua rotação.
    Retorna as dimensões em world space, com orientação.
    """
    # Obter a matriz de transformação do objeto
    matrix_world = obj.matrix_world
    
    # Calcular dimensões usando o bounding box orientado
    corners = [matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Extrair min/max em cada eixo
    min_x = min(corner.x for corner in corners)
    max_x = max(corner.x for corner in corners)
    min_y = min(corner.y for corner in corners)
    max_y = max(corner.y for corner in corners)
    min_z = min(corner.z for corner in corners)
    max_z = max(corner.z for corner in corners)
    
    # Calcular as dimensões
    dimensions = Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    
    # Calcular o centro
    center = Vector(((max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2))
    
    # Calcular a posição inferior
    bottom = Vector(((max_x + min_x)/2, (max_y + min_y)/2, min_z))
    
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
    
    # MODIFICADO: Calcular dimensões considerando a rotação do objeto
    dimensions = Vector((0, 0, 0))
    
    # Obter dimensões de cada objeto selecionado considerando a rotação
    all_dimensions = []
    min_point = Vector((float('inf'), float('inf'), float('inf')))
    max_point = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    # Calculate actual dimensions and collect face normals for orientation
    face_normals = []
    
    for obj in context.selected_objects:
        if obj.type == 'MESH':
            log_debug(f"Processing mesh object: {obj.name}")
            
            # Obter dimensões considerando a rotação
            obj_dim, obj_center, obj_bottom = get_object_oriented_dimensions(obj)
            all_dimensions.append(obj_dim)
            
            # Atualizar min/max points
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
    
    # Use as dimensões do bbox como fallback
    bbox_dimensions = max_point - min_point
    
    # Para múltiplos objetos, usamos a maior dimensão em cada eixo
    if len(all_dimensions) > 1:
        dimensions.x = max(dim.x for dim in all_dimensions)
        dimensions.y = max(dim.y for dim in all_dimensions)
        dimensions.z = max(dim.z for dim in all_dimensions)
    elif len(all_dimensions) == 1:
        dimensions = all_dimensions[0]
    else:
        dimensions = bbox_dimensions
    
    # Usar o centro e bottom calculados do bbox
    bottom = Vector((center.x, center.y, min_point.z))
    
    log_debug(f"Calculated bounds - Min: {min_point}, Max: {max_point}")
    log_debug(f"Final Dimensions: {dimensions}")
    log_debug(f"Bottom: {bottom}")
    
    # Determine orientation based on average normal
    orientation = Quaternion((1, 0, 0, 0))  # Default identity orientation
    
    # MODIFICADO: usar a orientação do objeto diretamente em vez de calcular normals
    if original_active and original_active.type == 'MESH':
        log_debug("Using active object's orientation directly")
        
        # Pegar a matriz de rotação do objeto
        rotation_matrix = original_active.matrix_world.to_3x3()
        
        # Remover escala da matriz
        rotation_matrix.normalize()
        
        # Converter para quaternion
        orientation = rotation_matrix.to_quaternion()
        
        log_debug(f"Object rotation quaternion: {orientation}")
        
        # Pegar e logar também os valores euler para debug
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
    
    # Restore original mode
    if original_mode and original_mode != 'OBJECT' and original_active:
        context.view_layer.objects.active = original_active
        bpy.ops.object.mode_set(mode=original_mode)
    
    log_debug("Finished get_selection_bounds")
    return dimensions, center, bottom, orientation

class SUPEREMPTY_OT_load_rig(Operator):
    """Load the Super Empty rig and parent selected objects to it"""
    bl_idname = "superempty.load_rig"
    bl_label = "Load Super Empty Rig"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Propriedades do operador
    adjust_shapes: BoolProperty(
        name="Adjust Shape Sizes",
        description="Automatically adjust the size of bone shapes to match the object",
        default=True
    )
    
    use_direct_copy: BoolProperty(
        name="Usar Cópia Direta de Orientação",
        description="Usar uma abordagem direta para copiar a orientação do objeto para o rig",
        default=True
    )
    
    def invoke(self, context, event):
        # Usar as propriedades da cena como valores padrão
        self.adjust_shapes = context.scene.super_empty_adjust_shapes
        self.use_direct_copy = context.scene.super_empty_use_direct_copy
        return self.execute(context)
    
    def execute(self, context):
        # Salvar as preferências do usuário na cena para uso futuro
        context.scene.super_empty_adjust_shapes = self.adjust_shapes
        context.scene.super_empty_use_direct_copy = self.use_direct_copy
        
        log_debug("\n\n==== STARTING SUPER EMPTY RIG OPERATOR ====")
        
        # Check if there are selected objects
        if len(context.selected_objects) == 0:
            self.report({'ERROR'}, "No objects selected")
            return {'CANCELLED'}
        
        # Filter mesh objects
        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not mesh_objects:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}
        
        # Store original selection
        original_selected = context.selected_objects.copy()
        original_active = context.view_layer.objects.active
        
        log_debug(f"Selected objects: {[obj.name for obj in original_selected]}")
        
        # Get dimensions, center, bottom and orientation using Blender's built-in tools
        dimensions, center, bottom, orientation = get_selection_bounds(context)
        
        # Log the values before modifying anything
        log_debug(f"BEFORE MODIFICATION - Dimensions: {dimensions}, Center: {center}, Bottom: {bottom}")
        log_debug(f"BEFORE MODIFICATION - Orientation: {orientation}")
        
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

        # Link the collection to the scene
        for col in data_to.collections:
            if col is not None:
                if col.name in bpy.context.scene.collection.children:
                    rig_col = bpy.context.scene.collection.children[col.name]
                    self.report({'WARNING'}, f"Collection {col.name} already exists, using existing one")
                else:
                    bpy.context.scene.collection.children.link(col)
                    rig_col = col

        # Find the armature
        armature = None
        for obj in rig_col.objects:
            if obj.type == 'ARMATURE' and obj.name == 'super_empty_template':
                armature = obj
                break
        
        if not armature:
            self.report({'ERROR'}, "Armature not found in the imported collection")
            return {'CANCELLED'}
        
        log_debug(f"Found armature: {armature.name}")
        
        # Create the selected_objects collection if it doesn't exist
        if "selected_objects" not in rig_col.children:
            selected_objects_col = bpy.data.collections.new("selected_objects")
            rig_col.children.link(selected_objects_col)
        else:
            selected_objects_col = rig_col.children["selected_objects"]
        
        # Guarda o modo original
        original_armature_mode = armature.mode
        
        # Step 1: Position the armature at the bottom position
        log_debug(f"STEP 1 - Positioning armature at bottom: {bottom}")
        armature.location = bottom
        
        # Apply the orientation to the armature object if using direct copy
        if self.use_direct_copy:
            log_debug("Using direct copy of orientation to armature object")
            armature.rotation_mode = 'QUATERNION'
            armature.rotation_quaternion = orientation
        
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
        if not self.use_direct_copy:
            log_debug(f"STEP 4 - Setting sub_root orientation")
            # Adjust the orientation to account for bone space vs object space difference
            adjusted_orientation = adjust_orientation_for_bone_space(orientation)
            
            sub_root_bone.rotation_mode = 'QUATERNION'
            sub_root_bone.rotation_quaternion = adjusted_orientation
            
            # Log the actual value that was set
            log_debug(f"sub_root_bone rotation after setting adjusted orientation: {sub_root_bone.rotation_quaternion}")
        else:
            log_debug("Skipping sub_root orientation - using direct copy to armature instead")
        
        # Step 5: Position the squash_top para ajustar a altura do objeto - EIXO Y
        height = dimensions.z
        if height < 0.001:
            height = 1.0
        
        # Ajustar a posição do squash_top para corresponder à altura do objeto
        # Independentemente se o objeto é maior ou menor que a altura base do rig
        height_adjustment = height - RIG_BASE_HEIGHT
        
        # Ajustamos a localização do squash_top para refletir a diferença entre a altura do objeto
        # e a altura base do rig - permitindo que ele fique negativo para objetos menores
        log_debug(f"STEP 5 - Object height: {height}, Rig base height: {RIG_BASE_HEIGHT}")
        log_debug(f"Setting squash_top height on Y axis: {height_adjustment}")
        squash_top_bone.location = Vector((0, height_adjustment, 0))
        
        # Step 6: Scale the bone shapes to match the object dimensions - AGORA NÃO UNIFORME
        if self.adjust_shapes:
            log_debug(f"STEP 6 - Scaling bone shapes to match object dimensions")
            
            # Calcular escala baseada nas dimensões do objeto
            # Considerando que o shape padrão é 2x2x2
            # Mapeamento dos eixos:
            # - Em object space: X,Y,Z
            # - Em bone space: X,Z,Y (Para o main bone)
            x_scale = max(dimensions.x / 2.0, 0.1)  # Eixo X
            z_scale = max(dimensions.y / 2.0, 0.1)  # Eixo Y no espaço obj = Z no bone space
            y_scale = max(dimensions.z / 2.0, 0.1)  # Eixo Z no espaço obj = Y no bone space
            
            log_debug(f"Non-uniform shape scale: X={x_scale}, Y={y_scale}, Z={z_scale}")
            
            # Aplicar escala não-uniforme no bone principal para cobrir toda a seleção
            main_bone.custom_shape_scale_xyz = Vector((x_scale, y_scale, z_scale))
            
            # Para os outros bones, usar uma escala uniforme baseada na maior dimensão
            max_dim = max(dimensions.x, dimensions.y, dimensions.z)
            uniform_scale = max(max_dim / 2.0, 0.1)
            
            # Aplicar escala uniforme para os outros bones
            for bone in armature.pose.bones:
                if bone.custom_shape and bone != main_bone:
                    bone.custom_shape_scale_xyz = Vector((uniform_scale, uniform_scale, uniform_scale))
        
        # Exit pose mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Step 7: Move selected objects to the selected_objects collection
        log_debug("STEP 7 - Moving objects to collection")
        for obj in original_selected:
            # Remove from current collections
            for col in obj.users_collection:
                col.objects.unlink(obj)
            # Add to our collection
            selected_objects_col.objects.link(obj)
        
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
        
        # Usar sempre a altura real do objeto para o rest_length
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
        
        # Now select the objects to be parented
        for obj in original_selected:
            obj.select_set(True)
        
        # Parent the objects to the main bone as the very last step
        bpy.ops.object.parent_set(type='BONE', keep_transform=True)
        
        # Log the objects' transformations after parenting
        log_debug("After parenting:")
        for obj in original_selected:
            log_debug(f"Object {obj.name} - Location: {obj.location}, Rotation: {obj.rotation_euler if obj.rotation_mode == 'XYZ' else obj.rotation_quaternion}")
        
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
        
        # Seção principal com informações sobre o addon
        box = layout.box()
        box.label(text="Super Empty Rig")
        box.label(text="Carrega um rig e parenteia objetos")
        
        # Adicionar as opções do operador diretamente no painel
        op_box = layout.box()
        op_box.label(text="Opções de Carregamento:")
        row = op_box.row()
        op = row.operator(SUPEREMPTY_OT_load_rig.bl_idname, text="Carregar Rig")
        
        col = op_box.column()
        col.prop(context.scene, "super_empty_adjust_shapes")
        col.prop(context.scene, "super_empty_use_direct_copy")
        
        # Adicionar atalhos para as preferências do addon
        pref_box = layout.box()
        pref_box.label(text="Preferências:")
        
        try:
            addon_prefs = context.preferences.addons["super_empty"].preferences
            row = pref_box.row()
            row.prop(addon_prefs, "use_pose_space_correction", text="Correção de Pose Space")
            
            # Botão para acessar todas as preferências
            pref_box.operator("preferences.addon_show", text="Mais Configurações").module="super_empty"
        except (AttributeError, KeyError):
            pref_box.label(text="Addon não encontrado")

# Armazenar preferências também na cena para uso no painel
def register_scene_properties():
    bpy.types.Scene.super_empty_adjust_shapes = BoolProperty(
        name="Ajustar Tamanho dos Shapes",
        description="Ajusta automaticamente o tamanho dos bone shapes para corresponder ao objeto",
        default=True
    )
    
    bpy.types.Scene.super_empty_use_direct_copy = BoolProperty(
        name="Usar Cópia Direta",
        description="Aplica a orientação diretamente no armature em vez do bone sub_root",
        default=True
    )

def unregister_scene_properties():
    del bpy.types.Scene.super_empty_adjust_shapes
    del bpy.types.Scene.super_empty_use_direct_copy

class SuperEmptyPreferences(AddonPreferences):
    """Addon preferences for Super Empty Rig"""
    bl_idname = "super_empty"
    
    use_pose_space_correction: BoolProperty(
        name="Usar Correção de Pose Space",
        description="Ativar correção de orientação para o espaço de bones do Blender (Y=up vs Z=up)",
        default=False  # MODIFICADO: agora desligado por padrão
    )
    
    enable_debug_logs: BoolProperty(
        name="Ativar Logs de Debug",
        description="Mostra mensagens de debug detalhadas no console do Blender",
        default=False
    )
    
    def draw(self, context):
        layout = self.layout
        
        # Seção de configurações de orientação
        box = layout.box()
        box.label(text="Configurações de Orientação:")
        box.prop(self, "use_pose_space_correction")
        box.label(text="Desligado: Preserva a rotação original do objeto")
        box.label(text="Ligado: Tenta converter entre espaços de coordenadas (Y-up vs Z-up)")
        
        # Seção de debug
        box = layout.box()
        box.label(text="Configurações de Debug:")
        box.prop(self, "enable_debug_logs")
        box.label(text="Ligado: Mostra mensagens detalhadas no console para troubleshooting")
        
        # Nota informativa sobre como ver os logs
        if self.enable_debug_logs:
            info_box = layout.box()
            info_box.label(text="Para ver os logs: Janela > Toggle System Console")


classes = (
    SUPEREMPTY_OT_load_rig,
    SUPEREMPTY_PT_panel,
    SuperEmptyPreferences,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Armazenar preferências também na cena para uso no painel
    register_scene_properties()

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Remover preferências da cena
    unregister_scene_properties()

if __name__ == "__main__":
    register() 