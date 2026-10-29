
import bpy
import math

# =====================================
# 1. CLEAR THE DEFAULT BLENDER SCENE
# =====================================
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# =====================================
# 2. CREATE MATERIALS
# =====================================
def create_material(name, color, emission=0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1)
    material.use_nodes = True

    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = 0.2
    shader.inputs["Roughness"].default_value = 0.4

    if emission > 0:
        shader.inputs["Emission Color"].default_value = (*color, 1)
        shader.inputs["Emission Strength"].default_value = emission

    return material

cyan = create_material("Radar Cyan", (0.0, 0.65, 1.0), 3)
blue = create_material("Radar Blue", (0.02, 0.12, 0.8), 2)
dark = create_material("Dark Environment", (0.008, 0.02, 0.05))

# =====================================
# 3. CREATE THE FLOOR
# =====================================
bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
floor = bpy.context.object
floor.name = "Radar Room Floor"
floor.data.materials.append(dark)

# =====================================
# 4. CREATE A SIMPLE SOFA
# =====================================
def cube_object(name, location, dimensions, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(
        location=False, rotation=False, scale=True
    )
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("Soft Edges", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 3
    return obj

cube_object("Sofa Seat", (0, 0, 0.65), (2.4, 0.9, 0.3), blue)
cube_object("Sofa Back", (0, 0.38, 1.15), (2.4, 0.2, 1.0), blue)
cube_object("Sofa Left Arm", (-1.1, 0, 0.9), (0.2, 0.9, 0.6), blue)
cube_object("Sofa Right Arm", (1.1, 0, 0.9), (0.2, 0.9, 0.6), blue)

# =====================================
# 5. CREATE A SIMPLE SEATED FIGURE
# =====================================
def sphere_object(name, location, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=20, ring_count=12, radius=1, location=location
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj

# Head and torso
sphere_object("Human Head", (0, -0.05, 2.0), (0.28, 0.28, 0.32), cyan)
sphere_object("Human Torso", (0, -0.08, 1.45), (0.35, 0.25, 0.55), cyan)

# Arms
sphere_object("Left Arm", (-0.48, -0.12, 1.55), (0.12, 0.42, 0.12), cyan)
sphere_object("Right Arm", (0.48, -0.12, 1.55), (0.12, 0.42, 0.12), cyan)

# Legs
sphere_object("Left Leg", (-0.2, -0.48, 0.95), (0.14, 0.48, 0.14), cyan)
sphere_object("Right Leg", (0.2, -0.48, 0.95), (0.14, 0.48, 0.14), cyan)

# =====================================
# 6. ADD GLOWING RADAR PARTICLES
# =====================================
import random
random.seed(7)

for i in range(700):
    x = random.uniform(-0.65, 0.65)
    y = random.uniform(-0.65, 0.2)
    z = random.uniform(0.9, 2.3)

    # Keep particles roughly around the human
    if abs(x) < 0.35 and z < 1.85:
        pass

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=6, ring_count=4,
        radius=random.uniform(0.008, 0.018),
        location=(x, y, z)
    )
    point = bpy.context.object
    point.name = f"Radar Particle {i:03d}"
    point.data.materials.append(cyan)

# =====================================
# 7. LIGHTS AND CAMERA
# =====================================
bpy.ops.object.light_add(type="AREA", location=(1, -4, 5))
light = bpy.context.object
light.data.energy = 1000
light.data.shape = "DISK"
light.data.size = 5

bpy.ops.object.camera_add(location=(4, -7, 3.4))
camera = bpy.context.object
direction = -camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 50
bpy.context.scene.camera = camera

# =====================================
# 8. RENDER SETTINGS
# =====================================
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

scene.world.color = (0.005, 0.01, 0.03)
scene.render.filepath = "//radar_scene.png"

# Save Blender project
bpy.ops.wm.save_as_mainfile(
    filepath=bpy.path.abspath("//radar_scene.blend")
)

print("SUCCESS: Radar scene created!")
