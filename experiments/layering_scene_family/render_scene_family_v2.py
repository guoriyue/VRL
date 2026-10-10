"""Render one randomized known-layer room scene (v2: perspective, 16 classes, textures) with per-part occlusion labels.

v2 over v1: perspective cameras, 16 object classes (adds bed, wardrobe, desk,
armchair, tv_stand, side_table, rug, pendant lamp, wall painting), procedural
wood/fabric textures, real-photo crops as painting canvases, windows and doors
in the room layer, a sun light, and 16-bit part-id passes.

Run inside Blender (not the training interpreter):

    blender --background --python render_scene_family.py -- --out DIR --seed N --split train|heldout

Every foreground object is a group of named parts; the room is layer 0. Outputs:
layer_i.png (RGBA, back to front), source.png (alpha composite), parts_visible.png
and parts_full_<i>.png (flat part-id passes), scene.json (every random choice),
render_manifest.json, occlusion.json and inferable_mask.png / unknowable_mask.png.

Occlusion labels come from geometry at generation time, never from a person:
a hidden part is *inferable* when it is still partly visible (continuation) or a
sibling of the same part class on the same object is mostly visible (symmetry);
otherwise it is *unknowable*. Reserved parameter ranges exist only in the
held-out split so generalization is measured on shapes never trained on.
"""

import argparse
import json
import math
import random
import struct
import sys
import zlib
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--split", choices=("train", "heldout"), required=True)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
parser.add_argument(
    "--avoid-combos", default=None, help="json list of object-class combos to avoid"
)
parser.add_argument(
    "--canvas-dir",
    default="/home/mingfeiguo/Desktop/VRL/data/external/object_move/coco_onto_val2017",
    help="real photos used as painting canvases",
)
opts = parser.parse_args(args)
output = Path(opts.out).resolve()
output.mkdir(parents=True, exist_ok=True)
rng = random.Random(opts.seed)
HELDOUT = opts.split == "heldout"

# ----------------------------------------------------------------------------
# Blender scene setup
# ----------------------------------------------------------------------------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
if opts.device == "GPU":
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type != "CPU"
scene.cycles.device = opts.device
scene.render.resolution_x = scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.world.color = (0.5, 0.5, 0.5)


def material(name, color, roughness=0.8):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    return mat


def textured(name, color, kind="wood", roughness=0.7):
    """Procedural grain (wood) or weave (fabric) modulating the base color."""
    mat = material(name, color, roughness)
    tree = mat.node_tree
    shader = tree.nodes.get("Principled BSDF")
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    dark = tuple(max(0.0, c * rng.uniform(0.78, 0.92)) for c in color)
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].color = (*color, 1)
    if kind == "wood":
        tex = tree.nodes.new("ShaderNodeTexWave")
        tex.inputs["Scale"].default_value = rng.uniform(2, 6)
        tex.inputs["Distortion"].default_value = rng.uniform(1, 4)
        tex.inputs["Detail"].default_value = 4
        tree.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    else:
        tex = tree.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = rng.uniform(40, 120)
        tex.inputs["Detail"].default_value = 6
        tree.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    return mat


def image_material(name, path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    shader = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tree.links.new(tex.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 0.6
    return mat


def hsv(h, s, v):
    import colorsys

    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ----------------------------------------------------------------------------
# Part primitives
# ----------------------------------------------------------------------------
PARTS = []  # (object_index, part_class, blender_object)


def register(obj, obj_index, part_class, mat):
    obj.data.materials.append(mat)
    bevel = obj.modifiers.new("Soft manufactured edge", "BEVEL")
    bevel.width = 0.015
    bevel.segments = 2
    PARTS.append((obj_index, part_class, obj))
    return obj


def box(obj_index, part_class, location, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = f"{obj_index}:{part_class}"
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return register(obj, obj_index, part_class, mat)


def cylinder(obj_index, part_class, location, radius, depth, mat, radius_top=None):
    if radius_top is None:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=radius, depth=depth, location=location
        )
    else:
        bpy.ops.mesh.primitive_cone_add(
            vertices=32, radius1=radius, radius2=radius_top, depth=depth, location=location
        )
    obj = bpy.context.object
    obj.name = f"{obj_index}:{part_class}"
    return register(obj, obj_index, part_class, mat)


def sphere(obj_index, part_class, location, radius, mat):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = f"{obj_index}:{part_class}"
    return register(obj, obj_index, part_class, mat)


def leg(obj_index, part_class, location, height, mat, style, thickness):
    x, y, z = location
    if style == "box":
        return box(obj_index, part_class, (x, y, z), (thickness, thickness, height), mat)
    if style == "cylinder":
        return cylinder(obj_index, part_class, (x, y, z), thickness / 2, height, mat)
    # tapered: reserved for the held-out split
    return cylinder(
        obj_index, part_class, (x, y, z), thickness / 2, height, mat, radius_top=thickness / 4
    )


# ----------------------------------------------------------------------------
# Object library (each returns a footprint (w, d) and the parts it placed)
# ----------------------------------------------------------------------------
LEG_STYLES_TRAIN = ["box", "cylinder"]
LEG_STYLES_HELDOUT = ["tapered"]


def pick_leg_style(record):
    style = rng.choice(LEG_STYLES_HELDOUT if HELDOUT and rng.random() < 0.7 else LEG_STYLES_TRAIN)
    record["leg_style"] = style
    return style


def build_cabinet(i, x, y, record):
    w = rng.uniform(0.8, 1.3)
    d = rng.uniform(0.6, 0.9)
    h = rng.uniform(0.6, 1.0)
    wood = textured(
        f"wood{i}", hsv(rng.uniform(0.05, 0.1), rng.uniform(0.4, 0.8), rng.uniform(0.3, 0.8))
    )
    metal = material(f"metal{i}", (0.2, 0.21, 0.23), 0.4)
    foot_h = rng.uniform(0.14, 0.26)
    box(i, "body", (x, y, foot_h + h / 2), (w, d, h), wood)
    box(i, "top", (x, y, foot_h + h + 0.04), (w + 0.1, d + 0.08, 0.08), wood)
    # Reserved for held-out: wide double doors (ratio 0.6-0.8); train has one drawer handle.
    if HELDOUT and rng.random() < 0.7:
        ratio = rng.uniform(0.6, 0.8)
        record["door_ratio"] = ratio
        for sx in (-1, 1):
            box(
                i,
                "door",
                (x + sx * w * ratio / 4, y - d / 2 - 0.012, foot_h + h / 2),
                (w * ratio / 2 - 0.02, 0.02, h * 0.8),
                wood,
            )
            box(
                i,
                "handle",
                (x + sx * w * 0.08, y - d / 2 - 0.035, foot_h + h / 2),
                (0.04, 0.03, 0.14),
                metal,
            )
    else:
        record["door_ratio"] = None
        box(i, "handle", (x, y - d / 2 - 0.03, foot_h + h * 0.6), (0.25, 0.06, 0.04), metal)
    style = pick_leg_style(record)
    for dx in (-w * 0.38, w * 0.38):
        for dy in (-d * 0.36, d * 0.36):
            leg(i, "foot", (x + dx, y + dy, foot_h / 2), foot_h, wood, style, 0.1)
    return w + 0.1, d + 0.08


def build_table(i, x, y, record):
    w = rng.uniform(1.0, 1.8)
    d = rng.uniform(0.6, 0.95)
    h = rng.uniform(0.7, 0.95)
    wood = textured(
        f"wood{i}", hsv(rng.uniform(0.04, 0.12), rng.uniform(0.3, 0.7), rng.uniform(0.4, 0.85))
    )
    box(i, "top", (x, y, h), (w, d, 0.07), wood)
    style = pick_leg_style(record)
    for dx in (-w * 0.45, w * 0.45):
        for dy in (-d * 0.4, d * 0.4):
            leg(i, "leg", (x + dx, y + dy, (h - 0.035) / 2), h - 0.035, wood, style, 0.09)
    if rng.random() < 0.5:
        box(i, "rail", (x, y + d * 0.4, h - 0.15), (w * 0.9, 0.05, 0.16), wood)
    return w, d


def build_chair(i, x, y, record):
    s = rng.uniform(0.8, 1.0)
    mat = material(
        f"chair{i}",
        hsv(
            rng.uniform(0.5, 0.75) if not HELDOUT else rng.uniform(0.0, 0.1),
            rng.uniform(0.5, 0.9),
            rng.uniform(0.35, 0.7),
        ),
    )
    seat_h = 0.55 * s
    box(i, "seat", (x, y, seat_h), (0.9 * s, 0.86 * s, 0.12), mat)
    style = pick_leg_style(record)
    for dx in (-0.34 * s, 0.34 * s):
        for dy in (-0.31 * s, 0.31 * s):
            leg(i, "leg", (x + dx, y + dy, seat_h / 2 - 0.03), seat_h - 0.06, mat, style, 0.1 * s)
    for dx in (-0.35 * s, 0.35 * s):
        box(
            i,
            "back_post",
            (x + dx, y + 0.33 * s, seat_h + 0.45 * s),
            (0.1 * s, 0.1 * s, 0.95 * s),
            mat,
        )
    box(i, "back", (x, y + 0.33 * s, seat_h + 0.72 * s), (0.8 * s, 0.1 * s, 0.4 * s), mat)
    return 0.92 * s, 0.9 * s


def build_lamp(i, x, y, record):
    metal = material(f"metal{i}", (0.18, 0.2, 0.22), 0.4)
    shade = material(
        f"shade{i}", hsv(rng.uniform(0.08, 0.16), rng.uniform(0.3, 0.6), rng.uniform(0.75, 0.95))
    )
    h = rng.uniform(1.2, 1.7)
    cylinder(i, "base", (x, y, 0.03), rng.uniform(0.15, 0.22), 0.06, metal)
    cylinder(i, "stem", (x, y, h / 2), 0.03, h, metal)
    style = "cylinder" if HELDOUT and rng.random() < 0.7 else "cone"
    record["shade_style"] = style
    if style == "cone":
        cylinder(
            i, "shade", (x, y, h + 0.17), rng.uniform(0.26, 0.34), 0.36, shade, radius_top=0.16
        )
    else:
        cylinder(i, "shade", (x, y, h + 0.17), rng.uniform(0.2, 0.28), 0.36, shade)
    return 0.5, 0.5


def build_plant(i, x, y, record):
    pot = material(
        f"pot{i}", hsv(rng.uniform(0.02, 0.07), rng.uniform(0.5, 0.8), rng.uniform(0.4, 0.7))
    )
    leaf = material(
        f"leaf{i}", hsv(rng.uniform(0.25, 0.4), rng.uniform(0.5, 0.8), rng.uniform(0.3, 0.6))
    )
    r = rng.uniform(0.14, 0.2)
    cylinder(i, "pot", (x, y, 0.14), r, 0.28, pot, radius_top=r * 1.15)
    sphere(i, "foliage", (x, y, 0.28 + rng.uniform(0.28, 0.4)), rng.uniform(0.28, 0.4), leaf)
    return 0.8, 0.8


def build_crate(i, x, y, record):
    w = rng.uniform(0.8, 1.5)
    d = rng.uniform(0.6, 0.9)
    h = rng.uniform(0.45, 0.75)
    mat = material(
        f"crate{i}", hsv(rng.uniform(0.2, 0.45), rng.uniform(0.3, 0.7), rng.uniform(0.25, 0.55))
    )
    slat = material(f"slat{i}", hsv(0.08, 0.5, 0.6))
    box(i, "body", (x, y, h / 2), (w, d, h), mat)
    box(i, "lid", (x, y, h + 0.025), (w + 0.05, d + 0.05, 0.05), mat)
    for dx in (-w * 0.3, w * 0.3):
        box(i, "slat", (x + dx, y - d / 2 - 0.015, h / 2), (0.08, 0.03, h * 0.85), slat)
    return w + 0.05, d + 0.05


def build_bookshelf(i, x, y, record):
    w = rng.uniform(0.8, 1.2)
    d = rng.uniform(0.3, 0.42)
    h = rng.uniform(1.3, 1.9)
    wood = textured(
        f"wood{i}", hsv(rng.uniform(0.04, 0.1), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.75))
    )
    # Reserved for held-out: 4-5 shelves; train has 2-3.
    shelves = rng.randint(4, 5) if HELDOUT and rng.random() < 0.7 else rng.randint(2, 3)
    record["shelves"] = shelves
    box(i, "back_panel", (x, y + d / 2 - 0.015, h / 2), (w, 0.03, h), wood)
    for sx in (-1, 1):
        box(i, "side", (x + sx * (w / 2 - 0.02), y, h / 2), (0.04, d, h), wood)
    box(i, "top", (x, y, h - 0.02), (w, d, 0.04), wood)
    for k in range(shelves):
        z = (k + 1) * h / (shelves + 1)
        box(i, "shelf", (x, y, z), (w - 0.08, d - 0.03, 0.035), wood)
    return w, d


def build_sofa(i, x, y, record):
    w = rng.uniform(1.4, 2.0)
    d = rng.uniform(0.8, 0.95)
    mat = material(
        f"sofa{i}", hsv(rng.uniform(0.55, 0.95), rng.uniform(0.3, 0.6), rng.uniform(0.35, 0.6))
    )
    foot_h = 0.12
    box(i, "seat", (x, y, foot_h + 0.2), (w, d, 0.4), mat)
    box(i, "back", (x, y + d / 2 - 0.12, foot_h + 0.65), (w, 0.24, 0.5), mat)
    for sx in (-1, 1):
        box(i, "arm", (x + sx * (w / 2 - 0.1), y, foot_h + 0.5), (0.2, d, 0.2), mat)
    style = pick_leg_style(record)
    for dx in (-w * 0.42, w * 0.42):
        for dy in (-d * 0.38, d * 0.38):
            leg(
                i,
                "foot",
                (x + dx, y + dy, foot_h / 2),
                foot_h,
                material(f"foot{i}{dx}{dy}", (0.15, 0.1, 0.07)),
                style,
                0.08,
            )
    return w, d


def build_stool(i, x, y, record):
    mat = material(
        f"stool{i}", hsv(rng.uniform(0.0, 1.0), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
    )
    h = rng.uniform(0.45, 0.7)
    cylinder(i, "seat", (x, y, h), rng.uniform(0.18, 0.24), 0.06, mat)
    style = pick_leg_style(record)
    for k in range(3):
        a = 2 * math.pi * k / 3
        leg(
            i,
            "leg",
            (x + 0.15 * math.cos(a), y + 0.15 * math.sin(a), h / 2 - 0.03),
            h - 0.06,
            mat,
            style,
            0.07,
        )
    return 0.55, 0.55


def wood(i, tag=""):
    return textured(
        f"wood{i}{tag}",
        hsv(rng.uniform(0.04, 0.12), rng.uniform(0.3, 0.75), rng.uniform(0.3, 0.85)),
        "wood",
    )


def fabric(i, tag="", hue=None):
    h = rng.uniform(0.0, 1.0) if hue is None else hue
    return textured(
        f"fabric{i}{tag}", hsv(h, rng.uniform(0.15, 0.6), rng.uniform(0.35, 0.85)), "fabric", 0.9
    )


def build_bed(i, x, y, record):
    w = rng.uniform(1.4, 1.9)
    d = rng.uniform(1.9, 2.2)
    frame = wood(i)
    sheet = fabric(i, "s", hue=rng.choice([0.1, 0.6, 0.0]))
    leg_h = rng.uniform(0.12, 0.22)
    box(i, "frame", (x, y, leg_h + 0.12), (w, d, 0.24), frame)
    box(i, "mattress", (x, y, leg_h + 0.36), (w - 0.06, d - 0.08, 0.24), sheet)
    box(
        i, "headboard", (x, y + d / 2 - 0.04, leg_h + 0.6), (w, 0.08, rng.uniform(0.7, 1.1)), frame
    )
    for sx in (-1, 1):
        box(
            i,
            "pillow",
            (x + sx * w * 0.24, y + d / 2 - 0.3, leg_h + 0.54),
            (w * 0.4, 0.4, 0.14),
            fabric(i, f"p{sx}", 0.1),
        )
    box(i, "blanket", (x, y - d * 0.12, leg_h + 0.5), (w - 0.02, d * 0.6, 0.06), fabric(i, "b"))
    style = pick_leg_style(record)
    for dx in (-w * 0.45, w * 0.45):
        for dy in (-d * 0.45, d * 0.45):
            leg(i, "leg", (x + dx, y + dy, leg_h / 2), leg_h, frame, style, 0.08)
    return w, d


def build_wardrobe(i, x, y, record):
    w = rng.uniform(1.0, 1.6)
    d = rng.uniform(0.5, 0.65)
    h = rng.uniform(1.8, 2.2)
    mat = wood(i)
    metal = material(f"metal{i}", (0.2, 0.21, 0.23), 0.4)
    box(i, "body", (x, y, 0.08 + h / 2), (w, d, h), mat)
    box(i, "plinth", (x, y, 0.04), (w - 0.04, d - 0.04, 0.08), mat)
    doors = 3 if HELDOUT and rng.random() < 0.7 else 2
    record["wardrobe_doors"] = doors
    for k in range(doors):
        cx = x - w / 2 + (k + 0.5) * w / doors
        box(
            i,
            "door",
            (cx, y - d / 2 - 0.01, 0.08 + h / 2),
            (w / doors - 0.02, 0.02, h - 0.06),
            mat,
        )
        box(
            i,
            "handle",
            (cx + (0.3 if k % 2 == 0 else -0.3) * w / doors, y - d / 2 - 0.035, 0.08 + h / 2),
            (0.03, 0.03, 0.25),
            metal,
        )
    return w, d


def build_desk(i, x, y, record):
    w = rng.uniform(1.1, 1.6)
    d = rng.uniform(0.55, 0.75)
    h = rng.uniform(0.72, 0.78)
    mat = wood(i)
    metal = material(f"metal{i}", (0.2, 0.21, 0.23), 0.4)
    box(i, "top", (x, y, h), (w, d, 0.04), mat)
    side = rng.choice([-1, 1])
    box(
        i,
        "drawer_block",
        (x + side * w * 0.32, y, (h - 0.02) / 2),
        (w * 0.32, d - 0.06, h - 0.02),
        mat,
    )
    for k in range(3):
        box(
            i,
            "drawer_handle",
            (x + side * w * 0.32, y - d / 2 + 0.01, h * (0.2 + 0.28 * k)),
            (0.12, 0.03, 0.025),
            metal,
        )
    style = pick_leg_style(record)
    for dy in (-d * 0.42, d * 0.42):
        leg(i, "leg", (x - side * w * 0.46, y + dy, (h - 0.02) / 2), h - 0.02, mat, style, 0.06)
    return w, d


def build_armchair(i, x, y, record):
    s = rng.uniform(0.85, 1.05)
    cover = fabric(i)
    foot = material(f"foot{i}", (0.12, 0.08, 0.05))
    box(i, "seat", (x, y, 0.3 * s), (0.85 * s, 0.8 * s, 0.28 * s), cover)
    box(i, "back", (x, y + 0.36 * s, 0.6 * s), (0.85 * s, 0.14 * s, 0.6 * s), cover)
    for sx in (-1, 1):
        box(i, "arm", (x + sx * 0.4 * s, y, 0.48 * s), (0.12 * s, 0.8 * s, 0.24 * s), cover)
    style = pick_leg_style(record)
    for dx in (-0.36 * s, 0.36 * s):
        for dy in (-0.32 * s, 0.32 * s):
            leg(i, "leg", (x + dx, y + dy, 0.08 * s), 0.16 * s, foot, style, 0.06)
    return 0.9 * s, 0.85 * s


def build_tv_stand(i, x, y, record):
    w = rng.uniform(1.4, 2.0)
    d = rng.uniform(0.4, 0.5)
    mat = wood(i)
    screen = material(f"screen{i}", (0.02, 0.02, 0.03), 0.2)
    box(i, "body", (x, y, 0.28), (w, d, 0.45), mat)
    for k in range(2):
        box(
            i,
            "door",
            (x + (k - 0.5) * w / 2, y - d / 2 - 0.01, 0.28),
            (w / 2 - 0.03, 0.02, 0.38),
            mat,
        )
    tw = w * rng.uniform(0.6, 0.8)
    box(i, "tv", (x, y, 0.5 + tw * 0.3), (tw, 0.05, tw * 0.58), screen)
    box(i, "tv_foot", (x, y, 0.53), (0.3, 0.2, 0.04), screen)
    return w, d


def build_side_table(i, x, y, record):
    mat = wood(i)
    r = rng.uniform(0.22, 0.32)
    h = rng.uniform(0.5, 0.65)
    cylinder(i, "top", (x, y, h), r, 0.04, mat)
    style = pick_leg_style(record)
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.4
        leg(
            i,
            "leg",
            (x + r * 0.6 * math.cos(a), y + r * 0.6 * math.sin(a), (h - 0.02) / 2),
            h - 0.02,
            mat,
            style,
            0.04,
        )
    return 2 * r, 2 * r


def build_rug(i, x, y, record):
    w = rng.uniform(1.6, 2.8)
    d = rng.uniform(1.2, 2.0)
    box(i, "rug", (x, y, 0.008), (w, d, 0.016), fabric(i))
    record["flat"] = True
    return 0.0, 0.0  # never blocks placement


def build_pendant(i, x, y, record):
    metal = material(f"metal{i}", (0.15, 0.16, 0.18), 0.3)
    shade = textured(
        f"shade{i}",
        hsv(rng.uniform(0.0, 1.0), rng.uniform(0.2, 0.6), rng.uniform(0.5, 0.9)),
        "fabric",
    )
    z = rng.uniform(2.0, 2.5)
    cylinder(i, "cord", (x, y, z + 0.9), 0.01, 1.8, metal)
    cylinder(i, "shade", (x, y, z), rng.uniform(0.18, 0.3), 0.25, shade, radius_top=0.06)
    return 0.0, 0.0


CANVASES = []


def build_painting(i, x, y, record):
    import glob

    if not CANVASES:
        CANVASES.extend(sorted(glob.glob(str(Path(opts.canvas_dir) / "*.jpg")))[:4000])
    w = rng.uniform(0.6, 1.2)
    h = w * rng.uniform(0.6, 1.0)
    frame = wood(i, "f")
    on_side = rng.random() < 0.3
    z = rng.uniform(1.4, 1.9)
    if on_side:
        px, py = -2.92, rng.uniform(-1.0, 2.0)
        box(i, "frame", (px, py, z), (0.04, w + 0.08, h + 0.08), frame)
        canvas = box(
            i,
            "canvas",
            (px + 0.025, py, z),
            (0.01, w, h),
            image_material(f"canvas{i}", rng.choice(CANVASES)) if CANVASES else frame,
        )
    else:
        px, py = rng.uniform(-2.0, 2.0), 2.92
        box(i, "frame", (px, py, z), (w + 0.08, 0.04, h + 0.08), frame)
        canvas = box(
            i,
            "canvas",
            (px, py - 0.025, z),
            (w, 0.01, h),
            image_material(f"canvas{i}", rng.choice(CANVASES)) if CANVASES else frame,
        )
    del canvas
    record["x"], record["y"] = round(px, 3), round(py, 3)
    return 0.0, 0.0


LIBRARY = {
    "cabinet": build_cabinet,
    "table": build_table,
    "chair": build_chair,
    "lamp": build_lamp,
    "plant": build_plant,
    "crate": build_crate,
    "bookshelf": build_bookshelf,
    "sofa": build_sofa,
    "stool": build_stool,
    "bed": build_bed,
    "wardrobe": build_wardrobe,
    "desk": build_desk,
    "armchair": build_armchair,
    "tv_stand": build_tv_stand,
    "side_table": build_side_table,
    "rug": build_rug,
    "pendant": build_pendant,
    "painting": build_painting,
}
ON_TOP_CANDIDATES = {"lamp", "plant"}  # can stand on a table or cabinet

# ----------------------------------------------------------------------------
# Room and layout
# ----------------------------------------------------------------------------
# Reserved for held-out: blue-ish wall hue band 0.55-0.65.
wall_h = (
    rng.uniform(0.55, 0.65)
    if HELDOUT and rng.random() < 0.7
    else rng.choice([rng.uniform(0.05, 0.2), rng.uniform(0.3, 0.5), rng.uniform(0.7, 0.95)])
)
wall = material("wall", hsv(wall_h, rng.uniform(0.08, 0.3), rng.uniform(0.7, 0.9)))
floor_kind = rng.choice(["checker", "brick", "plain"])
floor = material(
    "floor", hsv(rng.uniform(0.05, 0.12), rng.uniform(0.3, 0.6), rng.uniform(0.5, 0.8))
)
nodes = floor.node_tree.nodes
if floor_kind == "checker":
    tex = nodes.new("ShaderNodeTexChecker")
    tex.inputs["Color1"].default_value = (
        *hsv(rng.uniform(0.05, 0.1), 0.5, rng.uniform(0.35, 0.5)),
        1,
    )
    tex.inputs["Color2"].default_value = (
        *hsv(rng.uniform(0.05, 0.1), 0.4, rng.uniform(0.65, 0.8)),
        1,
    )
    tex.inputs["Scale"].default_value = rng.uniform(10, 22)
    floor.node_tree.links.new(
        tex.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"]
    )
elif floor_kind == "brick":
    tex = nodes.new("ShaderNodeTexBrick")
    tex.inputs["Color1"].default_value = (*hsv(0.07, 0.5, rng.uniform(0.3, 0.45)), 1)
    tex.inputs["Color2"].default_value = (*hsv(0.08, 0.45, rng.uniform(0.5, 0.65)), 1)
    tex.inputs["Mortar"].default_value = (0.25, 0.17, 0.1, 1)
    tex.inputs["Scale"].default_value = rng.uniform(6, 12)
    tex.inputs["Mortar Size"].default_value = 0.01
    floor.node_tree.links.new(
        tex.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"]
    )
skirting = material("skirting", hsv(0.07, 0.6, 0.4))
box(0, "floor", (0, 0, -0.1), (16, 16, 0.2), floor)
box(0, "back_wall", (0, 3.0, 2.5), (16, 0.15, 5), wall)
box(0, "side_wall", (-3.0, 0, 2.5), (0.15, 16, 5), wall)
box(0, "skirting", (0, 2.9, 0.12), (16, 0.08, 0.24), skirting)
box(0, "skirting", (-2.9, 0, 0.12), (0.08, 16, 0.24), skirting)
box(
    0,
    "ceiling",
    (0, 0, 3.05),
    (16, 16, 0.1),
    material("ceiling", hsv(wall_h, 0.05, rng.uniform(0.85, 0.95))),
)
if rng.random() < 0.8:
    glass = bpy.data.materials.new("window_glass")
    glass.use_nodes = True
    gtree = glass.node_tree
    for node in list(gtree.nodes):
        gtree.nodes.remove(node)
    emit = gtree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (
        *hsv(rng.uniform(0.5, 0.62), rng.uniform(0.05, 0.3), 1.0),
        1,
    )
    emit.inputs["Strength"].default_value = rng.uniform(1.5, 3.0)
    gout = gtree.nodes.new("ShaderNodeOutputMaterial")
    gtree.links.new(emit.outputs["Emission"], gout.inputs["Surface"])
    ww, wh, wx = rng.uniform(1.0, 2.2), rng.uniform(1.2, 1.8), rng.uniform(-1.8, 1.8)
    box(0, "window_glass", (wx, 2.92, 1.0 + wh / 2), (ww, 0.02, wh), glass)
    frame_mat = material("window_frame", (0.92, 0.92, 0.9), 0.5)
    for dz in (1.0, 1.0 + wh):
        box(0, "window_frame", (wx, 2.9, dz), (ww + 0.08, 0.05, 0.06), frame_mat)
    for dx in (-ww / 2, 0.0, ww / 2):
        box(0, "window_frame", (wx + dx, 2.9, 1.0 + wh / 2), (0.06, 0.05, wh + 0.06), frame_mat)
if rng.random() < 0.5:
    door_mat = textured("door", hsv(rng.uniform(0.05, 0.12), 0.4, rng.uniform(0.4, 0.9)), "wood")
    box(0, "door", (-2.92, rng.uniform(-1.5, 1.5), 1.05), (0.04, 0.9, 2.1), door_mat)

count = rng.randint(2, 7)
classes = list(LIBRARY)
avoid = set()
if opts.avoid_combos:
    avoid = {tuple(c) for c in json.loads(Path(opts.avoid_combos).read_text())}
TARGETS = [
    "cabinet",
    "table",
    "chair",
    "sofa",
    "bookshelf",
    "bed",
    "desk",
    "armchair",
    "wardrobe",
    "tv_stand",
]
OCCLUDERS = ["crate", "sofa", "cabinet", "table", "bookshelf", "chair", "armchair", "bed", "desk"]
for _ in range(50):
    # objects 1 and 2 form the designed occlusion pair (target behind, occluder in front)
    chosen = [rng.choice(TARGETS), rng.choice(OCCLUDERS)] + [
        rng.choice(classes) for _ in range(count - 2)
    ]
    if tuple(sorted(chosen)) not in avoid:
        break
footprints = []  # (x, y, w, d)


def free(x, y, w, d, margin=0.1):
    for px, py, pw, pd in footprints:
        if abs(x - px) < (w + pw) / 2 + margin and abs(y - py) < (d + pd) / 2 + margin:
            return False
    return True


objects = []
# Designed occlusion pair: object 1 (target) behind, object 2 (occluder) in front of it.
placements = []
tx, ty = rng.uniform(-1.4, 0.8), rng.uniform(0.3, 1.5)
placements.append((tx, ty))
placements.append((tx + rng.uniform(-0.35, 0.35), ty - rng.uniform(1.0, 1.4)))
for _ in range(2, count):
    for _ in range(200):
        x, y = rng.uniform(-1.9, 1.7), rng.uniform(-1.5, 1.9)
        if all(abs(x - px) > 0.9 or abs(y - py) > 0.9 for px, py in placements):
            placements.append((x, y))
            break
    else:
        placements.append((rng.uniform(-1.9, 1.7), rng.uniform(-1.5, 1.9)))

index = 1
records = []
for cls, (x, y) in zip(chosen, placements, strict=False):
    record = {"index": index, "class": cls, "x": round(x, 3), "y": round(y, 3)}
    w, d = LIBRARY[cls](index, x, y, record)
    footprints.append((x, y, w, d))
    records.append(record)
    index += 1
# Lift a lamp or plant onto a table/cabinet top when one exists (overlapping silhouettes).
for record in records:
    if record["class"] in ON_TOP_CANDIDATES and rng.random() < 0.6:
        hosts = [r for r in records if r["class"] in ("table", "cabinet")]
        if hosts:
            host = rng.choice(hosts)
            parts = [o for i, c, o in PARTS if i == host["index"] and c == "top"]
            if parts:
                top = parts[0]
                z = top.location.z + top.dimensions.z / 2
                dx = rng.uniform(-0.3, 0.3)
                for i, _c, o in PARTS:
                    if i == record["index"]:
                        o.location = (host["x"] + dx, host["y"], o.location.z + z)
                record["on_top_of"] = host["index"]
                record["x"], record["y"] = round(host["x"] + dx, 3), round(host["y"], 3)

cam_az = rng.uniform(-0.45, 0.45)
cam_dist = rng.uniform(3.6, 5.0)
cam_h = rng.uniform(1.3, 2.1)
base = math.atan2(-7.2, 4.8)
cam_loc = Vector((cam_dist * math.cos(base + cam_az), cam_dist * math.sin(base + cam_az), cam_h))
bpy.ops.object.camera_add(location=cam_loc)
camera = bpy.context.object
stand = [r for r in records if r["class"] not in ("painting", "rug", "pendant")] or records
centroid = Vector(
    (
        sum(r["x"] for r in stand) / len(stand),
        sum(r["y"] for r in stand) / len(stand),
        rng.uniform(0.5, 0.8),
    )
)
camera.rotation_euler = (centroid - camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.type = "PERSP"
camera.data.lens = rng.uniform(24, 36)
camera.data.sensor_width = 36
scene.camera = camera
sun_rot = (rng.uniform(0.6, 1.1), 0.0, rng.uniform(-2.5, -0.6))
bpy.ops.object.light_add(type="SUN", rotation=sun_rot)
sun = bpy.context.object
sun.data.energy = rng.uniform(1.0, 3.0)
bpy.ops.object.light_add(
    type="AREA", location=(rng.uniform(-1.0, 1.5), rng.uniform(-1.5, 1.0), 2.95)
)
light = bpy.context.object
light.data.energy = rng.uniform(140, 260)
light.data.size = rng.uniform(3.0, 5.0)
light.rotation_euler = (0.0, 0.0, 0.0)  # pointing down under the ceiling
scene.world.color = (0.12, 0.12, 0.12)


# Back-to-front layer order: room first, then objects by distance from the camera (far first).
def object_distance(i):
    centers = [o.location for ii, c, o in PARTS if ii == i]
    center = sum(centers, Vector((0, 0, 0))) / len(centers)
    return (center - camera.location).length


# Wall paintings and floor rugs sit directly on the room: they go right behind every
# standing object regardless of their centroid distance.
flat = [r["index"] for r in records if r["class"] in ("painting", "rug")]
standing = [r["index"] for r in records if r["index"] not in flat]
order = [0, *flat, *sorted(standing, key=object_distance, reverse=True)]

# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
real_materials = {o.name: list(o.data.materials) for _, _, o in PARTS}
part_ids = {}
id_materials = {}
# Blender suffixes duplicate names (foot, foot.001, ...); the registered class is the truth.
part_meta = {o.name: (i, c) for i, c, o in PARTS}
STEP = 1.0 / 160.0  # up to 159 parts per scene; id passes are written as 16-bit PNG
for k, (_i, _c, o) in enumerate(PARTS):
    part_ids[o.name] = k + 1
    value = srgb_to_linear((k + 1) * STEP)
    mat = bpy.data.materials.new(f"id{k}")
    mat.use_nodes = True
    tree = mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (value, value, value, 1)
    emission.inputs["Strength"].default_value = 1.0
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    id_materials[o.name] = mat


def set_visible(indices):
    for i, _, o in PARTS:
        o.hide_render = i not in indices


def render(path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def use_id_materials(on):
    for _, _, o in PARTS:
        o.data.materials.clear()
        o.data.materials.append(id_materials[o.name] if on else real_materials[o.name][0])
    scene.cycles.samples = 1 if on else opts.samples
    scene.cycles.use_denoising = not on
    scene.cycles.filter_width = 0.01 if on else 1.5
    scene.world.color = (0, 0, 0) if on else (0.12, 0.12, 0.12)
    light.hide_render = on
    sun.hide_render = on
    scene.render.image_settings.color_depth = "16" if on else "8"


use_id_materials(False)
for layer, i in enumerate(order):
    set_visible({i})
    render(output / f"layer_{layer}.png")
use_id_materials(True)
set_visible(set(order))
render(output / "parts_visible.png")
for layer, i in enumerate(order):
    set_visible({i})
    render(output / f"parts_full_{layer}.png")
use_id_materials(False)
set_visible(set(order))
bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))


# ----------------------------------------------------------------------------
# Labels: composite, hidden masks, inferable / unknowable
# ----------------------------------------------------------------------------
def read(path, raw=False):
    img = bpy.data.images.load(str(path))
    if raw:
        img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return px


def write_png(path, array):
    """Minimal RGBA/greyscale PNG writer (uint8 HxWx4 or HxW)."""
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        color_type, row_bytes = 0, array[:, :, None]
    else:
        color_type, row_bytes = 6, array
    raw = b"".join(b"\x00" + row_bytes[y].tobytes() for y in range(array.shape[0]))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", array.shape[1], array.shape[0], 8, color_type, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


layers = [read(output / f"layer_{k}.png") for k in range(len(order))]
# layer PNGs are display-referred sRGB; composite in that space (what the model sees).
composite = np.zeros_like(layers[0][..., :3])
for layer in layers:
    alpha = layer[..., 3:4]
    composite = layer[..., :3] * alpha + composite * (1 - alpha)
write_png(
    output / "source.png",
    np.concatenate([np.clip(composite, 0, 1), np.ones_like(composite[..., :1])], axis=-1)
    .__mul__(255)
    .round()
    .astype(np.uint8),
)


def ids_from(px):
    valid = px[..., 3] > 0.5
    ids = np.round(px[..., 0] / STEP).astype(np.int32)
    ids[~valid] = 0
    return ids


visible_ids = ids_from(read(output / "parts_visible.png", raw=True))
full_ids = [ids_from(read(output / f"parts_full_{k}.png", raw=True)) for k in range(len(order))]
name_by_id = {v: k for k, v in part_ids.items()}
objects_out = []
inferable_mask = np.zeros(visible_ids.shape, dtype=bool)
unknowable_mask = np.zeros(visible_ids.shape, dtype=bool)
layer_alpha = [layer[..., 3] > 0.5 for layer in layers]
for layer, i in enumerate(order):
    if i == 0:
        continue
    front = np.zeros(visible_ids.shape, dtype=bool)
    for above in layer_alpha[layer + 1 :]:
        front |= above
    hidden_obj = layer_alpha[layer] & front
    parts_out = []
    stats = {}
    for name, pid in part_ids.items():
        obj_index, part_class = part_meta[name]
        if obj_index != i:
            continue
        full = full_ids[layer] == pid
        visible = visible_ids == pid
        full_px, vis_px = int(full.sum()), int(visible.sum())
        if full_px == 0:
            continue
        vis_frac = vis_px / full_px
        stats[name] = (part_class, full, visible, vis_frac)
    for name, (part_class, full, visible, vis_frac) in stats.items():
        hidden = full & ~visible
        hidden_frac = 1 - vis_frac
        siblings = [
            v for other, (cls, _, _, v) in stats.items() if other != name and cls == part_class
        ]
        if hidden_frac <= 0.5:
            label = "visible" if hidden_frac < 0.02 else "inferable"
            reason = "mostly visible; hidden remainder is a continuation"
        elif vis_frac >= 0.25:
            label, reason = "inferable", "partly visible; continuation of a visible part"
        elif any(v >= 0.6 for v in siblings):
            label, reason = (
                "inferable",
                "a sibling of the same part class is mostly visible (symmetry)",
            )
        else:
            label, reason = "unknowable", "hidden with no visible sibling or continuation"
        if label == "inferable":
            inferable_mask |= hidden
        elif label == "unknowable":
            unknowable_mask |= hidden
        parts_out.append(
            {
                "part": name,
                "class": part_class,
                "full_pixels": int(full.sum()),
                "visible_fraction": round(vis_frac, 4),
                "hidden_pixels": int(hidden.sum()),
                "label": label,
                "reason": reason,
            }
        )
    objects_out.append(
        {
            "layer": layer,
            "object_index": i,
            "class": next(r["class"] for r in records if r["index"] == i),
            "hidden_pixels": int(hidden_obj.sum()),
            "visible_pixels": int((layer_alpha[layer] & ~front).sum()),
            "parts": parts_out,
        }
    )
write_png(output / "inferable_mask.png", (inferable_mask * 255).astype(np.uint8))
write_png(output / "unknowable_mask.png", (unknowable_mask * 255).astype(np.uint8))
summary = {
    "seed": opts.seed,
    "split": opts.split,
    "objects": objects_out,
    "inferable_pixels": int(inferable_mask.sum()),
    "unknowable_pixels": int(unknowable_mask.sum()),
    "has_inferable_occlusion": bool(inferable_mask.sum() >= 150),
    "has_unknowable_occlusion": bool(unknowable_mask.sum() >= 150),
    "rule": "hidden part (>50% hidden) is inferable if >=25% visible (continuation) or a same-class sibling on the object is >=60% visible (symmetry); otherwise unknowable. Masks are in 640x640 pixel space.",
}
(output / "occlusion.json").write_text(json.dumps(summary, indent=2) + "\n")
(output / "scene.json").write_text(
    json.dumps(
        {
            "seed": opts.seed,
            "split": opts.split,
            "samples": opts.samples,
            "wall_hue": round(wall_h, 4),
            "floor": floor_kind,
            "camera_azimuth": round(cam_az, 4),
            "ortho_scale": round(camera.data.ortho_scale, 3),
            "light_energy": round(light.data.energy, 1),
            "objects": records,
            "layer_order_object_indices": order,
        },
        indent=2,
    )
    + "\n"
)
manifest = {
    "kind": "Rendered ground-truth scene from the randomized family, not model output",
    "layers_back_to_front": [
        {
            "index": layer,
            "object": "room" if i == 0 else next(r["class"] for r in records if r["index"] == i),
            "object_index": i,
            "path": f"layer_{layer}.png",
        }
        for layer, i in enumerate(order)
    ],
    "cross_object_cast_shadows": False,
    "source_definition": "Alpha composition of isolated 3D renders; complete hidden object surfaces and clean room are available.",
    "purpose": "Training/held-out scene for occlusion-aware layer decomposition; held-out scenes use reserved parameter ranges.",
}
(output / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            "seed": opts.seed,
            "objects": [r["class"] for r in records],
            "layers": len(order),
            "inferable": summary["inferable_pixels"],
            "unknowable": summary["unknowable_pixels"],
        }
    )
)
