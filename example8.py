from scene import Scene
import taichi as ti
from taichi.math import *

night_mode = True
exposure = 1.0 + night_mode * 4.0
foam_material = 1.0 + night_mode * 1
foam_color = vec3(0.7, 0.8, 1.0)

scene = Scene(voxel_edges = 0, exposure = exposure)
scene.set_floor(-20, (0.6, 0.8, 1.0))
scene.set_directional_light((1, 1, 0), 0.2, vec3(1.0, 1.0, 1.0) / exposure)
scene.set_background_color(vec3(0.6, 0.8, 1.0) / exposure)

@ti.func
def create_ocean_base(pos, size, color):
    for ik in ti.grouped(ti.ndrange((0, size[0]), (0, size[2]))):    
        t, r = ((ti.sin(ik[0] / 23.0 * 3.14) * ti.sin(ik[1] / 27.0 * 3.14) + 1) / 2.0), ti.random()
        h = (t - 0.1 * r)* size[1] + (1 - t + 0.1 * r) * size[1] / 2
        for j in range(int(h)):
            scene.set_voxel(pos + ivec3(ik[0], j, ik[1]), 1, (0.3 + 0.7 * j / h) * color)
        if r < 0.02:
            scene.set_voxel(pos + ivec3(ik[0], int(h) - 1, ik[1]), foam_material, foam_color)

@ti.func
def create_wave(pos, radius, color, portion, flipped):
    for I in ti.grouped(ti.ndrange((-radius, radius), (-radius, radius), (-radius, radius))):
        uv = vec2(I[0], I[1]) / radius
        theta = ti.atan2(uv[1], uv[0]) / 3.14 * 2
        offset = I
        offset[0] *=  1 - flipped * 2        
        if theta >= 0 and theta <= portion:
            if abs(uv.norm() - 0.95) < 0.05 + 0.05 *ti.random():
                if 1 - ti.random()**2 < theta / portion - 0.1:
                    scene.set_voxel(pos + offset, foam_material, foam_color)
                else:
                    scene.set_voxel(pos + offset, 1, color)
        elif theta <= 0 and theta >= -1:
            if uv.norm() > 0.9 - 0.05 *ti.random():
                scene.set_voxel(pos + offset, 1, color)

@ti.func
def create_moon(pos, radius, color):
    for I in ti.grouped(ti.ndrange((-radius, radius), (-radius, radius), (-radius, radius))):
        if I.norm() < radius:
           scene.set_voxel(pos + I, 2, color)

@ti.kernel
def initialize_voxels():
    create_ocean_base(ivec3(-60, -40, -60), ivec3(120, 20, 120), vec3(0.2, 0.4, 1.0))

    create_wave(ivec3(-20, 0, -20), 40, vec3(0.2, 0.4, 1.0), 1, True)
    create_wave(ivec3(29, -5, 29), 30, vec3(0.2, 0.4, 1.0), 0.5, False)

    create_wave(ivec3(-20, -15, 15), 20, vec3(0.2, 0.4, 1.0), 0.7, True)
    create_wave(ivec3(-57, -15, 15), 20, vec3(0.2, 0.4, 1.0), 0.0, False)
    
    create_wave(ivec3(20, -15, -39), 20, vec3(0.2, 0.4, 1.0), 0.56, False)
    create_wave(ivec3(57, -15, -39), 20, vec3(0.2, 0.4, 1.0), 0.0, True)

    if night_mode:
        create_moon(ivec3(40, 40, -40), 10, vec3(1.0, 1.0, 0.1))

initialize_voxels()

scene.finish()
