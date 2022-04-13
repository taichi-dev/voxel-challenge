from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene()
scene.set_floor(0, (0.5, 0.5, 1))


@ti.kernel
def initialize_voxels():
    for i in range(31):
        for j in range(31):
            is_light = int(j % 10 != 0)
            scene.set_voxel(vec3(j, i, -30), is_light + 1, vec3(1, 1, 1))
            color = max(i, j)
            if color % 2 == 0:
                scene.set_voxel(
                    vec3(0, i, j - 30), 1,
                    vec3((color % 3 // 2) * 0.5 + 0.5,
                         ((color + 1) % 3 // 2) * 0.5 + 0.5,
                         ((color + 2) % 3 // 2) * 0.5 + 0.5))

    for i in range(31):
        for j in range(31):
            c = (i + j) % 2
            index = vec3(i, 0, j - 30)
            scene.set_voxel(index, 1,
                            vec3(c * 0.3 + 0.3, (1 - c) * 0.8 + 0.2, 1))


initialize_voxels()

scene.finish()
