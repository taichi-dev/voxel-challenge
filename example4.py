from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
scene.set_background_color((0.3, 0.4, 0.6))


@ti.kernel
def initialize_voxels():
    n = 60
    for i, j, k in ti.ndrange((-n, n), (-n, n), (-n, n)):
        x = ivec3(i, j, k)
        if x.dot(x) < n * n * 0.5:
            scene.set_voxel(vec3(i, j, k), 1, vec3(0.9, 0.3, 0.3))


initialize_voxels()
scene.finish()
