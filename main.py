from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=10)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))
scene.set_background_color((1.0, 0, 0))


@ti.kernel
def initialize_voxels():
    # Your code here! :-)
    scene.set_voxel(vec3(0, 0, 0), 2, vec3(0.9, 0.1, 0.1))


initialize_voxels()

scene.finish()
