from scene import Scene
import math

scene = Scene()

scene.set_floor(0, (1.0, 1.0, 1.0))

radius = 30
n = 1000
for i in range(n):
    t = i / n * 10

    for k in range(3):
        r = radius * k // 2
        scene.set_voxel(idx=(math.cos(t) * r, t * 2, math.sin(t) * r),
                        mat=2,
                        color=(0.9, 0.5, 0.3))

scene.finish()
