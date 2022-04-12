from scene import Scene
import random

scene = Scene(exposure=10)

scene.set_floor(-0.05, (1.0, 1.0, 1.0))

n = 50
for i in range(n):
    for j in range(n):
        if i == 0 or j == 0 or i == n - 1 or j == n - 1:
            scene.set_voxel(idx=(i, 0, j), mat=2, color=(0.9, 0.1, 0.1))
        else:
            scene.set_voxel(idx=(i, 0, j), color=(0.9, 0.1, 0.1))

            if random.random() < 0.04:
                height = int(random.random() * 20)

                for k in range(1, height):
                    scene.set_voxel(idx=(i, k, j),
                                    mat=1,
                                    color=(0.0, 0.5, 0.9))
                if height:
                    scene.set_voxel(idx=(i, height, j), mat=2, color=(1, 1, 1))

scene.finish()
