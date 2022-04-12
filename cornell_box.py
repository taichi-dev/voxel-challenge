from scene import Scene
import math

scene = Scene(voxel_edges=0, exposure=30)

scene.set_floor(0, (1.0, 1.0, 1.0))

n = 50
for i in range(n):
    for j in range(n):
        scene.set_voxel(idx=(0, i, j), color=(0.9, 0.3, 0.3))
        scene.set_voxel(idx=(n, i, j), color=(0.3, 0.9, 0.3))
        scene.set_voxel(idx=(i, n, j), color=(1, 1, 1))
        scene.set_voxel(idx=(i, 0, j), color=(1, 1, 1))
        scene.set_voxel(idx=(i, j, 0), color=(1, 1, 1))

for i in range(-n // 8, n // 8):
    for j in range(-n // 8, n // 8):
        scene.set_voxel(idx=(i + n // 2, n, j + n // 2),
                        mat=2,
                        color=(1, 1, 1))

for i in range(0, n // 4 * 3, 2):
    for j in range(n // 4 * 3):
        scene.set_voxel(idx=(j + n // 8, n // 4 + math.sin(
            (i + j) / n * 30) * 0.05 * n + i / 10, -i + n // 8 * 7),
                        color=(0.3, 0.3, 0.9))

scene.finish()
