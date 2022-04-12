from scene import Scene

scene = Scene()

for i in range(10):
    scene.set_voxel(idx=(i, i, i), mat=2, color=(0.9, 0.5, 0.3))

for i in range(31):
    for j in range(31):
        is_light = int(j % 10 != 0)
        scene.set_voxel(idx=(j, i, -30), mat=is_light + 1, color=(1, 1, 1))

for i in range(0, 31):
    for j in range(0, 31):
        c = (i + j) % 2
        index = (i, 0, j - 30)
        scene.set_voxel(idx=index,
                        color=(c * 0.3 + 0.3, (1 - c) * 0.8 + 0.2, 1))

for i in range(31):
    scene.set_voxel(idx=(i, 1, 6), mat=2)

scene.finish()
