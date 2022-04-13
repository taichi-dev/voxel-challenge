from scene import Scene

scene = Scene()

scene.set_floor(0, (0.5, 0.5, 1))

for i in range(31):
    for j in range(31):
        is_light = int(j % 10 != 0)
        scene.set_voxel(idx=(j, i, -30), mat=is_light + 1, color=(1, 1, 1))

        color = max(i, j)
        if color % 2 == 0:
            scene.set_voxel(idx=(0, i, j - 30),
                            color=((color % 3 // 2) * 0.5 + 0.5,
                                   ((color + 1) % 3 // 2) * 0.5 + 0.5,
                                   ((color + 2) % 3 // 2) * 0.5 + 0.5))

for i in range(31):
    for j in range(31):
        c = (i + j) % 2
        index = (i, 0, j - 30)
        scene.set_voxel(idx=index,
                        color=(c * 0.3 + 0.3, (1 - c) * 0.8 + 0.2, 1))

scene.finish()
