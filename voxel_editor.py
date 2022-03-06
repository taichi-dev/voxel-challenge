import taichi as ti
import os
import time

ti.init(arch=ti.vulkan, device_memory_GB=1)

from renderer import Renderer

res = 32
screen_res = (640, 360)
renderer = Renderer(dx=1 / res,
                    sphere_radius=0.3 / res, res=screen_res,
                    max_num_particles_million=1)



window = ti.ui.Window("Voxel Editor", screen_res, vsync=True)
selected_voxel_color = (0.2, 0.2, 0.2)

spp = 10

def main():
    canvas = window.get_canvas()
    renderer.set_camera_pos(3.24, 1.86, -4.57)
    renderer.floor_height[None] = -5e-3

    renderer.initialize_grid()

    total_voxels = renderer.total_non_empty_voxels()
    print('Total nonempty voxels', total_voxels)
    img = renderer.render_frame(spp=spp)

    while True:
        canvas.set_image(img)
        window.show()


main()
