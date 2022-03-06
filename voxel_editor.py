import taichi as ti
import os
import time

ti.init(arch=ti.cpu, device_memory_GB=1)

from renderer import Renderer

res = 32
screen_res = 640, 360
renderer = Renderer(dx=1 / res,
                    sphere_radius=0.3 / res, res=screen_res,
                    max_num_particles_million=1)


with_gui = True
gui = ti.GUI('Voxel Editor', screen_res)

spp = 10

def main():
    t = time.time()

    renderer.set_camera_pos(3.24, 1.86, -4.57)
    renderer.floor_height[None] = -5e-3

    renderer.initialize_particles_from_taichi_elements()

    total_voxels = renderer.total_non_empty_voxels()
    total_inserted_particles = renderer.total_inserted_particles()
    print('Total particles (with motion blur)', total_inserted_particles)
    print('Total nonempty voxels', total_voxels)
    print('Average particle_list_length',
          total_inserted_particles / total_voxels)
    img = renderer.render_frame(spp=spp)

    while True:
        gui.set_image(img)
        gui.show()
    ti.print_memory_profile_info()
    print(f'Frame rendered. {spp} take {time.time() - t} s.')


main()
