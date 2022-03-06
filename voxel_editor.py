import taichi as ti
import os
import time

ti.init(arch=ti.cpu, device_memory_GB=1)

from renderer import Renderer

res = 128
renderer = Renderer(dx=1 / res,
                    sphere_radius=0.3 / res,
                    max_num_particles_million=1)

with_gui = True
gui = ti.GUI('Voxel Editor', (1280, 720))

spp = 10

def main_():
    for f in range(args.begin, args.end, args.step):
        print('frame', f, end=' ')
        output_fn = f'{output_folder}/{f:05d}.png'
        if os.path.exists(output_fn) and not args.force:
            print('skip.')
            continue
        else:
            print('rendering...')

        t = time.time()

        renderer.set_camera_pos(3.24, 1.86, -4.57)
        renderer.floor_height[None] = -5e-3

        cur_render_input = f'{args.in_dir}/{f:05d}.npz'
        if not os.path.exists(cur_render_input):
            print(f'warning, {cur_render_input} not existed, skip!')
            continue
        Path(output_fn).touch()
        renderer.initialize_particles_from_taichi_elements(cur_render_input)

        total_voxels = renderer.total_non_empty_voxels()
        total_inserted_particles = renderer.total_inserted_particles()
        print('Total particles (with motion blur)', total_inserted_particles)
        print('Total nonempty voxels', total_voxels)
        print('Average particle_list_length',
              total_inserted_particles / total_voxels)
        img = renderer.render_frame(spp=spp)

        if with_gui:
            gui.set_image(img)
            gui.show(output_fn)
        else:
            ti.imwrite(img, output_fn)
        ti.print_memory_profile_info()
        print(f'Frame rendered. {spp} take {time.time() - t} s.')


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
