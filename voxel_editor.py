import taichi as ti
import os
import time

ti.init(arch=ti.vulkan, device_memory_GB=1)

from renderer import Renderer

GRID_RES = 32
SCREEN_RES = (640, 360)
renderer = Renderer(dx=1 / GRID_RES,
                    sphere_radius=0.3 / GRID_RES, res=SCREEN_RES)



window = ti.ui.Window("Voxel Editor", SCREEN_RES, vsync=True)
selected_voxel_color = (0.2, 0.2, 0.2)

SPP = 10

def show_hud():
  global selected_voxel_color
  window.GUI.begin("Options", 0.05, 0.05, 0.3, 0.2)
  selected_voxel_color = window.GUI.color_edit_3(
              "Voxel", selected_voxel_color)
  window.GUI.end()

def main():
    canvas = window.get_canvas()
    renderer.set_camera_pos(3.24, 1.86, -4.57)
    renderer.floor_height[None] = -5e-3

    renderer.initialize_grid()

    total_voxels = renderer.total_non_empty_voxels()
    print('Total nonempty voxels', total_voxels)
    img = renderer.render_frame(spp=SPP)

    while window.running:
        show_hud()
        canvas.set_image(img)
        window.show()


if __name__ == '__main__':
    main()
