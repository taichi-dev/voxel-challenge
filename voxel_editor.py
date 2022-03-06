import taichi as ti
import numpy as np
from renderer import Renderer

ti.init(arch=ti.vulkan)

GRID_RES = 32
SCREEN_RES = (640, 640)
renderer = Renderer(dx=1 / GRID_RES,
                    sphere_radius=0.3 / GRID_RES, res=SCREEN_RES)


window = ti.ui.Window("Voxel Editor", SCREEN_RES, vsync=True)
selected_voxel_color = (0.2, 0.2, 0.2)

SPP = 1

def show_hud():
  global selected_voxel_color
  window.GUI.begin("Options", 0.05, 0.05, 0.3, 0.2)
  selected_voxel_color = window.GUI.color_edit_3(
              "Voxel", selected_voxel_color)
  window.GUI.end()

def main():
    last_camera_pos = np.array((3.24, 1.86, -4.57))
    renderer.set_camera_pos(*last_camera_pos)
    renderer.floor_height[None] = -5e-3
    renderer.initialize_grid()

    total_voxels = renderer.total_non_empty_voxels()
    last_mouse_pos = np.array(window.get_cursor_pos())
    print('Total nonempty voxels', total_voxels)
    canvas = window.get_canvas()
    img = renderer.render_frame(spp=SPP)

    while window.running:
        show_hud()
        if window.is_pressed(ti.ui.LMB):
          mouse_pos = np.array(window.get_cursor_pos())
          delta2d = mouse_pos - last_mouse_pos
          delta = np.array([delta2d[0], delta2d[1], 0]) * 2
          print(f'mouse_pos={mouse_pos} delta={delta} last_camera_pos={last_camera_pos}')
          last_mouse_pos = mouse_pos
          last_camera_pos += delta
          renderer.set_camera_pos(*last_camera_pos)
          renderer.reset()
          img = renderer.render_frame(spp=SPP)
        canvas.set_image(img)
        window.show()


if __name__ == '__main__':
    main()
