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


class Camera:
    def __init__(self, window):
        self._window = window
        self._camera_pos = np.array((3.24, 1.86, -4.57))
        self._last_mouse_pos = np.array(self._window.get_cursor_pos())

    def update_camera(self):
        win = self._window
        if not win.is_pressed(ti.ui.CTRL):
            return False
        if not win.is_pressed(ti.ui.LMB):
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        delta2d = mouse_pos - self._last_mouse_pos
        delta = np.array([delta2d[0], delta2d[1], 0]) * 2
        self._last_mouse_pos = mouse_pos
        self._camera_pos += delta
        return True

    @property
    def position(self):
        return self._camera_pos


def main():
    # last_camera_pos = np.array((3.24, 1.86, -4.57))
    camera = Camera(window)
    renderer.set_camera_pos(*camera.position)
    renderer.floor_height[None] = -5e-3
    renderer.initialize_grid()

    total_voxels = renderer.total_non_empty_voxels()
    last_mouse_pos = np.array(window.get_cursor_pos())
    print('Total nonempty voxels', total_voxels)
    canvas = window.get_canvas()

    while window.running:
        show_hud()
        if camera.update_camera():
            renderer.set_camera_pos(*camera.position)
            renderer.reset_framebuffer()

        renderer.accumulate()
        img = renderer.fetch_image()
        canvas.set_image(img)
        window.show()


if __name__ == '__main__':
    main()
