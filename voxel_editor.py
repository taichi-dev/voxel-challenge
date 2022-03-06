import taichi as ti
import numpy as np
from renderer import Renderer

ti.init(arch=ti.vulkan)

GRID_RES = 64
SCREEN_RES = (640, 640)
SPP = 2

window = ti.ui.Window("Voxel Editor", SCREEN_RES, vsync=True)
selected_voxel_color = (0.2, 0.2, 0.2)
in_edit_mode = False


def show_hud():
    global selected_voxel_color
    global in_edit_mode
    window.GUI.begin("Options", 0.05, 0.05, 0.3, 0.2)
    label = 'Edit' if in_edit_mode else 'View'
    if window.GUI.button(label):
        in_edit_mode = not in_edit_mode
    selected_voxel_color = window.GUI.color_edit_3(
        "Voxel", selected_voxel_color)
    window.GUI.end()


def np_normalize(v):
    # https://stackoverflow.com/a/51512965/12003165
    return v / np.sqrt(np.sum(v**2))


class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((3.24, 1.86, -4.57))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = np.array(self._window.get_cursor_pos())

    def update_camera(self):
        res = self._update_by_mouse()
        res = res or self._update_by_wasd()
        return res

    def _update_by_mouse(self):
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

    def _update_by_wasd(self):
        win = self._window
        fwd = np_normalize(self._lookat_pos - self._camera_pos)
        left = np.cross(self._up, fwd)
        lut = [
            ('w', fwd),
            ('a', left),
            ('s', -fwd),
            ('d', -left),
        ]
        dir = None
        for key, d in lut:
            if win.is_pressed(key):
                dir = d
                break
        if dir is None:
            return False
        dir = np.array(dir) * 0.05
        self._lookat_pos += dir
        self._camera_pos += dir
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos


def main():
    global in_edit_mode
    camera = Camera(window, up=(0, 1, 0))
    renderer = Renderer(grid_res=GRID_RES,
                        image_res=SCREEN_RES, taichi_logo=False)
    renderer.set_camera_pos(*camera.position)
    renderer.floor_height[None] = -5e-3
    renderer.initialize_grid()

    canvas = window.get_canvas()

    while window.running:
        show_hud()
        mouse_pos = tuple(window.get_cursor_pos())
        mouse_pos = [int(mouse_pos[i] * SCREEN_RES[i]) for i in range(2)]
        if window.is_pressed(ti.ui.LMB):
            find_solid = not in_edit_mode
            ijk = renderer.raycast_voxel_grid(mouse_pos, solid=find_solid)
            if ijk is not None:
                if in_edit_mode:
                    renderer.add_voxel(ijk)
                else:
                    renderer.set_voxel_color(ijk, (0.8, 0.6, .5))
                renderer.reset_framebuffer()
        elif window.is_pressed(ti.ui.RMB):
            ijk = renderer.raycast_voxel_grid(mouse_pos, solid=True)
            if ijk is not None:
                print(f'RMB hit! ijk={ijk}')
                renderer.delete_voxel(ijk)
                renderer.reset_framebuffer()
        if camera.update_camera():
            renderer.set_camera_pos(*camera.position)
            look_at = camera.look_at
            renderer.set_look_at(*look_at)
            renderer.reset_framebuffer()

        for i in range(SPP):
            renderer.accumulate()
        img = renderer.fetch_image()
        canvas.set_image(img)
        window.show()


if __name__ == '__main__':
    main()
