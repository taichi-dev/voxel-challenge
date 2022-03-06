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
    camera = Camera(window, up=(0, 1, 0))
    renderer.set_camera_pos(*camera.position)
    renderer.floor_height[None] = -5e-3
    renderer.initialize_grid()

    total_voxels = renderer.total_non_empty_voxels()
    last_mouse_pos = np.array(window.get_cursor_pos())
    print('Total nonempty voxels', total_voxels)
    canvas = window.get_canvas()

    while window.running:
        show_hud()
        mouse_pos = tuple(window.get_cursor_pos())
        mouse_pos = [int(mouse_pos[i] * SCREEN_RES[i]) for i in range(2)]
        if window.is_pressed(ti.ui.LMB):
            hit, ijk, = renderer.raycast_voxel_grid(mouse_pos, solid=False)
            if hit:
                print(f'LMB hit! ijk={ijk}')
                renderer.add_voxel(ijk)
                renderer.reset_framebuffer()
        elif window.is_pressed(ti.ui.RMB):
            hit, ijk, = renderer.raycast_voxel_grid(mouse_pos, solid=True)
            if hit:
                print(f'RMB hit! ijk={ijk}')
                renderer.delete_voxel(ijk)
                renderer.reset_framebuffer()
        if camera.update_camera():
            renderer.set_camera_pos(*camera.position)
            look_at = camera.look_at
            renderer.set_look_at(*look_at)
            renderer.reset_framebuffer()

        renderer.accumulate()
        img = renderer.fetch_image()
        canvas.set_image(img)
        window.show()


if __name__ == '__main__':
    main()
