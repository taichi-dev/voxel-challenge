import math
import taichi as ti
import numpy as np
from renderer import Renderer

ti.init(arch=ti.vulkan)

VOXEL_DX = 1 / 32
SCREEN_RES = (640, 640)
SPP = 2
UP_DIR = (0, 1, 0)

window = ti.ui.Window("Voxel Editor", SCREEN_RES, vsync=True)
selected_voxel_color = (0.2, 0.2, 0.2)
in_edit_mode = False


def show_hud():
    global selected_voxel_color
    global in_edit_mode
    window.GUI.begin("Options", 0.05, 0.05, 0.3, 0.2)
    label = 'View' if in_edit_mode else 'Edit'
    if window.GUI.button(label):
        in_edit_mode = not in_edit_mode
    selected_voxel_color = window.GUI.color_edit_3(
        "Voxel", selected_voxel_color)
    window.GUI.end()


def np_normalize(v):
    # https://stackoverflow.com/a/51512965/12003165
    return v / np.sqrt(np.sum(v**2))


def np_rotate_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # https://stackoverflow.com/a/6802723/12003165
    axis = np_normalize(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((1.0, 1.5, -2.0))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None

    def update_camera(self):
        res = self._update_by_mouse()
        res = res or self._update_by_wasd()
        return res

    def _update_by_mouse(self):
        win = self._window
        if not win.is_pressed(ti.ui.CTRL) or not win.is_pressed(ti.ui.LMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        # Makes camera rotation feels right
        dx, dy = self._last_mouse_pos - mouse_pos
        self._last_mouse_pos = mouse_pos

        out_dir = self._camera_pos - self._lookat_pos
        leftdir = self._compute_left_dir(np_normalize(-out_dir))

        rotx = np_rotate_matrix(self._up, dx)
        roty = np_rotate_matrix(leftdir, dy)

        out_dir_homo = np.array(list(out_dir) + [0.0, ])
        new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        self._camera_pos = self._lookat_pos + new_out_dir

        return True

    def _update_by_wasd(self):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ('w', tgtdir),
            ('a', leftdir),
            ('s', -tgtdir),
            ('d', -leftdir),
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

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)


def main():
    global in_edit_mode
    camera = Camera(window, up=UP_DIR)
    renderer = Renderer(dx=VOXEL_DX,
                        image_res=SCREEN_RES,
                        up=UP_DIR,
                        taichi_logo=False)
    renderer.set_camera_pos(*camera.position)
    renderer.floor_height[None] = -5e-3
    renderer.initialize_grid()

    canvas = window.get_canvas()

    while window.running:
        show_hud()
        mouse_pos = tuple(window.get_cursor_pos())
        # screen space
        mouse_pos_ss = [int(mouse_pos[i] * SCREEN_RES[i]) for i in range(2)]
        renderer.raycast_voxel_grid(mouse_pos_ss, solid=True)
        if window.is_pressed(ti.ui.LMB):
            find_solid = not in_edit_mode
            ijk = renderer.raycast_voxel_grid(mouse_pos_ss, solid=find_solid)
            if ijk is not None:
                if in_edit_mode:
                    renderer.add_voxel(ijk)
                else:
                    renderer.set_voxel_color(ijk, (0.8, 0.6, .5))
                renderer.reset_framebuffer()
        elif window.is_pressed(ti.ui.RMB):
            ijk = renderer.raycast_voxel_grid(mouse_pos_ss, solid=True)
            if ijk is not None:
                renderer.delete_voxel(ijk)
                renderer.reset_framebuffer()
        if camera.update_camera():
            renderer.set_camera_pos(*camera.position)
            look_at = camera.look_at
            renderer.set_look_at(*look_at)
            renderer.reset_framebuffer()

        for _ in range(SPP):
            renderer.accumulate()
        img = renderer.fetch_image()
        canvas.set_image(img)
        window.show()


if __name__ == '__main__':
    main()
