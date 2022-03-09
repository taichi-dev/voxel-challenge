from curses import napms
import math
import taichi as ti
import numpy as np
from renderer import Renderer

VOXEL_DX = 1 / 32
SCREEN_RES = (1280, 720)
SPP = 10
UP_DIR = (0, 1, 0)


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
        self._camera_pos = np.array((1.0, 1.5, 2.0))
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
            ('e', [0, 1, 0]),
            ('q', [0, -1, 0]),
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


class HudManager:
    def __init__(self, window):
        self._window = window
        self.in_edit_mode = False

    class UpdateStatus:
        def __init__(self):
            self.edit_mode_changed = False

    def update(self):
        res = HudManager.UpdateStatus()
        win = self._window
        win.GUI.begin('Options', 0.02, 0.9, 0.15, 0.08)
        label = 'To View Mode' if self.in_edit_mode else 'To Edit Mode'
        if win.GUI.button(label):
            self.in_edit_mode = not self.in_edit_mode
            res.edit_mode_changed = True
        # self.voxel_color = win.GUI.color_edit_3(
        #     'Voxel', self.voxel_color)
        win.GUI.end()
        return res


def print_help():
    msg = '''
=========================== Voxel Editor ===========================
Camera:
* Press Ctrl + Left Mouse to rotate the camera
* Press WASD to move the camera

Edit Mode:
* Press Left Mouse to add a voxel
* Press Right Mouse to remove a highlighted voxel
====================================================================
    '''
    print(msg)


class EditModeProcessor:
    def __init__(self, window, renderer):
        self._window = window
        self._renderer = renderer
        self._event_handled = False
        self._last_mouse_pos = None
        self._mouse_moved = False

    def process(self):
        win = self._window
        renderer = self._renderer
        mouse_pos = np.array(win.get_cursor_pos())
        mov_delta = 0
        if self._last_mouse_pos is not None:
            d = mouse_pos - self._last_mouse_pos
            mov_delta = np.dot(d, d)
        self._mouse_moved = (mouse_pos != self._last_mouse_pos).any()
        # TODO: Use a state machine to handle the logic
        if not self._event_handled:
            self._last_mouse_pos = mouse_pos
            # screen space
            mouse_pos_ss = [int(mouse_pos[i] * SCREEN_RES[i])
                            for i in range(2)]
            ijk = renderer.raycast_voxel_grid(mouse_pos_ss, solid=True)
            if win.is_pressed(ti.ui.LMB):
                ijk = renderer.raycast_voxel_grid(mouse_pos_ss, solid=False)
                if ijk is not None:
                    renderer.add_voxel(ijk, color=(0.6, 0.7, 0.9))
                    self._event_handled = True
            elif win.is_pressed(ti.ui.RMB):
                if ijk is not None:
                    renderer.delete_voxel(ijk)
                    self._event_handled = True
        elif win.get_events(ti.ui.RELEASE) or mov_delta > 1e-5:
            self._event_handled = False


def main():
    ti.init(arch=ti.vulkan)
    print_help()
    window = ti.ui.Window("Voxel Editor", SCREEN_RES, vsync=True)
    camera = Camera(window, up=UP_DIR)
    hud_mgr = HudManager(window)
    renderer = Renderer(dx=VOXEL_DX,
                        image_res=SCREEN_RES,
                        up=UP_DIR,
                        taichi_logo=False)
    renderer.set_camera_pos(*camera.position)
    renderer.floor_height[None] = -5e-3
    renderer.initialize_grid()

    canvas = window.get_canvas()
    edit_proc = EditModeProcessor(window, renderer)
    while window.running:
        hud_res = hud_mgr.update()
        should_reset_framebuffer = False
        if hud_mgr.in_edit_mode:
            edit_proc.process()
            should_reset_framebuffer = edit_proc._event_handled or edit_proc._mouse_moved
            # should_reset_framebuffer = True
        elif hud_res.edit_mode_changed:
            renderer.clear_cast_voxel()

        if camera.update_camera():
            renderer.set_camera_pos(*camera.position)
            look_at = camera.look_at
            renderer.set_look_at(*look_at)
            should_reset_framebuffer = True

        if should_reset_framebuffer:
            renderer.reset_framebuffer()

        for _ in range(SPP):
            renderer.accumulate()
        img = renderer.fetch_image()
        canvas.set_image(img)
        window.show()


if __name__ == '__main__':
    main()
