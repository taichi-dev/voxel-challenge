import math
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import taichi as ti

from renderer import Renderer

VOXEL_DX = 1 / 32
SCREEN_RES = (1280, 720)
SPP = 10
UP_DIR = (0, 1, 0)
HELP_MSG = '''
=========================== Voxel Editor ===========================
Camera:
* Press Ctrl + Left Mouse to rotate the camera
* Press WASD to move the camera

Edit Mode:
* Move the mouse to highlight a voxel
* When a voxel is highlighted:
*   Press F to add a voxel around it
*   Press G to remove the highlighted voxel
*   Click Left Mouse to lock this voxel,
      so that you can edit its attributes
* Press Right Mouse to unlock a selected voxel

Save/Load:
* By default, saved voxels will be stored under
    `~/taichi_voxel_editor_saveslots/`
====================================================================
'''

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

    @property
    def mouse_exclusive_owner(self):
        return self._window.is_pressed(ti.ui.CTRL)

    def update_camera(self):
        res = self._update_by_mouse()
        res = res or self._update_by_wasd()
        return res

    def _update_by_mouse(self):
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.LMB):
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

        out_dir_homo = np.array(list(out_dir) + [
            0.0,
        ])
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

    def __init__(self, window, save_func: Callable = None, load_func: Callable = None, saveslots: Path = None):
        self._window = window
        self.in_edit_mode = False
        self.is_saveload_enabled = False
        self.save_func = save_func or (lambda: print("Save func needs to be defined!"))
        self.load_func = load_func or (lambda: print("Load func needs to be defined!"))
        self.saveslots = saveslots or Path.home() / Path("./taichi_voxel_editor_saveslots")
        self.saveslots.mkdir(parents=True, exist_ok=True)

    class UpdateStatus:

        def __init__(self):
            self.edit_mode_changed = False

    def update_edit_mode(self):
        res = HudManager.UpdateStatus()
        win = self._window
        win.GUI.begin('Mode', 0.02, 0.2, 0.15, 0.15)
        label = 'To View' if self.in_edit_mode else 'To Edit'
        if win.GUI.button(label):
            self.in_edit_mode = not self.in_edit_mode
            res.edit_mode_changed = True
        versioning = win.GUI.checkbox("Enable Save/Load", self.is_saveload_enabled)
        win.GUI.end()

        if versioning:
            win.GUI.begin("Save/Load", 0.02, 0.6, 0.25, 0.15)
            self.is_saveload_enabled = True
            if win.GUI.button("Save"):
                self.save_func()
            for f in self.saveslots.glob("taichi_voxel_*.npz"):
                if win.GUI.button(f.stem):
                    self.load_func(f)
            win.GUI.end()

        win.GUI.begin('Tutorial', 0.02, 0.55, 0.4, 0.4)
        win.GUI.text(HELP_MSG)
        win.GUI.end()
        return res

    def update_voxel_info(self, voxel_idx, renderer):
        win = self._window
        win.GUI.begin('Voxel', 0.02, 0.12, 0.15, 0.2)
        if voxel_idx is not None:
            vc = renderer.get_voxel_color(voxel_idx)
            vc = win.GUI.color_edit_3('Color', vc)
            renderer.set_voxel_color(voxel_idx, vc)
        else:
            win.GUI.text('No voxel selected')
        win.GUI.end()


def print_help():
    print(HELP_MSG)


class EditModeProcessor:

    def __init__(self, window, renderer):
        self._window = window
        self._renderer = renderer
        self._last_mouse_pos = None
        self._event_handled = False
        self._cur_hovered_voxel_idx = None
        self._voxel_locked = False

    def update_mouse_hovered_voxel(self, mouse_excluded):
        if mouse_excluded:
            return

        win = self._window
        if win.is_pressed(ti.ui.RMB):
            self._voxel_locked = False
        if self._voxel_locked:
            return

        mouse_pos = np.array(self._window.get_cursor_pos())
        mouse_pos_ss = [int(mouse_pos[i] * SCREEN_RES[i]) for i in range(2)]
        ijk = self._renderer.raycast_voxel_grid(mouse_pos_ss, solid=True)
        self._cur_hovered_voxel_idx = ijk
        if ijk is not None and win.is_pressed(ti.ui.LMB):
            self._voxel_locked = True

    def edit_grid(self):
        win = self._window
        renderer = self._renderer
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
        # TODO: Use a state machine to handle the logic
        if self._voxel_locked:
            return False

        mouse_moved = (mouse_pos != self._last_mouse_pos).any()
        self._last_mouse_pos = mouse_pos
        if not self._event_handled:
            # screen space
            mouse_pos_ss = [
                int(mouse_pos[i] * SCREEN_RES[i]) for i in range(2)
            ]
            ijk = self._cur_hovered_voxel_idx
            if win.is_pressed('f'):
                ijk = renderer.raycast_voxel_grid(mouse_pos_ss,
                                                solid=False)
                if ijk is not None:
                    renderer.add_voxel(ijk, color=(0.6, 0.7, 0.9))
                    self._event_handled = True
            elif win.is_pressed('g'):
                if ijk is not None:
                    renderer.delete_voxel(ijk)
                    self._event_handled = True
        elif win.get_events(ti.ui.RELEASE) or mouse_moved:
            self._event_handled = False
        should_rerender = mouse_moved or self._event_handled
        return should_rerender

    @property
    def cur_locked_voxel_idx(self):
        if not self._voxel_locked:
            return None
        return self._cur_hovered_voxel_idx


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
    
    # setup save/load funcs
    hud_mgr.save_func = partial(renderer.spit_local, hud_mgr.saveslots)
    hud_mgr.load_func = renderer.slurp_local

    renderer.set_camera_pos(*camera.position)
    renderer.floor_height[None] = -5e-3
    renderer.initialize_grid()

    canvas = window.get_canvas()
    edit_proc = EditModeProcessor(window, renderer)
    while window.running:
        mouse_excluded = camera.mouse_exclusive_owner
        hud_res = hud_mgr.update_edit_mode()
        should_reset_framebuffer = False
        if hud_mgr.in_edit_mode:
            edit_proc.update_mouse_hovered_voxel(mouse_excluded)
            should_reset_framebuffer = edit_proc.edit_grid()
            hud_mgr.update_voxel_info(
                edit_proc.cur_locked_voxel_idx, renderer)
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
