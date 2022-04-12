from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable
import time

import numpy as np
import taichi as ti

from renderer import Renderer, StorageBackend, SAVESLOT_FORMAT, MYNAME
from np_utils import np_normalize, np_rotate_matrix

VOXEL_DX = 1 / 64
SCREEN_RES = (1280, 720)
TARGET_FPS = 30
UP_DIR = (0, 1, 0)
HELP_MSG = '''
=========================== Voxel Editor ===========================
Camera:
* Press Ctrl + Left Mouse to rotate the camera
* Press W/A/S/D/Q/E to move the camera
====================================================================
'''


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
            ('e', [0, -1, 0]),
            ('q', [0, 1, 0]),
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
    def __init__(self,
                 window,
                 save_func: Callable = None,
                 load_func: Callable = None,
                 saveslots: Path = None):
        self._window = window
        self.in_edit_mode = False
        self.is_saveload_enabled = False
        self.save_func = save_func or (
            lambda: print("Save func needs to be defined!"))
        self.load_func = load_func or (
            lambda: print("Load func needs to be defined!"))
        self.saveslots = saveslots or Path.home() / Path(
            f"./{MYNAME}_saveslots")
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
        versioning = win.GUI.checkbox("Enable Save/Load",
                                      self.is_saveload_enabled)
        win.GUI.end()

        if versioning:
            win.GUI.begin("Save/Load", 0.02, 0.6, 0.25, 0.15)
            self.is_saveload_enabled = True
            if win.GUI.button("Save"):
                self.save_func()
            # TODO: move the display logic to a SaveLoadProcessor
            # and define multimethods for multiple storage backends
            loadable_slots = self.saveslots.glob(f"{MYNAME}_*.npz")
            for f in sorted(
                    loadable_slots,
                    key=lambda f: datetime.strptime(
                        f.stem.split(f"{MYNAME}_")[-1], SAVESLOT_FORMAT),
                    reverse=True):
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

    @property
    def cur_locked_voxel_idx(self):
        if not self._voxel_locked:
            return None
        return self._cur_hovered_voxel_idx


class Scene:
    def __init__(self):
        ti.init(arch=ti.vulkan)
        print_help()
        self.window = ti.ui.Window("Voxel Editor", SCREEN_RES, vsync=True)
        self.camera = Camera(self.window, up=UP_DIR)
        self.hud_mgr = HudManager(self.window)
        self.renderer = Renderer(dx=VOXEL_DX,
                                 image_res=SCREEN_RES,
                                 up=UP_DIR,
                                 taichi_logo=False)

        # hard-code to local storage for now
        storage = StorageBackend.LOCAL
        print(
            f'[{MYNAME}] You are currently using {storage.name} as storage backend.'
        )

        # setup save/load funcs
        self.hud_mgr.save_func = partial(self.renderer.spit, storage,
                                         self.hud_mgr.saveslots)
        self.hud_mgr.load_func = partial(self.renderer.slurp, storage)

        self.renderer.set_camera_pos(*self.camera.position)
        self.renderer.floor_height[None] = -5e-3

    def set_voxel(self, idx, mat=1, color=(1, 1, 1)):
        self.renderer.add_voxel(idx, mat, color)

    def erase_voxel(self, idx):
        self.renderer.erase_voxel(idx)

    def set_floor(self, height, color):
        self.renderer.floor_height[None] = height
        self.renderer.floor_color[None] = color

    def finish(self):
        self.renderer.recompute_bbox()
        canvas = self.window.get_canvas()
        edit_proc = EditModeProcessor(self.window, self.renderer)
        spp = 1
        while self.window.running:
            mouse_excluded = self.camera.mouse_exclusive_owner
            hud_res = self.hud_mgr.update_edit_mode()
            should_reset_framebuffer = False

            if self.camera.update_camera():
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                should_reset_framebuffer = True

            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()

            t = time.time()
            for _ in range(spp):
                self.renderer.accumulate()
            img = self.renderer.fetch_image()
            canvas.set_image(img)
            elapsed_time = time.time() - t
            if elapsed_time * TARGET_FPS > 1:
                spp = max(spp - 1, 1)
            else:
                spp += 1
            self.window.show()
