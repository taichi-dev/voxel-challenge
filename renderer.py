from datetime import datetime
from pathlib import Path

import numpy as np
import taichi as ti

from renderer_utils import (eps, inf, inside_taichi, out_dir,
                            ray_aabb_intersection)

MAX_RAY_DEPTH = 4
use_directional_light = True

DIS_LIMIT = 100

EXPOSURE = 3


@ti.data_oriented
class Renderer:

    def __init__(self, dx, image_res, up, taichi_logo=True):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.taichi_logo = taichi_logo
        self.current_spp = 0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.voxel_color = ti.Vector.field(3, dtype=ti.u8)
        self.voxel_material = ti.field(dtype=ti.u8)

        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_direction_noise = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.cast_voxel_hit = ti.field(ti.i32, shape=())
        self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

        self.voxel_edges = 0.06

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.floor_height = ti.field(dtype=ti.f32, shape=())

        self.supporter = 2
        # What's the difference between `voxel_inv_dx` and `voxel_grid_res`...?
        self.voxel_dx = dx
        self.voxel_inv_dx = 1 / dx
        self.voxel_grid_res = 128
        voxel_grid_offset = [-self.voxel_grid_res // 2 for _ in range(3)]

        ti.root.dense(ti.ij, image_res).place(self.color_buffer)
        ti.root.dense(ti.ijk,
                      self.voxel_grid_res).place(self.voxel_color,
                                                 self.voxel_material,
                                                 offset=voxel_grid_offset)

        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.set_up(*up)
        self.set_fov(0.23)

        self.light_direction[None] = [1.2, 0.3, 0.7]
        self.light_direction_noise[None] = 0.03
        L = 0.0  # Turn off directional light
        self.light_color[None] = [L, L, L]

    @ti.func
    def inside_grid(self, ipos):
        return ipos.min() >= -self.voxel_grid_res // 2 and ipos.max(
        ) < self.voxel_grid_res // 2

    # The dda algorithm requires the voxel grid to have one surrounding layer of void region
    # to correctly render the outmost voxel faces
    @ti.func
    def inside_grid_loose(self, ipos):
        return ipos.min() >= -self.voxel_grid_res // 2 - 1 and ipos.max(
        ) <= self.voxel_grid_res // 2

    @ti.func
    def query_density(self, ipos):
        inside = self.inside_grid(ipos)
        ret = 0.0
        if inside:
            ret = self.voxel_material[ipos]
        else:
            ret = 0.0
        return ret

    @ti.func
    def _to_voxel_index(self, pos):
        p = pos * self.voxel_inv_dx
        voxel_index = ti.floor(p).cast(ti.i32)
        return voxel_index

    @ti.func
    def voxel_surface_color(self, pos):
        p = pos * self.voxel_inv_dx
        p -= ti.floor(p)
        voxel_index = self._to_voxel_index(pos)

        boundary = self.voxel_edges
        count = 0
        for i in ti.static(range(3)):
            if p[i] < boundary or p[i] > 1 - boundary:
                count += 1

        f = 0.0
        if count >= 2:
            f = 1.0

        voxel_color = ti.Vector([0.0, 0.0, 0.0])
        is_light = 0
        if self.inside_particle_grid(voxel_index):
            voxel_color = self.voxel_color[voxel_index] * (1.0 / 255)
            if self.voxel_material[voxel_index] == 2:
                is_light = 1

        return voxel_color * (1.3 - 1.2 * f), is_light

    @ti.func
    def sdf(self, o):
        dist = 0.0
        if ti.static(self.supporter == 0):
            o -= ti.Vector([0.5, 0.002, 0.5])
            p = o
            h = 0.02
            ra = 0.29
            rb = 0.005
            d = (ti.Vector([p[0], p[2]]).norm() - 2.0 * ra + rb, abs(p[1]) - h)
            dist = min(max(d[0], d[1]), 0.0) + ti.Vector(
                [max(d[0], 0.0), max(d[1], 0)]).norm() - rb
        elif ti.static(self.supporter == 1):
            o -= ti.Vector([0.5, 0.002, 0.5])
            dist = (o.abs() - ti.Vector([0.5, 0.02, 0.5])).max()
        else:
            dist = o[1] - self.floor_height[None]

        return dist

    @ti.func
    def ray_march(self, p, d):
        j = 0
        dist = 0.0
        limit = 200
        while j < limit and self.sdf(p + dist * d) > 1e-8 and dist < DIS_LIMIT:
            dist += self.sdf(p + dist * d)
            j += 1
        if dist > DIS_LIMIT:
            dist = inf
        return dist

    @ti.func
    def sdf_normal(self, p):
        d = 1e-3
        n = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            inc = p
            dec = p
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self.sdf(inc) - self.sdf(dec))
        return n.normalized()

    @ti.func
    def sdf_color(self, p):
        scale = 0.0
        if ti.static(self.taichi_logo):
            scale = 0.4
            if inside_taichi(ti.Vector([p[0], p[2]])):
                scale = 1
        else:
            scale = 1.0
        return ti.Vector([0.3, 0.5, 0.7]) * scale

    @ti.func
    def dda_voxel(self, eye_pos, d):
        for i in ti.static(range(3)):
            if abs(d[i]) < 1e-6:
                d[i] = 1e-6
        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]
        inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos,
                                                 d)
        hit_distance = inf
        hit_light = 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        voxel_index = ti.Vector([0, 0, 0])
        if inter:
            near = max(0, near)

            pos = eye_pos + d * (near + 5 * eps)

            o = self.voxel_inv_dx * pos
            ipos = int(ti.floor(o))
            dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
            running = 1
            i = 0
            hit_pos = ti.Vector([0.0, 0.0, 0.0])
            while running:
                last_sample = int(self.query_density(ipos))
                if not self.inside_particle_grid(ipos):
                    running = 0

                if last_sample:
                    mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) -
                            rsign * 0.5) * rinv
                    hit_distance = mini.max() * self.voxel_dx + near
                    hit_pos = eye_pos + (hit_distance + 1e-3) * d
                    voxel_index = self._to_voxel_index(hit_pos)
                    c, hit_light = self.voxel_surface_color(hit_pos)
                    running = 0
                else:
                    mm = ti.Vector([0, 0, 0])
                    if dis[0] <= dis[1] and dis[0] < dis[2]:
                        mm[0] = 1
                    elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                        mm[1] = 1
                    else:
                        mm[2] = 1
                    dis += mm * rsign * rinv
                    ipos += mm * rsign
                    normal = -mm * rsign
                i += 1
        return hit_distance, normal, c, hit_light, voxel_index

    @ti.kernel
    def raycast(self, mouse_x: ti.i32, mouse_y: ti.i32, offset: ti.f32):
        closest = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        d = self.get_cast_dir(mouse_x, mouse_y)
        closest, normal, c, _, vx_idx = self.dda_voxel(self.camera_pos[None],
                                                       d)

        if closest < inf:
            self.cast_voxel_hit[None] = 1
            p = self.camera_pos[None] + (closest + offset) * d
            self.cast_voxel_index[None] = self._to_voxel_index(p)
        else:
            self.cast_voxel_hit[None] = 0

    @ti.func
    def inside_particle_grid(self, ipos):
        pos = ipos * self.voxel_dx
        return self.bbox[0][0] <= pos[0] and pos[0] < self.bbox[1][
            0] and self.bbox[0][1] <= pos[1] and pos[1] < self.bbox[1][
                1] and self.bbox[0][2] <= pos[2] and pos[2] < self.bbox[1][2]

    @ti.func
    def next_hit(self, pos, d, t):
        closest = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        hit_light = 0
        closest, normal, c, hit_light, vx_idx = self.dda_voxel(pos, d)

        if d[2] != 0:
            ray_closest = -(pos[2] + 5.5) / d[2]
            if ray_closest > 0 and ray_closest < closest:
                closest = ray_closest
                normal = ti.Vector([0.0, 0.0, 1.0])
                c = ti.Vector([0.6, 0.7, 0.7])

        ray_march_dist = self.ray_march(pos, d)
        if ray_march_dist < DIS_LIMIT and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.sdf_normal(pos + d * closest)
            c = self.sdf_color(pos + d * closest)

        # Highlight the selected voxel
        if self.cast_voxel_hit[None]:
            cast_vx_idx = self.cast_voxel_index[None]
            if all(cast_vx_idx == vx_idx):
                c = ti.Vector([1.0, 0.65, 0.0])
                # For light sources, we actually invert the material to make it
                # more obvious
                hit_light = 1 - hit_light
        return closest, normal, c, hit_light

    @ti.kernel
    def set_camera_pos(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.camera_pos[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_up(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.up[None] = ti.Vector([x, y, z]).normalized()

    @ti.kernel
    def set_look_at(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.look_at[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_fov(self, fov: ti.f32):
        self.fov[None] = fov

    @ti.func
    def get_cast_dir(self, u, v):
        fov = self.fov[None]
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        fu = (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
              fov * self.aspect_ratio - 1e-5)
        fv = 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5
        du = d.cross(self.up[None]).normalized()
        dv = du.cross(d).normalized()
        d = (d + fu * du + fv * dv).normalized()
        return d

    @ti.kernel
    def render(self):
        ti.block_dim(256)
        for u, v in self.color_buffer:
            d = self.get_cast_dir(u, v)
            pos = self.camera_pos[None]
            t = 0.0

            contrib = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            c = ti.Vector([1.0, 1.0, 1.0])

            depth = 0
            hit_light = 0

            while depth < MAX_RAY_DEPTH:
                closest, normal, c, hit_light = self.next_hit(pos, d, t)
                hit_pos = pos + closest * d
                depth += 1
                ray_depth = depth
                if not hit_light and normal.norm() != 0:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c

                    if ti.static(use_directional_light):
                        dir_noise = ti.Vector([
                            ti.random() - 0.5,
                            ti.random() - 0.5,
                            ti.random() - 0.5
                        ]) * self.light_direction_noise[None]
                        direct = (self.light_direction[None] +
                                  dir_noise).normalized()
                        dot = direct.dot(normal)
                        if dot > 0:
                            hit_light_ = 0
                            dist, _, _, hit_light_ = self.next_hit(
                                pos, direct, t)
                            if dist > DIS_LIMIT:
                                contrib += throughput * \
                                    self.light_color[None] * dot
                else:  # hit sky or light
                    depth = MAX_RAY_DEPTH

                max_c = throughput.max()
                if ti.random() > max_c:
                    depth = MAX_RAY_DEPTH
                    throughput = [0, 0, 0]
                else:
                    throughput /= max_c

            if hit_light:
                contrib += throughput * c
            else:
                throughput *= 0

            self.color_buffer[u, v] += contrib

    @ti.kernel
    def _render_to_image(self, samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.image_res[0]
            v = 1.0 * j / self.image_res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0])**2 +
                (v - self.vignette_center[1])**2) - self.vignette_radius), 0)

            for c in ti.static(range(3)):
                self._rendered_image[i, j][c] = ti.sqrt(
                    self.color_buffer[i, j][c] * darken * EXPOSURE / samples)

    @ti.kernel
    def total_non_empty_voxels(self) -> ti.i32:
        counter = 0

        for I in ti.grouped(self.voxel_material):
            if self.voxel_material[I] > 0:
                counter += 1

        return counter

    def initialize_grid(self):
        for i in range(3):
            self.bbox[0][i] = -1
            self.bbox[1][i] = 1
            print(f'Bounding box dim {i}: {self.bbox[0][i]} {self.bbox[1][i]}')

        for i in range(31):
            for j in range(31):
                is_light = int(j % 10 != 0)
                self.voxel_material[j, i, -30] = is_light + 1
                self.voxel_color[j, i, -30] = [255, 255, 255]

        for i in range(0, 31):
            for j in range(0, 31):
                index = (i, 0, j - 30)
                self.voxel_material[index] = 1
                c = (i + j) % 2
                self.voxel_color[index] = [
                    c * 55 + 200, (1 - c) * 55 + 200, 255
                ]

        for i in range(31):
            index = (i, 1, 6)
            self.voxel_material[index] = 2
            self.voxel_color[index] = [255, 255, 255]

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    def accumulate(self):
        self.render()
        self.current_spp += 1

    def fetch_image(self):
        self._render_to_image(self.current_spp)
        return self._rendered_image

    def raycast_voxel_grid(self, mouse_pos, solid):
        """
        Parameters:
          mouse_pos: the mouse position, in pixels
          solid: return the first solid cell or the last empty cell along the ray

        Returns:
          ijk: of the selected grid pos, None if not found
        """
        if solid:
            offset = 1e-3
        else:
            offset = -1e-3

        self.raycast(mouse_pos[0], mouse_pos[1], offset)

        if self.cast_voxel_hit[None]:
            return self.cast_voxel_index[None]
        return None

    @ti.kernel
    def clear_cast_voxel(self):
        self.cast_voxel_hit[None] = 0
        self.cast_voxel_index[None] = ti.Vector([0, 0, 0])

    def add_voxel(self, ijk, mat=1, color=(0.5, 0.5, 0.5)):
        ijk = tuple(ijk)
        self.voxel_material[ijk] = mat
        self.voxel_color[ijk] = [int(color[i] * 255) for i in range(3)]

    def delete_voxel(self, ijk):
        self.voxel_material[tuple(ijk)] = 0

    def set_voxel_color(self, ijk, color):
        ijk = tuple(ijk)
        self.voxel_color[ijk] = [int(color[i] * 255) for i in range(3)]

    def get_voxel_color(self, ijk):
        ijk = tuple(ijk)
        res = self.voxel_color[ijk]
        return tuple([float(x) / 255.0 for x in res])

    def spit_local(self, dir: Path):
        """
        Spit to `dir` on local storage.

        Depends on the filesystem, this function is likely to
        hit permission issues, when it happens, run the editor
        with sudo permission.
        """
        to_save = dir / Path(f'taichi_voxel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
        try:
            np.savez(to_save, voxel_material=self.voxel_material.to_numpy(), voxel_color=self.voxel_color.to_numpy())
            print(f"Saved to {to_save}")
        except PermissionError:
            print(f"Failed to save {to_save}, try start the editor with `sudo` mode?")

    def slurp_local(self, to_slurp: Path):
        """Slurp from local storage for `to_slurp`."""
        if to_slurp.exists():
           slurped = np.load(to_slurp, allow_pickle=False)
           _voxel_material, _voxel_color = slurped["voxel_material"], slurped["voxel_color"]
           self.voxel_material.from_numpy(_voxel_material)
           self.voxel_color.from_numpy(_voxel_color)
        #    self.clear_cast_voxel()
           self.reset_framebuffer()
           print(f"Loaded from {to_slurp}")
        else:
            print(f"Failed to load from {to_slurp}")
