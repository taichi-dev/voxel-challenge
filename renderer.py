import taichi as ti

from math_utils import (eps, inf, out_dir, ray_aabb_intersection)

MAX_RAY_DEPTH = 4
use_directional_light = True

DIS_LIMIT = 100


@ti.data_oriented
class Renderer:
    def __init__(self, dx, image_res, up, voxel_edges, exposure=3):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.voxel_color = ti.Vector.field(3, dtype=ti.u8)
        self.voxel_material = ti.field(dtype=ti.i8)

        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_direction_noise = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.cast_voxel_hit = ti.field(ti.i32, shape=())
        self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

        self.voxel_edges = voxel_edges
        self.exposure = exposure

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.floor_height = ti.field(dtype=ti.f32, shape=())
        self.floor_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.voxel_dx = dx
        self.voxel_inv_dx = 1 / dx
        # Note that voxel_inv_dx == voxel_grid_res iff the box has width = 1
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

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)

    def set_directional_light(self, direction, light_direction_noise,
                              light_color):
        direction_norm = (direction[0]**2 + direction[1]**2 +
                          direction[2]**2)**0.5
        self.light_direction[None] = (direction[0] / direction_norm,
                                      direction[1] / direction_norm,
                                      direction[2] / direction_norm)
        self.light_direction_noise[None] = light_direction_noise
        self.light_color[None] = light_color

    @ti.func
    def inside_grid(self, ipos):
        return ipos.min() >= -self.voxel_grid_res // 2 and ipos.max(
        ) < self.voxel_grid_res // 2

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
    def ray_march(self, p, d):
        dist = inf
        if d[1] < -eps:
            dist = (self.floor_height[None] - p[1]) / d[1]
        return dist

    @ti.func
    def sdf_normal(self, p):
        return ti.Vector([0.0, 1.0, 0.0])  # up

    @ti.func
    def sdf_color(self, p):
        return self.floor_color[None]

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
        ti.loop_config(block_dim=256)
        for u, v in self.color_buffer:
            d = self.get_cast_dir(u, v)
            pos = self.camera_pos[None]
            t = 0.0

            contrib = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            c = ti.Vector([1.0, 1.0, 1.0])

            depth = 0
            hit_light = 0
            hit_background = 0

            # Tracing begin
            for bounce in range(MAX_RAY_DEPTH):
                depth += 1
                closest, normal, c, hit_light = self.next_hit(pos, d, t)
                hit_pos = pos + closest * d
                if not hit_light and normal.norm() != 0 and closest < 1e8:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c

                    if ti.static(use_directional_light):
                        dir_noise = ti.Vector([
                            ti.random() - 0.5,
                            ti.random() - 0.5,
                            ti.random() - 0.5
                        ]) * self.light_direction_noise[None]
                        light_dir = (self.light_direction[None] +
                                     dir_noise).normalized()
                        dot = light_dir.dot(normal)
                        if dot > 0:
                            hit_light_ = 0
                            dist, _, _, hit_light_ = self.next_hit(
                                pos, light_dir, t)
                            if dist > DIS_LIMIT:
                                # far enough to hit directional light
                                contrib += throughput * \
                                    self.light_color[None] * dot
                else:  # hit background or light voxel, terminate tracing
                    hit_background = 1
                    break

                # Russian roulette
                max_c = throughput.max()
                if ti.random() > max_c:
                    throughput = [0, 0, 0]
                    break
                else:
                    throughput /= max_c
            # Tracing end

            if hit_light:
                contrib += throughput * c
            else:
                if depth == 1 and hit_background:
                    # Direct hit to background
                    contrib = self.background_color[None]
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
                    self.color_buffer[i, j][c] * darken * self.exposure /
                    samples)

    @ti.kernel
    def recompute_bbox(self):
        for d in ti.static(range(3)):
            self.bbox[0][d] = 1e9
            self.bbox[1][d] = -1e9
        for I in ti.grouped(self.voxel_material):
            if self.voxel_material[I] != 0:
                for d in ti.static(range(3)):
                    ti.atomic_min(self.bbox[0][d], (I[d] - 1) * self.voxel_dx)
                    ti.atomic_max(self.bbox[1][d], (I[d] + 2) * self.voxel_dx)

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    def accumulate(self):
        self.render()
        self.current_spp += 1

    def fetch_image(self):
        self._render_to_image(self.current_spp)
        return self._rendered_image

    @staticmethod
    @ti.func
    def to_vec3u(c):
        c = ti.math.clamp(c, 0.0, 1.0)
        r = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i] * 255, ti.u8)
        return r

    @staticmethod
    @ti.func
    def to_vec3(c):
        r = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i], ti.f32) / 255.0
        return r

    @ti.func
    def set_voxel(self, idx, mat, color):
        self.voxel_material[idx] = ti.cast(mat, ti.i8)
        self.voxel_color[idx] = self.to_vec3u(color)

    @ti.func
    def get_voxel(self, ijk):
        mat = self.voxel_material[ijk]
        color = self.voxel_color[ijk]
        return mat, self.to_vec3(color)
