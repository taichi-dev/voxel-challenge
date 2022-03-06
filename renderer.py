import taichi as ti
import numpy as np
import math
import time
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi

max_ray_depth = 4
use_directional_light = True

dist_limit = 100
# TODO: why doesn't it render normally when shutter_begin = -1?
shutter_begin = -0.5

exposure = 1.5
light_direction = [1.2, 0.3, 0.7]
light_direction_noise = 0.03
light_color = [1.0, 1.0, 1.0]


@ti.data_oriented
class Renderer:
    def __init__(self,
                 dx=1 / 1024,
                 sphere_radius=0.3 / 1024,
                 shutter_time=1e-3,
                 taichi_logo=True,
                 res=None,
                 max_num_particles_million=128):
        self.res = res
        self.aspect_ratio = res[0] / res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.taichi_logo = taichi_logo
        self.shutter_time = shutter_time  # usually half the frame time
        self.enable_motion_blur = self.shutter_time != 0.0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.voxel_grid_density = ti.field(dtype=ti.f32)
        self.voxel_has_particle = ti.field(dtype=ti.i32)
        self.fov = ti.field(dtype=ti.f32, shape=())

        self.particle_x = ti.Vector.field(3, dtype=ti.f32)
        if self.enable_motion_blur:
            self.particle_v = ti.Vector.field(3, dtype=ti.f32)
        self.particle_color = ti.Vector.field(3, dtype=ti.u8)
        self.pid = ti.field(ti.i32)
        self.num_particles = ti.field(ti.i32, shape=())


        self.voxel_edges = 0.1

        self.particle_grid_res = 2048

        self.dx = dx
        self.inv_dx = 1 / self.dx

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.floor_height = ti.field(dtype=ti.f32, shape=())

        self.supporter = 2
        self.sphere_radius = sphere_radius
        self.particle_grid_offset = [
            -self.particle_grid_res // 2 for _ in range(3)
        ]

        self.voxel_grid_res = self.particle_grid_res
        voxel_grid_offset = [-self.voxel_grid_res // 2 for _ in range(3)]
        self.max_num_particles_per_cell = 8192 * 1024
        self.max_num_particles = 1024 * 1024 * max_num_particles_million

        self.voxel_dx = self.dx
        self.voxel_inv_dx = 1 / self.voxel_dx

        assert self.sphere_radius * 2 < self.dx

        ti.root.dense(ti.ij, res).place(self.color_buffer)

        self.block_size = 8
        self.block_offset = [
            o // self.block_size for o in self.particle_grid_offset
        ]
        self.particle_bucket = ti.root.pointer(
            ti.ijk, self.particle_grid_res // self.block_size)

        self.particle_bucket.dense(ti.ijk, self.block_size).dynamic(
            ti.l, self.max_num_particles_per_cell,
            chunk_size=32).place(self.pid,
                                 offset=self.particle_grid_offset + [0])

        self.voxel_block_offset = [
            o // self.block_size for o in voxel_grid_offset
        ]
        ti.root.pointer(ti.ijk,
                        self.particle_grid_res // self.block_size).dense(
                            ti.ijk,
                            self.block_size).place(self.voxel_has_particle,
                                                   offset=voxel_grid_offset)
        voxel_block = ti.root.pointer(ti.ijk,
                                      self.voxel_grid_res // self.block_size)

        voxel_block.dense(ti.ijk,
                          self.block_size).place(self.voxel_grid_density,
                                                 offset=voxel_grid_offset)

        particle = ti.root.dense(ti.l, self.max_num_particles)

        particle.place(self.particle_x)
        if self.enable_motion_blur:
            particle.place(self.particle_v)
        particle.place(self.particle_color)

        self.set_up(0, 1, 0)
        self.set_fov(0.23)

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
            ret = self.voxel_grid_density[ipos]
        else:
            ret = 0.0
        return ret

    @ti.func
    def voxel_color(self, pos):
        p = pos * self.inv_dx

        p -= ti.floor(p)

        boundary = self.voxel_edges
        count = 0
        for i in ti.static(range(3)):
            if p[i] < boundary or p[i] > 1 - boundary:
                count += 1
        f = 0.0
        if count >= 2:
            f = 1.0
        return ti.Vector([0.9, 0.8, 1.0]) * (1.3 - 1.2 * f)

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
        while j < limit and self.sdf(p +
                                     dist * d) > 1e-8 and dist < dist_limit:
            dist += self.sdf(p + dist * d)
            j += 1
        if dist > dist_limit:
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
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
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
                    hit_pos = eye_pos + hit_distance * d
                    c = self.voxel_color(hit_pos)
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
        return hit_distance, normal, c

    @ti.func
    def inside_particle_grid(self, ipos):
        pos = ipos * self.dx
        return self.bbox[0][0] <= pos[0] and pos[0] < self.bbox[1][
            0] and self.bbox[0][1] <= pos[1] and pos[1] < self.bbox[1][
                1] and self.bbox[0][2] <= pos[2] and pos[2] < self.bbox[1][2]


    @ti.func
    def next_hit(self, pos, d, t):
        closest = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        closest, normal, c = self.dda_voxel(pos, d)

        if d[2] != 0:
            ray_closest = -(pos[2] + 5.5) / d[2]
            if ray_closest > 0 and ray_closest < closest:
                closest = ray_closest
                normal = ti.Vector([0.0, 0.0, 1.0])
                c = ti.Vector([0.6, 0.7, 0.7])

        ray_march_dist = self.ray_march(pos, d)
        if ray_march_dist < dist_limit and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.sdf_normal(pos + d * closest)
            c = self.sdf_color(pos + d * closest)

        return closest, normal, c

    @ti.kernel
    def set_camera_pos(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.camera_pos[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_up(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.up[None] = ti.Vector([x, y, z]).normalized()

    @ti.kernel
    def look_at(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.look_at[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_fov(self, fov: ti.f32):
        self.fov[None] = fov

    @ti.kernel
    def render(self):
        ti.block_dim(256)
        for u, v in self.color_buffer:
            fov = self.fov[None]
            pos = self.camera_pos[None]
            d = (self.look_at[None] - self.camera_pos[None]).normalized()
            fu = (2 * fov * (u + ti.random(ti.f32)) / self.res[1] -
                  fov * self.aspect_ratio - 1e-5)
            fv = 2 * fov * (v + ti.random(ti.f32)) / self.res[1] - fov - 1e-5
            du = d.cross(self.up[None]).normalized()
            dv = du.cross(d).normalized()
            d = (d + fu * du + fv * dv).normalized()
            t = (ti.random() + shutter_begin) * self.shutter_time

            contrib = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])

            depth = 0
            hit_sky = 1
            ray_depth = 0

            while depth < max_ray_depth:
                closest, normal, c = self.next_hit(pos, d, t)
                hit_pos = pos + closest * d
                depth += 1
                ray_depth = depth
                if normal.norm() != 0:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c

                    if ti.static(use_directional_light):
                        dir_noise = ti.Vector([
                            ti.random() - 0.5,
                            ti.random() - 0.5,
                            ti.random() - 0.5
                        ]) * light_direction_noise
                        direct = (ti.Vector(light_direction) +
                                  dir_noise).normalized()
                        dot = direct.dot(normal)
                        if dot > 0:
                            dist, _, _ = self.next_hit(pos, direct, t)
                            if dist > dist_limit:
                                contrib += throughput * ti.Vector(
                                    light_color) * dot
                else:  # hit sky
                    hit_sky = 1
                    depth = max_ray_depth

                max_c = throughput.max()
                if ti.random() > max_c:
                    depth = max_ray_depth
                    throughput = [0, 0, 0]
                else:
                    throughput /= max_c

            if hit_sky:
                if ray_depth != 1:
                    # contrib *= max(d[1], 0.05)
                    pass
                else:
                    # directly hit sky
                    pass
            else:
                throughput *= 0

            # contrib += throughput
            self.color_buffer[u, v] += contrib

    @ti.kernel
    def copy(self, img: ti.ext_arr(), samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.res[0]
            v = 1.0 * j / self.res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0])**2 +
                (v - self.vignette_center[1])**2) - self.vignette_radius), 0)

            for c in ti.static(range(3)):
                img[i, j, c] = ti.sqrt(self.color_buffer[i, j][c] * darken *
                                       exposure / samples)

    @ti.kernel
    def total_non_empty_voxels(self) -> ti.i32:
        counter = 0

        for I in ti.grouped(self.voxel_has_particle):
            if self.voxel_has_particle[I]:
                counter += 1

        return counter

    def reset(self):
        self.particle_bucket.deactivate_all()
        self.voxel_grid_density.snode.parent(n=2).deactivate_all()
        self.voxel_has_particle.snode.parent(n=2).deactivate_all()
        self.color_buffer.fill(0)

    def initialize_particles_from_taichi_elements(self):
        self.reset()

        np_x = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        np_color = np.array([[0, 255, 0]], dtype=np.uint8)
        num_part = len(np_x)

        assert num_part <= self.max_num_particles

        for i in range(3):
            self.bbox[0][i] = -1
            self.bbox[1][i] = 1
            print(f'Bounding box dim {i}: {self.bbox[0][i]} {self.bbox[1][i]}')

        for i in range(10):
            self.voxel_has_particle[(2, i, 2)] = 1
            self.voxel_grid_density[(2, i, 2)] = 1


    def render_frame(self, spp):
        last_t = 0
        for i in range(1, 1 + spp):
            self.render()

            interval = 20
            if i % interval == 0:
                if last_t != 0:
                    ti.sync()
                    print("time per spp = {:.2f} ms".format(
                        (time.time() - last_t) * 1000 / interval))
                last_t = time.time()

        img = np.zeros((self.res[0], self.res[1], 3), dtype=np.float32)
        self.copy(img, spp)
        return img
