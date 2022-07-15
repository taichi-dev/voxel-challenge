from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=1)
scene.set_directional_light((1, .3, .3), .8, (1, 1, 1))
scene.set_background_color((0, 0, 0))
scene.set_floor(-64, (0.01, 0.01, 0.012))


@ti.func
def rgb(r, g, b):
    return vec3(r/255.0, g/255.0, b/255.0)


@ti.func
def gray(g):
    return rgb(g, g, g)


@ti.func
def get_emmit_color(r):
    return mix(rgb(242, 239, 193), rgb(236, 195, 107), r)


@ti.func
def make_tiny_cloud(pos, s, r1, r2, density, grayVal):
    u = [int(r2 * x) for x in s]
    for i, j, k in ti.ndrange((-u[0], u[0]), (-u[1], u[1]), (-u[2], u[2])):
        x = vec3(i/s[0], j/s[1], k/s[2])
        if x.dot(x) < r1 + (r2-r1) * (ti.random()) and ti.random() < density:
            scene.set_voxel(
                vec3(pos[0]+i, pos[1]+j, pos[2]+k), 1, gray(grayVal))


@ti.func
def make_cloud_city(base, n):
    center = ti.Vector([0, 0])
    for i, j in ti.ndrange((-n, n), (-n, n)):
        i3 = ti.Vector([i, j])
        dis = ti.pow(ti.max(0, 1 - ti.math.distance(i3, center)/n) * 1.1, 3)
        height = (ti.random() * n * dis * 1)
        for k in ti.ndrange((int(-height*.6+base), int(height*1.2+base))):
            if k > base and dis*.1 > ti.random():
                scene.set_voxel(vec3(i, k, j), 2, get_emmit_color(ti.random()))
            else:
                scene.set_voxel(vec3(i, k, j), 1, gray(
                    (1-.8*ti.pow(dis, .6)) * 255))


@ti.kernel
def initialize_voxels():
    n = 60
    base = -24
    make_cloud_city(base, n)

    make_tiny_cloud((30, -30, -20), (2, 1, 2), 20, 40, .3, 120)
    make_tiny_cloud((20, -28, 24), (2, 1, 2), 10, 30, .4, 80)
    make_tiny_cloud((-30, -32, 28), (2, 1, 2), 10, 30, .35, 80)
    make_tiny_cloud((-40, -50, -34), (3, 2, 3), 10, 30, .2, 120)
    make_tiny_cloud((36, -46, -36), (2, 1, 2.4), 20, 50, .3, 90)


initialize_voxels()
scene.finish()
