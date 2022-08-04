from scene import Scene; import taichi as ti; from taichi.math import *
scene = Scene(0, 2.75)
scene.set_background_color((10.0, 10.0, 10.0))
scene.set_floor(-1e5, (0, 0, 0))
scene.set_directional_light((0, 0.55, -1), 0.03, (0.9, 0.9, 0.9))
@ti.func
def rd(): return ti.random()
@ti.func
def set(idx, mat, color = vec3(0), noise = vec3(0)): scene.set_voxel(idx, mat, color + rd() * noise)
@ti.func
def fill(p0, s, mat, color = vec3(0), noise = vec3(0), paint = False):
    for p in ti.grouped(ti.ndrange((p0.x, p0.x + s.x), (p0.y, p0.y + s.y), (p0.z, p0.z + s.z))):
        if not paint or scene.get_voxel(p)[0] != 0: set(p, mat, color, noise)
@ti.func
def brighten(p0, s, scale):
    for p in ti.grouped(ti.ndrange((p0.x, p0.x + s.x), (p0.y, p0.y + s.y), (p0.z, p0.z + s.z))):
        mat, color = scene.get_voxel(p); set(p, mat, scale * color)
@ti.func
def c1(): v=rd(); return vec3(1) if v<.7 else vec3(.5,1,1) if v<.8 else vec3(1,.5,1) if v<.9 else vec3(1,1,.5)
@ti.func
def c2(): a = rd(); return vec3(1, rd(), 0) if a < 0.4 else vec3(rd(), 1, 0) if a < 0.7 else vec3(0, rd(), 1)
@ti.func
def stuff(p0, s, r):
    for x in range(s.x): fill(p0 + ivec3(x, 0, 0), ivec3(1, ti.round(s.y - r * rd()), s.z - ti.round(rd())), 1, c2())
@ti.kernel
def initialize():
    wood = vec3(0.6, 0.5, 0.3)
    fill(ivec3(-64, -20, -60), ivec3(128, 74, 120), 1, vec3(0.6)) # Wall
    fill(ivec3(-64, -19, -60), ivec3(128, 1, 120), 1, vec3(0.2, 0.1, 0.0))
    fill(ivec3(-63, -19, -59), ivec3(126, 72, 119), 0)
    fill(ivec3(0, 52, -60), ivec3(64, 1, 120), 2, vec3(1.0, 0.85, 0.7))
    for x, y in ti.ndrange((-64, 64), (-18, 54)): set(ivec3(x, y, -60), 1, vec3(0.5, 0.55, 0.6)
        if x % 9 == 1 or x % 9 == 7 or (abs(x % 9 - 4) + abs(y % 7 - 3)) == 1 else vec3(0.6))
    for x, z in ti.ndrange((-64, 64), (-60, 60)): # Floor
        set(ivec3(x, -20, z), 1, vec3(1.0, 0.7, 0.35) * (0.7 if x % 4 == 0 else 1), vec3(0.1))
    fill(ivec3(-32, -3, -64), ivec3(64, 40, 6), 1, vec3(1)) # Window
    fill(ivec3(-31, -2, -63), ivec3(62, 38, 5), 0)
    fill(ivec3(1, -1, -64), ivec3(21, 27, 1), 0)
    fill(ivec3(1, 28, -64), ivec3(21, 7, 1), 0)
    fill(ivec3(24, -1, -64), ivec3(6, 36, 1), 0)
    for x, y in ti.ndrange((-32, 0), (-4, 37)): c = c1() if (x % 6 == 2 and y % 5 == 4) or (x % 6 == 3 and y % 5 == 3
        ) else vec3(0.9, 0.6, 0.7); set(ivec3(x, y, -56 + ti.round(ti.sin(x / 3 * pi))), 1, 0.65 * c, vec3(0.03))
    brighten(ivec3(-22, -1, -58), ivec3(21, 27, 5), 1.8)
    brighten(ivec3(-22, 28, -58), ivec3(21, 7, 5), 1.8)
    brighten(ivec3(-31, -1, -58), ivec3(6, 36, 5), 1.8)
    # Carpet
    for x,z in ti.ndrange((-30,0),(-22,38)):set(ivec3(x,-19,z),1,vec3(1)if(24<max(abs(z-8),-x)<27)else vec3(.9,.6,.7))
    for a in range(1024): v = a / 1024 * pi; x, z = (ti.round((10 * abs(ti.sin(12 * v)) + 10) * ti.cos(10 * v)),
        8 + (10 * abs(ti.sin(12 * v)) + 10) * ti.sin(10 * v)); set(ivec3(x, -19, z), 1 if x < 0 else 0, vec3(1))
    fill(ivec3(-8, -19, -50), ivec3(8, 15, 8), 1, vec3(1)) # Box 1
    fill(ivec3(-7, -19, -50), ivec3(6, 1, 8), 0)
    fill(ivec3(-7, -9, -49), ivec3(6, 4, 7), 0)
    stuff(ivec3(-6, -9, -49), ivec3(4, 3, 5), 2)
    fill(ivec3(-7, -14, -49), ivec3(6, 4, 7), 0)
    stuff(ivec3(-6, -14, -49), ivec3(4, 3, 5), 2)
    fill(ivec3(-7, -17, -43), ivec3(6, 2, 2), 0)
    fill(ivec3(-33, -8, -50), ivec3(24, 1, 14), 1, wood, vec3(0.1)) # Desk
    fill(ivec3(-32, -19, -49), ivec3(22, 12, 12), 1, wood, vec3(0.1))
    fill(ivec3(-31, -19, -49), ivec3(20, 9, 12), 0)
    stuff(ivec3(-30, -7, -48), ivec3(7, 6, 6), 4)
    fill(ivec3(-27, -19, -30), ivec3(8, 14, 1), 1, wood, vec3(0.1)) # Chair
    fill(ivec3(-27, -19, -37), ivec3(8, 6, 8), 1, wood, vec3(0.1))
    fill(ivec3(-27, -19, -36), ivec3(8, 5, 6), 0)
    fill(ivec3(-26, -19, -37), ivec3(6, 5, 8), 0)
    fill(ivec3(-27, -13, -37), ivec3(8, 1, 7), 1, vec3(0.5, 0.2, 0.3), vec3(0.1))
    fill(ivec3(-15, -7, -45), ivec3(3, 1, 3), 1, vec3(0.2, 0.1, 0.1), vec3(0.1)) # Lamp
    fill(ivec3(-14, -7, -44), ivec3(1, 6, 1), 1, vec3(0.2, 0.1, 0.1), vec3(0.1))
    for p in ti.grouped(ti.ndrange((-4, 5), 5, (-4, 5))):
        if p.norm() < 4: set(ivec3(-14, -2, -44) + p, 1, 1.5 * vec3(0.9, 0.6, 0.7), vec3(0.1))
    for x in range(-62, -35): h = 11 + ti.round(1.7 * ti.cos((x + 49) * 0.3)); fill(ivec3(x, -19, -57), # Bed
        ivec3(1, h + 5, 1), 1, wood, vec3(0.1)); fill(ivec3(x, -19, 20), ivec3(1, h, 1), 1, wood, vec3(0.1))
    fill(ivec3(-62, -15, -56), ivec3(26, 1, 76), 1, wood, vec3(0.1))
    fill(ivec3(-61, -14, -56), ivec3(24, 3, 76), 1, vec3(1), vec3(0.1))
    fill(ivec3(-56, -11, -54), ivec3(14, 2, 9), 1, vec3(1), vec3(0.1))
    fill(ivec3(-55, -9, -54), ivec3(12, 1, 9), 1, vec3(1), vec3(0.1))
    fill(ivec3(-62, -14, -36), ivec3(26, 3, 52), 1, vec3(0.9, 0.6, 0.7), vec3(0.1))
    fill(ivec3(-61, -11, -36), ivec3(24, 1, 52), 1, vec3(0.9, 0.6, 0.7), vec3(0.1))
    for a in range(1024): v=a/1024*2*pi;x,z=9*ti.cos(3*v)-49,15*ti.sin(5*v)-10;set(ivec3(x,-11,z),1,vec3(1),vec3(0.1))
    fill(ivec3(-6, -4, -48), ivec3(4), 1, vec3(0.5, 0.4, 0.3)) # Plant
    fill(ivec3(-5, 0, -47), ivec3(2, 3, 2), 1, vec3(0.3, 0.6, 0.5))
    for p in ti.grouped(ti.ndrange(6, 4, 6)): set(ivec3(-7, 3, -49) + p, 1 if rd() < 0.2 else 0, vec3(0.3, 0.6, 0.5))
    fill(ivec3(-43, 3, -59), ivec3(11, 9, 1), 1, vec3(0.2, 0.1, 0.1)) # White Board
    fill(ivec3(-42, 4, -59), ivec3(9, 7, 1), 1, vec3(0.5), vec3(0.4))
    fill(ivec3(-61, 3, -59), ivec3(17, 8, 9), 1, vec3(1)) # Box 2
    fill(ivec3(-60, 4, -58), ivec3(7, 6, 8), 0)
    stuff(ivec3(-59, 4, -58), ivec3(5, 5, 6), 3)
    fill(ivec3(-52, 4, -58), ivec3(7, 6, 8), 0)
    stuff(ivec3(-51, 4, -58), ivec3(5, 5, 6), 3)
    fill(ivec3(-53, 17, -59), ivec3(10, 1, 9), 1, vec3(1)) # Shelf
    stuff(ivec3(-51, 18, -59), ivec3(6, 5, 7), 3)
    fill(ivec3(-60, 26, -59), ivec3(25, 16, 11), 1, vec3(0.8), vec3(0.2)) # Box 3
    fill(ivec3(-60, 27, -59), ivec3(25, 1, 11), 1, vec3(0.6), vec3(0.1))
    fill(ivec3(-48, 28, -59), ivec3(1, 14, 11), 1, vec3(0.6), vec3(0.1))
# scene.camera._camera_pos[:] = [0.0, 0.0, 1.9]; scene.camera._lookat_pos[:] = [0.0, 0.0, 0.0]
# scene.renderer.set_camera_pos(*scene.camera.position); scene.renderer.set_look_at(*scene.camera.look_at)
initialize(); scene.finish()
