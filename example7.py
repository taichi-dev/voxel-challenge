from scene import Scene; import taichi as ti; from taichi.math import *
day = True; manual_seed = 77
scene = Scene(voxel_edges=0, exposure=2 - day)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))
scene.set_background_color((0.9, 0.98, 1) if day else (0.01, 0.01, 0.02))
scene.set_directional_light((1, 1, 1), 0.1, (0.9, 0.98, 1) if day else (0.01, 0.01, 0.02))
lgrid, ngrid = 15, 8

@ti.func
def rand(i, j): return fract(ti.sin(dot(vec2(i, j), vec2(12.9898, 78.233))) * 43758.5453)
@ti.func
def is_road(i, j):
    return 0 <= i < ngrid and 0 <= j <= ngrid and scene.get_voxel(vec3(i, -8, j))[0] == 1

@ti.kernel
def initialize():
    for i, j in ti.ndrange(8, 8): scene.set_voxel(vec3(i, -8, j), 0, vec3(0))
    start, end = 1+int(vec2(ti.random(),ti.random())*(ngrid-2)), 1+int(vec2(ti.random(),ti.random())*(ngrid-2))
    turn = start + 1
    while any((abs(turn-start)==1)|(abs(turn-end)==1)): turn = 1+int(vec2(ti.random(),ti.random())*(ngrid-2))
    for k in ti.static([0, 1]):
        d = vec2(k, 1-k); p = start[k]*vec2(1-k, k)-d
        while p[1-k] < ngrid - 1:
            p += d; scene.set_voxel(vec3(p.x, -8, p.y), 1, vec3(0.5))
            if p[1-k] == turn[1-k]: d = (1 if start[k] < end[k] else -1) * vec2(1-k, k)
            if p[k] == end[k]: d = vec2(k, 1-k)
@ti.func
def build_road(X, uv, d):
    if d.sum() <= 2:
        if ((d.x | d.z) ^ (d.y | d.w)) & 1: uv = vec2(uv.y, uv.x) if (d.y | d.w) & 1 else uv
        else: # curve
            while d.z == 0 or d.w == 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)
            uv = vec2(uv.norm(), ti.atan2(uv.x, uv.y)*2/pi*lgrid)
    elif d.sum() >= 3: # junction
        while d.sum() == 3 and d.y != 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)  # rotate T-junction
        if d.sum() > 3 or uv.x <= 7:
            uv = vec2(mix(14-uv.x, uv.x, uv.x <= 7), mix(14-uv.y, uv.y, uv.y <= 7))
            uv = vec2(uv.norm(), ti.atan2(uv.x, uv.y)*2/pi*lgrid)
    scene.set_voxel(vec3(X.x, 0, X.y), 1, vec3(1 if uv.x==7 and 4<uv.y<12 else 0.5)) # pavement
    if uv.x <= 1 or uv.x >= 13: scene.set_voxel(vec3(X.x, 1, X.y), 1, vec3(0.7, 0.65, 0.6)) # sidewalk
    if uv.y == 7 and (uv.x == 1 or uv.x == 13): # lights
        for i in range(2, 9): scene.set_voxel(vec3(X.x, i, X.y), 1, vec3(0.6, 0.6, 0.6))
    if uv.y == 7 and (1<=uv.x<=2 or 12<=uv.x<=13): scene.set_voxel(vec3(X.x, 8, X.y), 1, vec3(0.6, 0.6, 0.6))
    if uv.y == 7 and (uv.x == 2 or uv.x == 12): scene.set_voxel(vec3(X.x, 7, X.y), 2, vec3(1, 1, 0.6))
@ti.func
def build_building(X, uv, d, r):
    while d.sum() > 0 and d.z == 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)  # rotate
    fl = int(3 + 10 * r); style = rand(r, 5)
    wall = vec3(rand(r, 1),rand(r, 2),rand(r, 2)) * 0.2+0.4
    wall2 = mix(vec3(rand(r, 9)*0.2+0.2), wall, style > 0.5 and rand(r, 4) < 0.4)
    maxdist = max(abs(uv.x - 7), abs(uv.y - 7))
    for i in range(2, fl * 4):
        light = mix(vec3(0.25,0.35,0.38), vec3(0.7,0.7,0.6), rand(rand(X.x, X.y), i//2)>0.6)
        if maxdist < 6:
            scene.set_voxel(vec3(X.x, i, X.y), mix(1, 0, i%4<2), mix(wall2, light, i%4<2))
            if (uv.x == 2 or uv.x == 12) and (uv.y == 2 or uv.y == 12) or style>0.5 and (uv.x%3==1 or uv.y%3==1):
                scene.set_voxel(vec3(X.x, i, X.y), 1, wall)
        if maxdist < 5:  scene.set_voxel(vec3(X.x, i, X.y), mix(1, 2, i%4<2), mix(wall, light, i%4<2))
    if maxdist == 5: 
        for i in range(fl*4, fl*4+2): scene.set_voxel(vec3(X.x, i, X.y), 1, wall) # roof
    if maxdist < 5: scene.set_voxel(vec3(X.x, fl*4, X.y), 1, vec3(rand(r, 7)*0.2+0.4))
    for i in range(2): scene.set_voxel(vec3(X.x, i, X.y), 1, vec3(0.7, 0.65, 0.6)) # sidewalk
    if fl > 10 and uv.x == 6 and uv.y == 6: # antenna
        for i in range(fl+1):
            scene.set_voxel(vec3(X.x, fl*5-i, X.y), mix(1, 2, i==0), mix(vec3(0.6), vec3(0.8,0,0), i==0))
    if d.sum() > 0 and uv.y == 2 and 4 < uv.x < 10: # billboard
        for i in range(5, 7):
            scene.set_voxel(vec3(X.x,i,X.y), 2, vec3(int(r*3)==0,int(r*3)==1,int(r*3)==2)*(0.2+ti.random()*0.3))
        for i in range(2, 5): scene.set_voxel(vec3(X.x,i,X.y), 0, vec3(0)) 
    if d.sum() > 0 and uv.y == 3 and 4 < uv.x < 10: 
        for i in range(2, 5): scene.set_voxel(vec3(X.x,i,X.y), 1, vec3(0.7,0.7,0.6))
    if max(abs(uv.x - rand(r, 8)*7-4), abs(uv.y - rand(r, 10)*7-4)) < 1.5: # HVAC
        for i in range(fl*4+1, fl*4+3): scene.set_voxel(vec3(X.x, i, X.y), 1, vec3(0.6))
@ti.func
def build_park(X, uv, d, r):
    center, height = int(vec2(rand(r, 1) * 7 + 4, rand(r, 2) * 7 + 4)), 9 + rand(r, 3) * 5
    for i in range(height + 3): # tree
        if (uv - center).norm() < 1:
            scene.set_voxel(vec3(X.x, i, X.y), 1, vec3(0.36, 0.18, 0.06))
        if i > min(height-4, (height+5)//2) and (uv - center).norm() < (height+3-i) * (rand(r, 4)*0.6 + 0.4):
            scene.set_voxel(vec3(X.x, i, X.y), ti.random()<0.8, vec3(0.1, 0.3 + ti.random()*0.2, 0.1))
    h = 2 * ti.sin((uv.x**2+uv.y**2+rand(r, 0)**2*256)/1024 * 2*pi) + 2 + (ti.random() > 0.95)
    for i in range(h): # grass
        scene.set_voxel(vec3(X.x, i, X.y), 1, vec3(0.2, 0.5 + ti.random() * 0.2, 0.05))
    if max(abs(uv.x - rand(r, 4)*7-4), abs(uv.y - rand(r, 5)*7-4)) < 0.5: # light
        for i in range(3):
            scene.set_voxel(vec3(X.x, h+i, X.y), 1+(i==1), mix(vec3(0.2),vec3(0.9,0.8,0.6),vec3(i==1)))

@ti.kernel
def draw():
    for X in ti.grouped(ti.ndrange((-60, 60), (-60, 60))):
        I, uv = (X+60) // lgrid, float((X + 60) % lgrid)
        d = int(vec4(is_road(I.x,I.y+1),is_road(I.x+1,I.y),is_road(I.x,I.y-1),is_road(I.x-1,I.y)))
        r = mix(rand(I.x, I.y), any(d>0), 0.4)
        if is_road(I.x, I.y): build_road(X, uv, d)
        elif r > 0.5: build_building(X, uv, d, 2*r-1)
        else: build_park(X, uv, d, 2*r)

[initialize() for _ in range(manual_seed + 1)]; draw(); scene.finish()