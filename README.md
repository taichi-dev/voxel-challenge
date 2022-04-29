# <a name="title">Taichi Voxel Challenge</a>

<p align="center">
<img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/tree.jpg" width="60%"></img>
</p>

> We recommend you replacing the above image by your artwork so other people can quickly see your results.

You are invited to create your voxel artwork by putting your code in `main.py`.

+ You can import only two modules: `taichi` (pip installation guide below) and `scene.py` (in the repo).
+ The code in `main.py` cannot exceed 99 lines, and each line cannot exceed 120 characters.

The availabe APIs are:

+ `scene = Scene(voxel_edges, exposure)`
+ `scene.set_voxel(voxel_id, material, color)`
+ `material, color = scene.get_voxel(voxel_id)`
+ `scene.set_floor(height, color)`
+ `scene.set_directional_light(dir, noise, color)`
+ `scene.set_background_color(color)`

And also call `scene.finish()` at the last.

**Modifying other files except `main.py` is not allowed.**


## Installation

Make sure your `pip` is the latest one:

```bash
pip3 install pip --upgrade
```

Assume you have a Python 3 environment properly, you can simply run:

```bash
pip3 install -r requirements.txt
```

to install the dependendies of the voxel renderer.

## Quickstart

```sh
python3 example1.py  # example2/3/4.py
```

Mouse and keyboard interface:

+ Drag with your left mouse button to rotate camera.
+ Press `W/A/S/D/Q/E` to move camera.
+ Press `P` to save screenshot.

## More examples

<a href="https://github.com/raybobo/taichi-voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/city.jpg" width="45%"></img></a>  <a href="https://github.com/victoriacity/voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/city2.jpg" width="45%"></img></a> 
<a href="https://github.com/yuanming-hu/voxel-art"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/tree2.jpg" width="45%"></img></a> <a href="https://github.com/neozhaoliang/voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/desktop.jpg" width="45%"></img></a> 
<a href="https://github.com/maajor/maajor-voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/earring_girl.jpg" width="45%"></img></a>  <a href="https://github.com/rexwangcc/taichi-voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/pika.jpg" width="45%"></img></a> 
<a href="https://github.com/houkensjtu/qbao_voxel_art"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/yinyang.jpg" width="45%"></img></a>  <a href="https://github.com/ltt1598/voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/lang.jpg" width="45%"></img></a> 

## Show your artwork 

Please put your artwork at the beginning of this readme file. For example, assuming your screenshot is saved as `demo.jpg`, you can put `![](./demo.jpg)` directly under the [title](#title).
