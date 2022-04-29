# Taichi Voxel Challenge

You are required to create your voxel artwork by putting your code in `main.py`.

+ You can import only two modules: `taichi` (pip installation guide below) and `scene.py` (in the repo).
+ Your code cannot exceed 99 lines.

The availabe APIs are:

+ `scene = Scene(voxel_edges=, exposure=)`
+ `scene.set_voxel(voxel_id, material, color)`
+ `materal, color = scene.get_voxel(voxel_id)`
+ `scene.set_floor(height, color)`
+ `scene.set_directional_light(dir, noise, color)`
+ `scene.set_background_color(color)`

And also call `scene.finish()` in the last.

Modifying other files except `main.py` is not allowed.


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

Please fill in your code in `main.py` and include your result in this README.md file.

## Quickstart

```sh
python3 example1.py  # example2/3/4.py
```

Mouse and keyboard interface:

+ Drag with your left mouse button to rotate camera.
+ Press `W/A/S/D/Q/E` to move camera.
+ Press `P` to save screenshot.



## Some examples made by Taichi developers

<img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/city.jpg" width="45%"></img> <img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/city2.jpg" width="45%"></img>
<img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/tree.jpg" width="45%"></img> <img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/desktop.jpg" width="45%"></img>
<img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/earring_girl.jpg" width="45%"></img> <img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/pika.jpg" width="45%"></img>
<img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/yinyang.jpg" width="45%"></img> <img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/lang.jpg" width="45%"></img>

## Show your artwork 

Please put your artwork here (replace `demo.jpg`)

![](./deno.jpg)
