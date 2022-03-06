import taichi as ti

ti.init(arch=ti.vulkan)

selected_voxel_color = (0.2, 0.2, 0.2)
WINDOW_RES = (800, 800)
window = ti.ui.Window("Voxel Editor", WINDOW_RES, vsync=True)
canvas = window.get_canvas()

def show_hud():
  global selected_voxel_color
  window.GUI.begin("Options", 0.05, 0.05, 0.3, 0.2)
  selected_voxel_color = window.GUI.color_edit_3(
              "Voxel color", selected_voxel_color)
  window.GUI.end()

while window.running:
  show_hud()
  window.show()
