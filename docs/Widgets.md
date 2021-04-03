# Widgets Interface
For Jupyter Notebooks and Google Colab.

Launch via `gui='widgets'` in [`view()`](phi/vis/index.html#phi.vis.view) or [`show()`](phi/vis/index.html#phi.vis.show).

## Configuration

| Parameter       | Description |            Default |
|-----------------|-------------|-------------------:|
| `select`          | Creates sliders for these dimensions to select one field.  |          `'frames,'` |
| `style`           | Matplotlib plotting style. |               `None` |
| `update_interval` | While playing, redraws the current figure only after this many seconds have passed. | `1.2`, <img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>`2.5`  |
| `plt_args` | Keyword arguments passed to [`plot()`](https://tum-pbs.github.io/PhiFlow/phi/vis/#phi.vis.plot) | `{}` |
