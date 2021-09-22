import numpy

from phi.field import Grid, CenteredGrid
from ._console_util import underline, get_arrow
from ...math import extrapolation

FILLED = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']


def heatmap(grid: Grid, cols: int, rows: int, title: str):
    inner_cols = cols - 10
    inner_rows = rows - 2
    grid >>= CenteredGrid(0, extrapolation.ZERO, x=inner_cols, y=inner_rows, bounds=grid.bounds)
    data = grid.values.numpy('y,x')
    min_, max_ = numpy.min(data), numpy.max(data)
    col_indices = (data - min_) / (max_ - min_) * len(FILLED)
    col_indices = numpy.clip(numpy.round(col_indices).astype(numpy.int8), 0, len(FILLED) - 1)
    lines = []
    # lines.append("   " + "_" * inner_cols + " ")
    title = title[:inner_cols]
    padded_title = " " * ((inner_cols - len(title)) // 2) + title + " " * ((inner_cols - len(title) + 1) // 2)
    lines.append("   " + underline(padded_title) + "\033[0m ")
    for index_row in col_indices[::-1]:
        line = [FILLED[col_index] for col_index in index_row]
        lines.append("  |" + "".join(line)+"|")
    lines[-1] = lines[-1][:3] + underline(lines[-1][3:inner_cols+3]) + lines[-1][inner_cols+3:]
    return lines


def quiver(grid: Grid, cols: int, rows: int, title: str, threshold: float, basic_chars=True):
    inner_cols = cols - 10
    inner_rows = rows - 2
    grid >>= CenteredGrid(0, extrapolation.ZERO, x=inner_cols, y=inner_rows, bounds=grid.bounds)
    data = grid.values.numpy('y,x,vector')[::-1]
    thick_threshold = numpy.max(numpy.sum(data ** 2, -1)) / 4  # half the vector length

    lines = []
    title = title[:inner_cols]
    padded_title = " " * ((inner_cols - len(title)) // 2) + title + " " * ((inner_cols - len(title) + 1) // 2)
    lines.append("   " + underline(padded_title) + "\033[0m ")
    for y in range(data.shape[0]):
        line = ""
        for x in range(data.shape[1]):
            u, v = data[y, x]
            len_squared = u ** 2 + v ** 2
            if len_squared < threshold ** 2:
                arrow = " "
            else:
                arrow = get_arrow(u, v, thick=len_squared >= thick_threshold, basic_char=basic_chars)
            line += arrow
        lines.append("  |" + "".join(line)+"|")
    lines[-1] = lines[-1][:3] + underline(lines[-1][3:inner_cols+3]) + lines[-1][inner_cols+3:]
    return lines
