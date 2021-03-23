import numpy

from phi.field import Grid

from ...physics import Domain

FILLED = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']


def heatmap(grid: Grid, cols: int, rows: int):
    inner_cols = cols - 10
    inner_rows = rows - 2
    grid >>= Domain(x=inner_cols, y=inner_rows, bounds=grid.bounds).scalar_grid()
    data = grid.values.numpy('y,x')
    min_, max_ = numpy.min(data), numpy.max(data)
    col_indices = (data - min_) / (max_ - min_) * len(FILLED)
    col_indices = numpy.clip(numpy.round(col_indices).astype(numpy.int8), 0, len(FILLED) - 1)
    lines = []
    lines.append("   " + "_" * inner_cols + " ")
    for index_row in col_indices[::-1]:
        line = [FILLED[col_index] for col_index in index_row]
        lines.append("  |" + "".join(line)+"|")
    lines.append("   " + "‾" * inner_cols + " ")
    return lines
