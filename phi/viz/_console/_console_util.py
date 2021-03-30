import shutil

# Unicode special characters
FULL_BLOCK = "\u2588"
# ANSI escape
MOVE_TO_PREV_LINE_START = "\033[F"
CLEAR_LINE = "\033[K"

BASIC_LINES = "--||\\/\\/"
ARROWS = "🡠🡢🡡🡣🡤🡥🡦🡧"
HEAVY_ARROWS = "🡸🡺🡹🡻🡼🡽🡾🡿"


def get_arrow(x, y, thick=False, basic_char=False):
    charset = BASIC_LINES if basic_char else (HEAVY_ARROWS if thick else ARROWS)
    if x == 0:
        return charset[2] if y > 0 else charset[3]  # 🡡 🡣
    frac = y / x
    if x > 0:
        if frac > 2.41421:  # tan(3/2 45°)
            return charset[2]  # 🡡
        elif frac > 0.41421:  # tan(1/2 45°)
            return charset[5]  # 🡥
        elif frac > -0.41421:
            return charset[1]  # 🡢
        elif frac > -2.41421:
            return charset[6]  # 🡦
        else:
            return charset[3]  # 🡣
    else:
        if -frac > 2.41421:  # tan(3/2 45°)
            return charset[2]  # 🡡
        elif -frac > 0.41421:  # tan(1/2 45°)
            return charset[4]  # 🡤
        elif -frac > -0.41421:
            return charset[0]  # 🡠
        elif -frac > -2.41421:
            return charset[7]  # 🡧
        else:
            return charset[3]  # 🡣




def underline(text):
    return f"\033[4m{text}\033[0m"


def cursor_up(n: int):
    return f"\x1b[{n}A"


def cursor_down(n: int):
    return f"\x1b[{n}B"


def cursor_left(n: int):
    return f"\x1b[{n}D"


def cursor_right(n: int):
    return f"\x1b[{n}C"
    return f"\03[{n}G"


# https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
# u001b = x1b
# Black: "\x1b[30m"
# Red: "\x1b[31m"
# Green: "\x1b[32m"
# Yellow: "\x1b[33m"
# Blue: "\x1b[34m"
# Magenta: "\x1b[35m"
# Cyan: "\x1b[36m"
# White: "\x1b[37m"
# Reset: "\x1b[0m"

def _test_text_colors_256():
    for i in range(0, 16):
        for j in range(0, 16):
            code = str(i * 16 + j)
            print("\x1b[38;5;" + code + "m " + code.ljust(4), end="")
        print(u"\x1b[0m")  # normal text color


def _test_background_colors_256():
    for i in range(0, 16):
        for j in range(0, 16):
            code = str(i * 16 + j)
            print(u"\x1b[48;5;" + code + "m " + code.ljust(4), end="")
        print(u"\x1b[0m")


terminal_size = shutil.get_terminal_size(fallback=(80, 20))
# print("This will be overwritten", end='\r')
