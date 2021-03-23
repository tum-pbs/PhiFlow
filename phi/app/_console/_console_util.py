import shutil

# Unicode special characters
FULL_BLOCK = "\u2588"
# ANSI escape
MOVE_TO_PREV_LINE_START = "\033[F"
CLEAR_LINE = "\033[K"


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
