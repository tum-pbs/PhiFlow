import io
import os
# from nbformat import read
from unittest import TestCase
# from IPython.core.interactiveshell import InteractiveShell


def run_notebook(path_from_root):
    phiflow_path = os.path.abspath('')
    if phiflow_path.endswith('tests'):
        phiflow_path = os.path.dirname(phiflow_path)
    path = os.path.join(phiflow_path, path_from_root)
    with io.open(path, 'r', encoding='utf-8') as f:
        nb = read(f, 4)
    shell = InteractiveShell.instance()

    for cell in nb.cells:
        if cell.cell_type == 'code':
            code = shell.input_transformer_manager.transform_cell(cell.source)
            print(code)
            exec(code)  # pylint: disable-msg = exec-used
            print('Executed OK.\n\n')



class TestNotebooks(TestCase):

    pass
