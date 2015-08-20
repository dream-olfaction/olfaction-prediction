import sys
import os

utils_path = os.path.dirname(os.path.abspath(__file__))
opc_python_path  = os.path.dirname(utils_path)
root_path = os.path.dirname(opc_python_path)
if root_path not in sys.path:
    sys.path.append(root_path)