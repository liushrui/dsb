""" setup python path """
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

lib_path = osp.join(osp.dirname(__file__), '../','lib')
add_path(lib_path)
