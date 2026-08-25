"""Sanity checks that the project environment is wired up correctly."""

import numpy as np
import cv2
from PIL import Image

import maxflow
import segmentation


def test_core_dependencies_import():
    assert np.__version__
    assert cv2.__version__
    assert Image.__version__


def test_project_packages_import():
    assert maxflow.__doc__
    assert segmentation.__doc__
