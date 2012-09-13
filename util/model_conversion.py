# model_conversion.py

# Imports
import os
import numpy as np
from mesh.faces import faces_from_cell_array
from visualise.vtk_ import vtk, vtkPolyData_to_numpy

EXTENSIONS = {'.obj' : vtk.vtkOBJReader,
              '.stl' : vtk.vtkSTLReader}

# supported_ext
def supported_ext():
    return EXTENSIONS.keys()

# load_mesh
def load_mesh(full_path):
    head, ext = os.path.splitext(full_path)

    reader_class = EXTENSIONS[ext]
    reader = reader_class()

    reader.SetFileName(full_path)
    reader.Update()

    poly_data = reader.GetOutput()
    V, cells = vtkPolyData_to_numpy(poly_data)
    V = V.astype(np.float64)

    return V, cells

