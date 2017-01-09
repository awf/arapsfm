# project.py

# Imports
import os
import numpy as np
from mesh import faces

# Constants
DATA_ROOT = 'data'
INPUT_MODEL_DIR = os.path.join(DATA_ROOT, 'models')
PROJECTIONS_ROOT = os.path.join(DATA_ROOT, 'projection_constraints')
SILHOUETTES_ROOT = os.path.join(DATA_ROOT, 'silhouettes')
FRAMES_ROOT = os.path.join(DATA_ROOT, 'frames')

OUTPUT_ROOT = 'working'

# Workspace
class Workspace(object):
    def __init__(self, model, frame_stem):
        self.model = model
        self.frame_stem = frame_stem

    def load_triangles(self):
        z = np.load(os.path.join(INPUT_MODEL_DIR, self.model + '.npz'))
        return faces.faces_from_cell_array(z['cells'])

    def load_projection(self, stem):
        path = os.path.join(PROJECTIONS_ROOT, self.model,  stem + '.npz')
        z = np.load(path)

        # load
        C = np.asarray(z['point_ids'], dtype=np.int32)
        P = np.asarray(z['positions'], dtype=np.float64)
        T = np.asarray(z['T'], dtype=np.float64)
        V = np.asarray(z['V'], dtype=np.float64)

        S = T[:3, :3]
        t = T[:3, -1]

        V = np.dot(V, np.transpose(S)) + t

        return V, C, P

    def get_frame_path(self, index):
        return os.path.join(FRAMES_ROOT, self.frame_stem, '%d.png' % index)

    def save(self, file_suffix, **kwargs):
        output_path = os.path.join(OUTPUT_ROOT, '%s_%s.npz' % 
            (self.model, file_suffix))

        np.savez_compressed(output_path, **kwargs)

    def load(self, file_suffix):
        path = os.path.join(OUTPUT_ROOT, '%s_%s.npz' % (self.model, 
            file_suffix))

        return np.load(path)

    def load_silhouette_info(self):
        z = self.load('silhouette_info')
        return {k:z[k] for k in z.keys()}

    def get_silhouette(self, index):
        z = np.load(os.path.join(SILHOUETTES_ROOT, self.frame_stem, '%d_S.npz' % index))
        return z['S'], z['SN']

        

