# visualise.py

# Imports
import numpy as np
from vtk_ import *
from pprint import pprint

# Color conversion

# int2dbl
def int2dbl(*x):
    return np.array(x, dtype=np.float64) / 255. 

# Visualisation

if HAS_VTK:
    # VisualiseMesh
    class VisualiseMesh(object):
        def __init__(self):
            self.objects = []
            self.actors = {}
            self._init_renderer()

        def _init_renderer(self):
            ren = vtk.vtkRenderer()
            ren.SetBackground(1., 1., 1.)

            ren_win = vtk.vtkRenderWindow()
            ren_win.AddRenderer(ren)
            ren_win.SetSize(500, 500)

            # enable opacity changes
            ren_win.SetAlphaBitPlanes(True)
            ren_win.SetMultiSamples(8)
            ren.SetUseDepthPeeling(True)
            ren.SetMaximumNumberOfPeels(100)
            ren.SetOcclusionRatio(0.)

            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(ren_win)

            style = InteractorStyle()
            iren.SetInteractorStyle(style)

            self.ren = ren
            self.ren_win = ren_win
            self.iren = iren

        def add_mesh(self, V, T, L=None, actor_name='model',
                     compute_normals=False, feature_angle=90.):
            cells = faces_to_vtkCellArray(T)

            self.keys = {}
            self.V = V
            self.T_ = cells

            self.objects = []
            model_pd = numpy_to_vtkPolyData(V, cells)

            if compute_normals:
                pd_normals = vtk.vtkPolyDataNormals()
                pd_normals.SetInput(model_pd)
                pd_normals.ComputeCellNormalsOn()
                pd_normals.SetFeatureAngle(feature_angle)
                pd_normals.Update()
                model_pd = pd_normals.GetOutput()

            # setup the programmable filter to color specific faces
            color_face = vtk.vtkProgrammableFilter()
            color_face.SetInput(model_pd)

            def color_face_callback():
                input_ = color_face.GetInput()
                output = color_face.GetOutput()
                output.ShallowCopy(input_)

                color_lookup = vtk.vtkIntArray()
                color_lookup.SetNumberOfValues(model_pd.GetNumberOfPolys())
                npy_cl = vtk_to_numpy(color_lookup)
                npy_cl.fill(0)

                if L is not None:
                    npy_cl[L] = 1

                output.GetCellData().SetScalars(color_lookup)

            color_face.SetExecuteMethod(color_face_callback)
            self.objects.append(color_face)

            # setup the lookup table for coloring the different cells
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(2)
            lut.Build()

            lut_SetTableValue = lut.SetTableValue
            def SetTableValue(i, r, g, b, alpha=1.0):
                return lut_SetTableValue(i, r, g, b, alpha)
            lut.SetTableValue = SetTableValue
            lut.SetTableValue(0, *int2dbl(31, 120, 180))
            lut.SetTableValue(1, *int2dbl(178, 223, 138))

            # setup model mapper, actor, renderer and render window
            model_mapper = vtk.vtkPolyDataMapper()
            model_mapper.SetInputConnection(color_face.GetOutputPort())
            model_mapper.SetScalarRange(0, 1)
            model_mapper.SetLookupTable(lut)

            model_actor = vtk.vtkActor()
            model_actor.SetMapper(model_mapper)
            model_actor.VisibilityOn()
            model_actor.GetProperty().SetLighting(True)
            model_actor.GetProperty().SetRepresentationToWireframe()
            model_actor.GetProperty().SetOpacity(1.0)

            self.actors[actor_name] = model_actor
            self.ren.AddActor(model_actor)

        def add_image(self, filename):
            reader = vtk.vtkPNGReader()
            reader.SetFileName(filename)

            cast = vtk.vtkImageCast()
            cast.SetInput(reader.GetOutput())
            cast.SetOutputScalarTypeToUnsignedChar()
            cast.Update()

            image_data = cast.GetOutput()
            self.objects.append(image_data)

            image_actor = vtk.vtkImageActor()
            image_actor.SetInput(image_data)
            image_actor.PickableOff()

            self.actors['image'] = image_actor
            self.ren.AddActor(image_actor)

        def add_laplacians(self, laplacians):
            V = self.V
            laplacian_points = vtk.vtkPoints()
            laplacian_points.SetNumberOfPoints(2*V.shape[0])

            laplacian_lines = vtk.vtkCellArray()

            for i in xrange(V.shape[0]):
                laplacian_points.SetPoint(2*i, V[i])
                laplacian_points.SetPoint(2*i + 1, V[i] - laplacians[i])

                laplacian_lines.InsertNextCell(2)
                laplacian_lines.InsertCellPoint(2*i)
                laplacian_lines.InsertCellPoint(2*i + 1)

            laplacian_pd = vtk.vtkPolyData()
            laplacian_pd.SetPoints(laplacian_points)
            laplacian_pd.SetLines(laplacian_lines)

            laplacian_tube = vtk.vtkTubeFilter()
            laplacian_tube.SetInput(laplacian_pd)
            laplacian_tube.SetRadius(0.1)
            laplacian_tube.SetNumberOfSides(32)
            self.objects.append(laplacian_tube)

            laplacian_mapper = vtk.vtkPolyDataMapper()
            laplacian_mapper.SetInputConnection(laplacian_tube.GetOutputPort())

            laplacian_actor = vtk.vtkActor()
            laplacian_actor.SetMapper(laplacian_mapper)
            laplacian_actor.GetProperty().SetColor(*int2dbl(106, 61, 154))
            laplacian_actor.GetProperty().SetOpacity(1.0)
            laplacian_actor.GetProperty().SetLighting(True)
            laplacian_actor.VisibilityOn()

            self.keys['l'] = 'laplacian'
            self.actors['laplacian'] = laplacian_actor
            self.ren.AddActor(laplacian_actor)

        def add_silhouette(self, Q, path, bounds, S):
            path_length = path.shape[0]

            proj_points = vtk.vtkPoints()
            proj_points.SetNumberOfPoints(2*path_length)

            proj_lines = vtk.vtkCellArray()

            j = 0
            i = bounds[0]
            done = False
            while not done: 
                proj_lines.InsertNextCell(2)

                proj_points.SetPoint(2*j, *np.r_[S[i], 0])
                proj_points.SetPoint(2*j + 1, *Q[path[j]])

                proj_lines.InsertCellPoint(2*j)
                proj_lines.InsertCellPoint(2*j + 1)

                if i == bounds[1] and j > 0:
                    done = True
                else:
                    j += 1
                    i = (i + 1) % S.shape[0]

            proj_poly_data = vtk.vtkPolyData()
            proj_poly_data.SetPoints(proj_points)
            proj_poly_data.SetLines(proj_lines)

            proj_tube = vtk.vtkTubeFilter()
            proj_tube.SetInput(proj_poly_data)
            proj_tube.SetRadius(1.0)
            proj_tube.SetNumberOfSides(8)
            self.objects.append(proj_tube)
            
            proj_mapper = vtk.vtkPolyDataMapper()
            proj_mapper.SetInputConnection(proj_tube.GetOutputPort())

            proj_actor = vtk.vtkActor()
            proj_actor.SetMapper(proj_mapper)
            proj_actor.GetProperty().SetColor(*int2dbl(253, 191, 111))
            proj_actor.GetProperty().SetLighting(True)
            proj_actor.VisibilityOn()
            proj_actor.PickableOff()

            self.keys['h'] = 'silhouette_projection'
            self.actors['silhouette_projection'] = proj_actor
            self.ren.AddActor(proj_actor)
          
        def add_projection(self, C, P):
            V = self.V

            # setup the projection lines
            proj_points = vtk.vtkPoints()
            proj_points.SetNumberOfPoints(2*C.shape[0])

            proj_lines = vtk.vtkCellArray()

            for j, i in enumerate(C):
                proj_lines.InsertNextCell(2)

                proj_points.SetPoint(2*j, *np.r_[P[j], 0])
                proj_points.SetPoint(2*j + 1, *V[i])

                proj_lines.InsertCellPoint(2*j)
                proj_lines.InsertCellPoint(2*j + 1)

            proj_poly_data = vtk.vtkPolyData()
            proj_poly_data.SetPoints(proj_points)
            proj_poly_data.SetLines(proj_lines)

            proj_tube = vtk.vtkTubeFilter()
            proj_tube.SetInput(proj_poly_data)
            proj_tube.SetRadius(3.0)
            proj_tube.SetNumberOfSides(8)

            proj_mapper = vtk.vtkPolyDataMapper()
            proj_mapper.SetInputConnection(proj_tube.GetOutputPort())

            proj_actor = vtk.vtkActor()
            proj_actor.SetMapper(proj_mapper)
            proj_actor.GetProperty().SetColor(*int2dbl(227, 26, 28))
            proj_actor.GetProperty().SetLighting(True)
            proj_actor.VisibilityOn()

            self.keys['r'] ='projection_constraints' 
            self.actors['projection_constraints'] = proj_actor
            self.ren.AddActor(proj_actor)

        def actor_properties(self, actor_name, *actions):
            prop = self.actors[actor_name].GetProperty()
            for key, value in actions:
                getattr(prop, key)(*value)
            
        def camera_actions(self, *actions):
            camera = self.ren.GetActiveCamera()
            for key, value in actions:
                getattr(camera, key)(*value)

            self.ren.ResetCamera()

        def execute(self, magnification=1):
            self.ren_win.Render()

            self.iren.GetInteractorStyle().SetMagnification(magnification)
            self.iren.Initialize()
            self.iren.Start()

        def write(self, filename, magnification=1):
            self.ren_win.Render()

            render_large = vtk.vtkRenderLargeImage()
            render_large.SetInput(self.ren)
            render_large.SetMagnification(magnification)

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(filename)
            writer.SetInputConnection(render_large.GetOutputPort())
            writer.Write()
else:
    class VisualiseMesh(object):
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, attr):
            return lambda *args, **kwargs: None

# visualise_multiple_geometries
def visualise_multiple_geometries(mesh_objects):
    N = len(mesh_objects)

    # rectangles
    x = 0.
    rects = []
    for i in xrange(N):
        x1 = x + (1. / N) if i < (N-1) else 1.0
        # rects.append((x, 0., x1, 1.))
        rects.append((0., x, 1., x1))
        x = x1
        
    style = None
    renderers = []

    for i, r in enumerate(rects): 
        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
        ren.SetViewport(*r)

        for actor in mesh_objects[i].actors.itervalues():
            ren.AddActor(actor)

        camera = ren.GetActiveCamera()
        camera.SetParallelProjection(True)
        ren.ResetCamera()

        if style is None:
            try:
                style = mesh_objects[i].style
            except AttributeError:
                pass
            
        renderers.append(ren)

    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(600, 600)

    for ren in renderers:
        renWin.AddRenderer(ren)

    if style is None:
        style = vtk.vtkInteractorStyleTrackballCamera()

    style.SetCurrentRenderer(renderers[0])
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    iren.Initialize()
    renWin.Render()
    iren.Start()

# bool_from_string
def bool_from_string(string):
    if string == 'True': return True
    if string == 'False' : return False

    raise ValueError('unable to create bool from string: %s' %
                     string)

# tuple_from_string
def tuple_from_string(string):
    if not string: return tuple()

    tup_strings = string.split(',')
    for type_ in [int, float, bool_from_string]:
        try:
            return tuple(type_(v) for v in tup_strings)
        except ValueError:
            continue

    raise ValueError('unable to create tuple from string: %s' %
                     string)

# parse_camera_action
def parse_camera_action(action):
    # get method and value string
    method, value_string = action.split('=')

    # check if immediately save rendering after this camera action
    save_after = True
    if value_string.endswith(','):
        save_after = False
        value_string = value_string[:-1]
    
    # convert value string to tuple
    tup = tuple_from_string(value_string)

    return method, tup, save_after

