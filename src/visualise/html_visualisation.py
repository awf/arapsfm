# html_visualisation.py

# Imports
import os
import shutil
import subprocess
import numpy as np
import pprint
import StringIO
import datetime
from operator import itemgetter

# Defaults
DEF_VIS_SCRIPT = 'visualise/visualise_standalone.py'

# Templates
ENTRY_TEMPLATE = '''<div class="vis_info">
    {subheading}
    <div>
    <p>
    {variable_summary}
    </p>
    </div>
    {images}
</div>
'''
IMG_TEMPLATE = '<a href="{image_path}"><img src="{thumbnail_path}"></a>'

VARIABLE_TEMPLATE = '<strong>{key}</strong>: {value}'

INDEX_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" type="text/css" href="style.css" />
</head>
<body>
{body}
</body>
</html>
'''

# variable_summary
def variable_summary(input_path, variables):
    z = np.load(input_path)

    r = []
    for key in variables:
        if key not in z.keys():
            continue

        output = StringIO.StringIO()
        pprint.pprint(z[key], stream=output)
        r.append((key, output.getvalue().rstrip()))

    return r

# VisualisationPage
class VisualisationPage(object):
    def __init__(self, project_root, title=None, subheading=None,
                 vis_args=[], vis_vars=[], var_aliases={},
                 vis_script=DEF_VIS_SCRIPT):

        if not os.path.exists(project_root):
            os.makedirs(project_root)

        self.title = title
        self.subheading = subheading
        self.project_root = project_root
        self.vis_args = vis_args
        self.vis_vars = vis_vars
        self.var_aliases = var_aliases
        self.vis_script = vis_script

        self.tests = []

    def add_test(self, input_path, output_subdir=None, subheading=None,
                 skip_summary=False, **kwargs):

        vis_script = kwargs.get('vis_script', self.vis_script)
        vis_args = kwargs.get('vis_args', self.vis_args)
        vis_vars = kwargs.get('vis_vars', self.vis_vars)

        # create figures by calling `visualise_standalone`
        source_dir, filename = os.path.split(input_path)
        root, ext = os.path.splitext(filename)

        if output_subdir is None:
            output_subdir = root

        output_dir = os.path.join(self.project_root, output_subdir)

        args = ['python', vis_script, input_path,                          
                '-o', output_dir] + vis_args

        print 'Calling:', ' '.join(args)
        subprocess.check_call(args)

        # make index entry
        all_im_files = sorted(os.listdir(output_dir))
        image_numbers = []
        for filename in all_im_files:
            head, ext = os.path.splitext(filename)
            try:
                head_int = int(head)
            except:
                continue

            image_numbers.append((head_int, filename))

        imgs = []
        for number, filename in sorted(image_numbers, key=itemgetter(0)):
            head, ext = os.path.splitext(filename)
            thumbnail_filename = head + '_thumbnail' + ext

            img = IMG_TEMPLATE.format(
                image_path=os.path.join(output_subdir, filename),
                thumbnail_path=os.path.join(output_subdir, thumbnail_filename))

            imgs.append(img)

        # map_key_to_alias
        def map_key_to_alias(key):
            return self.var_aliases.get(key, key)
            
        if skip_summary:
            summary = []
        else:
            summary = map(lambda t: 
                VARIABLE_TEMPLATE.format(key=map_key_to_alias(t[0]), value=t[1]),
                variable_summary(input_path, vis_vars))

        if subheading is not None:
            subheading = '<h1 class="subheading">%s</h1>' % subheading
        else:
            subheading = ''

        entry = ENTRY_TEMPLATE.format(
            variable_summary='<br>\n    '.join(summary),
            images='\n    '.join(imgs),
            subheading=subheading)

        self.tests.append(entry)

    def generate(self):
        lines = []
        if self.title is not None:
            lines.append('<h1>%s</h1>' % self.title)

        if self.subheading is not None:
            lines.append('<h2>%s</h2>' % self.subheading)

        date = datetime.date.today()
        lines.append('<h3>%s</h3>' % date.strftime('%d %B %Y'))

        body = '\n'.join(lines + self.tests)

        output_path = os.path.join(self.project_root, 'index.html')
        with open(output_path, 'w') as fp:
            fp.write(INDEX_TEMPLATE.format(body=body))

        shutil.copy('visualise/style.css', 
                    os.path.join(self.project_root, 'style.css'))
        
# main_single_frames
def main_single_frames():
    page = VisualisationPage('Cheetah_4',
        vis_vars=['mesh',
                  'input', 
                  'solver', 
                  'user_constraints', 
                  'output', 
                  'input_frame', 
                  'lambdas', 
                  'preconditioners',
                  'solver_options', 
                  'narrowband'],

        vis_args=['-c', 'SetParallelProjection=True,',
                  '-c', 'Azimuth=-90,', 
                  '-c', 'Azimuth=0',
                  '-c', 'Azimuth=30',
                  '-c', 'Azimuth=30',
                  '-c', 'Azimuth=30',
                  '-c', 'Azimuth=30', 
                  '-c', 'Azimuth=30',
                  '-c', 'Azimuth=30',
                  '-c', 'Azimuth=-90,',
                  '-c', 'Elevation=60',
                  '-a', 'model:SetRepresentation=3',
                  '--magnification', '3'])

    page.add_test('cheetah1/Cheetah_4_1_0.dat')
    page.add_test('cheetah1/Cheetah_4_1_1.dat')
    page.add_test('cheetah1/Cheetah_4_1_2.dat')
    page.add_test('cheetah1/Cheetah_4_2_0.dat')
    page.add_test('cheetah1/Cheetah_4_2_1.dat')
    page.add_test('cheetah1/Cheetah_4_2_2.dat')
    page.add_test('cheetah1/Cheetah_4_3_0.dat')
    page.add_test('cheetah1/Cheetah_4_3_1.dat')
    page.add_test('cheetah1/Cheetah_4_3_2.dat')
    page.add_test('cheetah1/Cheetah_4_4_0.dat')
    page.add_test('cheetah1/Cheetah_4_4_1.dat')
    page.add_test('cheetah1/Cheetah_4_4_2.dat')
    page.generate()

# main_multiple_frames
def main_multiple_frames():
    for i in xrange(4):
        if i == 0:
            # Test Laplacian lambdas
            test = 'test_laplacian_lambdas' 
            title = 'Multiple frame core recovery: Testing Laplacian lambdas'

            def subheading_fn(path):
                head, file_ = os.path.split(path)
                root, ext = os.path.splitext(file_)

                if root == 'core':
                    z = np.load(path)
                    return ('Frames: %s, lambda: %.3f' %
                            (str(z['indices']), z['lambdas'][5]))

                return None

        if i == 1:
            # Test frame subsets
            test = 'test_frame_subsets' 
            title = 'Multiple frame core recovery: Testing frame subsets'

            def subheading_fn(path):
                head, file_ = os.path.split(path)
                root, ext = os.path.splitext(file_)

                if root == 'core':
                    z = np.load(path)
                    return 'Frames: ' + str(z['indices'])

                return None

        if i == 2:
            # Test ARAP Lambda
            test = 'test_arap_lambda'
            title = 'Multiple frame core recovery: Testing ARAP lambda'

            def subheading_fn(path):
                head, file_ = os.path.split(path)
                root, ext = os.path.splitext(file_)

                if root == 'core':
                    z = np.load(path)
                    arap_lambda = z['lambdas'][3]
                    return 'lambda: %.3f' % arap_lambda

                return None

        if i == 3:
            # Test Laplacians 2
            test = 'test_laplacian_lambda2' 
            title = ('Multiple frame core recovery: '
                     'Testing Laplacian lambdas 2 (four frames)')

            def subheading_fn(path):
                head, file_ = os.path.split(path)
                root, ext = os.path.splitext(file_)

                if root == 'core':
                    z = np.load(path)
                    return ('Frames: %s, lambda: %.3f' %
                            (str(z['indices']), z['lambdas'][5]))

                return None

        # General subheading
        subheading = 'cheetah1:Cheetah_4'

        # setup page
        data_root ='cheetah1_Cheetah4/%s/output_data/' % test

        page = VisualisationPage('cheetah1_Cheetah4/%s/page' % test,
            title=title,
            subheading=subheading,

            vis_vars=['index',
                      'mesh',
                      'lambdas', 
                      'preconditioners',
                      'narrowband',
                      'uniform_weights',
                      'max_restarts',
                      'find_circular_path',
                      'frames',
                      'indices'],
            var_aliases={'indices':'frame number(s)',
                         'index':'frame number'},
            vis_args=['--output_image_first',
                      '-c', 'SetParallelProjection=True,',
                      '-c', 'Azimuth=0',
                      '-c', 'Azimuth=-90,', 
                      '-c', 'Azimuth=0',
                      '-c', 'Azimuth=30',
                      '-c', 'Azimuth=30',
                      '-c', 'Azimuth=30',
                      '-c', 'Azimuth=30', 
                      '-c', 'Azimuth=30',
                      '-c', 'Azimuth=30',
                      '-c', 'Azimuth=-90,',
                      '-c', 'Elevation=60',
                      '-a', 'model:SetRepresentation=3',
                      '--magnification', '3'])

        def file_key(filename):
            root, ext = os.path.splitext(filename)
            try:
                return int(root)
            except ValueError:
                return -1

        for subdir in sorted(next(os.walk(data_root))[1], key=int):
            subdir_path = os.path.join(data_root, str(subdir))
            files = sorted(os.listdir(subdir_path), key=file_key)

            for file_ in files:
                path = os.path.join(subdir_path, file_)
                subheading = subheading_fn(path)

                print path
                root, ext = os.path.splitext(file_)

                page.add_test(path, 
                              output_subdir='%s-%s' % (subdir, root),
                              subheading=subheading)

        page.generate()

# main_cheetah1B_Cheetah_4
def main_cheetah1B_Cheetah_4():
    title = 'Multiple frame core recovery: cheetah1B_Cheetah_4'

    def subheading_fn(path):
        head, file_ = os.path.split(path)
        root, ext = os.path.splitext(file_)

        if root == 'core':
            z = np.load(path)
            return ('Frames: %s, ARAP lambda: %.3f' %
                    (str(z['indices']), z['lambdas'][3]))

        return None

    # General subheading
    subheading = 'cheetah1B:Cheetah_4'

    # setup page
    data_root ='cheetah1B/Cheetah_4_All_D'

    page = VisualisationPage('cheetah1B/Cheetah4_All_D-PAGE',
        title=title,
        subheading=subheading,

        vis_vars=['index',
                  'mesh',
                  'lambdas', 
                  'preconditioners',
                  'narrowband',
                  'uniform_weights',
                  'max_restarts',
                  'find_circular_path',
                  'frames',
                  'indices'],
        var_aliases={'indices':'frame number(s)',
                     'index':'frame number'},
        vis_args=['--output_image_first',
                  '-c', 'SetParallelProjection=True,',
                  '-c', 'Azimuth=0',
                  '-c', 'Azimuth=-90,', 
                  '-c', 'Azimuth=0',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45', 
                  '-c', 'Azimuth=-90,',
                  '-c', 'Elevation=60',
                  '-a', 'model:SetRepresentation=3',
                  '--magnification', '3'])

    def file_key(filename):
        root, ext = os.path.splitext(filename)
        try:
            return int(root)
        except ValueError:
            return -1

    files = sorted(os.listdir(data_root), key=file_key)
    print 'files:'
    pprint.pprint(files)

    for file_ in files:
        path = os.path.join(data_root, file_)
        subheading = subheading_fn(path)

        print path
        root, ext = os.path.splitext(file_)

        page.add_test(path, subheading=subheading)

    page.generate()

# main_polynomial_residual_transform
def main_polynomial_residual_transform():
    title = 'Single frame recovery w/ PiecewisePolynomialTransform'

    page = VisualisationPage('Cheetah_4_6_PiecewisePolynomialTransform',
        vis_vars=['mesh',
                  'input', 
                  'solver', 
                  'user_constraints', 
                  'output', 
                  'input_frame', 
                  'lambdas', 
                  'preconditioners',
                  'polynomial_piecewise',
                  'solver_options', 
                  'narrowband'],

        vis_args=['-c', 'SetParallelProjection=True,',
                  '-c', 'Azimuth=-90,', 
                  '-c', 'Azimuth=0',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45', 
                  '-c', 'Azimuth=-90,',
                  '-c', 'Elevation=60',
                  '-a', 'model:SetRepresentation=3',
                  '--magnification', '3'])


    def subheading_fn(path):
        z = np.load(path)
        tau, p = map(float, z['piecewise_polynomial'].split(','))
        if p == 2.0:
            return 'Quadratic Only'
        else:
            return 'Extension: tau = %.3f (pixels), p = %.3f' % (tau, p)

    for path in ('Cheetah_4_6_wRegularExtension.dat',
                 'Cheetah_4_6_wCubicExtension.dat'):
        subheading = subheading_fn(path)
        page.add_test(path, subheading=subheading)

    page.generate()

# main_cheetah1B_Cheetah_5
def main_cheetah1B_Cheetah_5():
    # Test Laplacian lambdas
    title = 'Multiple frame core recovery: cheetah1B_Cheetah_5'

    def subheading_fn(path):
        head, file_ = os.path.split(path)
        root, ext = os.path.splitext(file_)

        if root == 'core':
            z = np.load(path)
            return ('Frames: %s, ARAP lambda: %.3f' %
                    (str(z['indices']), z['lambdas'][3]))

        return None

    # General subheading
    subheading = 'cheetah1B:Cheetah_5'

    # setup page
    # data_root = 'cheetah1B/Cheetah_5/Experiments/0-6-8/'
    data_root = 'cheetah1B/Cheetah_5/Experiments/3-4-6-8-12/'

    page = VisualisationPage('cheetah1B/Cheetah_5/Experiments/3-4-6-8-12-PAGE',
        title=title,
        subheading=subheading,

        vis_vars=['index',
                  'mesh',
                  'lambdas', 
                  'preconditioners',
                  'narrowband',
                  'uniform_weights',
                  'max_restarts',
                  'find_circular_path',
                  'frames',
                  'indices'],
        var_aliases={'indices':'frame number(s)',
                     'index':'frame number'},
        vis_args=['--output_image_first',
                  '-c', 'SetParallelProjection=True,',
                  '-c', 'Azimuth=0',
                  '-c', 'Azimuth=-90,', 
                  '-c', 'Azimuth=0',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45', 
                  '-c', 'Azimuth=-90,',
                  '-c', 'Elevation=60',
                  '-a', 'model:SetRepresentation=3',
                  '--magnification', '3'])

    def file_key(filename):
        root, ext = os.path.splitext(filename)
        try:
            return int(root)
        except ValueError:
            return -1

    files = sorted(os.listdir(data_root), key=file_key)
    print 'files:'
    pprint.pprint(files)

    for file_ in files:
        path = os.path.join(data_root, file_)

        if os.path.isdir(path):
            continue

        subheading = subheading_fn(path)

        print path
        root, ext = os.path.splitext(file_)

        page.add_test(path, subheading=subheading)

    page.generate()

# main_scaled_rotations
def main_scaled_rotations():
    title = 'Scaling of individual ARAP rotations in axis-angle representation'

    scales = '(0.5, 0.75, 1.0, 1.25, 1.5)'

    page = VisualisationPage('Cheetah_4_6_Individual_ARAP_Rotations',
        vis_script = 'visualise/visualise_scaled_rotations.py',
        title=title,

        vis_vars=['mesh',
                  'input', 
                  'solver', 
                  'user_constraints', 
                  'output', 
                  'input_frame', 
                  'lambdas', 
                  'preconditioners',
                  'polynomial_piecewise',
                  'solver_options', 
                  'narrowband'],

        vis_args=[scales,
                  '-c', 'SetParallelProjection=True,',
                  '-c', 'Azimuth=0', 
                  '-c', 'Azimuth=-90,', 
                  '-c', 'Azimuth=0',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45',
                  '-c', 'Azimuth=45', 
                  '-c', 'Azimuth=-90,',
                  '-c', 'Elevation=60',
                  '--magnification', '3'])

    page.add_test('cheetah1B/Cheetah_4/Experiments/Cheetah_4_6.dat',
                  subheading='Scales: %s' % scales)

    page.generate()

# test_variable_summary
def test_variable_summary():
    input_path = '../working/chihuahua_lap_silhouette.npz'
    print variable_summary(input_path, 
        variables=['lm_lambdas', 'lm_preconditioners'])

# main_nonlinear_basis_preconditioner_test
def main_nonlinear_basis_preconditioner_test():
    title = ('Non-linear rotation basis - Changing preconditioner on basis '
             'coefficients')

    scales = np.linspace(-0.5, 0.5, 6)
    scales = str(tuple(np.around(scales, decimals=2)))

    def_vis_args = ['--output_image_first',
                    '-c', 'SetParallelProjection=True,',
                    '-c', 'Azimuth=0', 
                    '-c', 'Azimuth=-90,', 
                    '-c', 'Azimuth=0',
                    '-c', 'Azimuth=45',
                    '-c', 'Azimuth=45',
                    '-c', 'Azimuth=45',
                    '-c', 'Azimuth=45', 
                    '-c', 'Azimuth=-90,',
                    '-c', 'Elevation=60',
                    '--magnification', '3',
                    '-a', 'model:SetRepresentation=3']

    page = VisualisationPage('Cheetah_4_6-7-8_Preconditioner_Test',
        title=title,
        var_aliases={'indices':'frame number(s)',
                     'index':'frame number'},
        vis_vars=['mesh',
                  'lambdas', 
                  'preconditioners',
                  'solver_options', 
                  'uniform_weights',
                  'find_circular_path',
                  'max_restarts',
                  'narrowband',
                  'num_basis_rotations',
                  'indices',
                  'index'],

        vis_args=def_vis_args)

    page.add_test('6-7-8_NB=1_y_=64.0/core.npz', 
                  vis_script='visualise/visualise_scaled_rotations.py',
                  vis_args=[scales, 
                            '--rotations_index', '0', '--affine_register'] 
                            + def_vis_args[1:-2])
    page.add_test('6-7-8_NB=1_y_=64.0/6.npz')
    page.add_test('6-7-8_NB=1_y_=64.0/7.npz')
    page.add_test('6-7-8_NB=1_y_=64.0/8.npz')

    page.generate()

# main_nonlinear_basis_experiment
def main_nonlinear_basis_experiment(dir_, title, output, *args):
    scales = np.linspace(-1.0, 1.0, 6)
    scales = str(tuple(np.around(scales, decimals=2)))

    def_vis_args = ['--with_core', None,
                    '--output_image_first',
                    '-c', 'SetParallelProjection=True,',
                    '-c', 'Azimuth=0', 
                    '-c', 'Azimuth=-90,', 
                    '-c', 'Azimuth=0',
                    '-c', 'Azimuth=45',
                    '-c', 'Azimuth=45',
                    '-c', 'Azimuth=45',
                    '-c', 'Azimuth=45', 
                    '-c', 'Azimuth=-90,',
                    '-c', 'Elevation=60',
                    '--magnification', '3',
                    '-a', 'model:SetRepresentation=3']

    full_path = lambda f: os.path.join(dir_, f)

    def valid_file(f):
        return os.path.isfile(f) and (f.endswith('.dat') or f.endswith('.npz'))
    files = filter(lambda f: valid_file(full_path(f)), os.listdir(dir_))

    def key(f):
        root, ext = os.path.splitext(f)
        try:
            return int(root)
        except ValueError:
            return -1

    sorted_files = sorted(files, key=key)
    def_vis_args[1] = full_path(sorted_files[0])

    page = VisualisationPage(output,
        title=title,
        var_aliases={'indices':'frame number(s)',
                     'index':'frame number'},
        vis_vars=['mesh',
                  'lambdas', 
                  'preconditioners',
                  'piecewise_polynomial',
                  'solver_options', 
                  'uniform_weights',
                  'find_circular_path',
                  'max_restarts',
                  'narrowband',
                  'num_basis_rotations',
                  'indices',
                  'index'],

        vis_args=def_vis_args)

    for f in sorted_files:
        abs_f = full_path(f)

        if 'core' in f:
            # skip core
            continue

            z = np.load(abs_f)
            if 'num_basis_rotations' in z.keys():
                # show core with deformations
                num_basis_rotations = z['num_basis_rotations']

                for i in xrange(num_basis_rotations):
                    page.add_test(abs_f,
                                  vis_script=('visualise/'
                                              'visualise_scaled_rotations.py'),
                                  vis_args=[scales, 
                                            '--rotations_index', str(i),
                                            '--rigidly_register',
                                            '--normalise_rotations'] 
                                            + def_vis_args[3:-2],
                                  output_subdir='core_%d' % i,
                                  skip_summary=(i > 0))
            else:
                # show core as a regular test
                page.add_test(abs_f)
                
        else:
            # page.add_test(abs_f)
            page.add_test(abs_f, vis_args=def_vis_args[2:] + 
                          ['--subdivide', '2', '--compute_normals'])

    page.generate()

# main
def main():
    # walker = os.walk('cheetah1B/Cheetah_4/Experiments/NonlinearRotationBasis/')
    # walker = os.walk('/mnt/sshfs/RichardStebbing_fbg-cluster/Code/Projects/Core_Recovery/')
    # walker = os.walk('.')
    walker = os.walk('20121110')

    dir_, subdirs, files = next(walker)

    for subdir in subdirs:
        # if '4B_5-6-10_NB=2_y_=128.0_ARAP=4.0_U=64.0_R=64.0_s=128.0_1e-4_R2=128.0_PP' not in subdir:
        #     continue
        if 'OLD' in subdir:
            continue

        full_path = os.path.join(dir_, subdir)
        title = subdir
        output = os.path.join('NonlinearRotationBasis/TO/Subdivided/', subdir)

        main_nonlinear_basis_experiment(full_path, title, output)

if __name__ == '__main__':
    # test_variable_summary()
    # main()
    # main_single_frames()
    # main_cheetah1B_Cheetah_4()
    # main_cheetah1B_Cheetah_5()
    # main_polynomial_residual_transform()
    # main_scaled_rotations()
    # main_nonlinear_basis_experiment()
    main()
    
