# html_visualisation.py

# Imports
import os
import shutil
import subprocess
import numpy as np
import pprint
import StringIO

# Templates
ENTRY_TEMPLATE = '''<div class="vis_info">
    <div>
    <p>
    {variable_summary}
    </p>
    </div>
    {images}
</div>
'''
IMG_TEMPLATE = '<a href="{image_path}"><img src="{image_path}"></a>'

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
    def __init__(self, project_root, 
                 vis_args=[], vis_vars=[]):

        if not os.path.exists(project_root):
            os.makedirs(project_root)

        self.project_root = project_root
        self.vis_args = vis_args
        self.vis_vars = vis_vars

        self.tests = []

    def add_test(self, input_path):
        # create figures by calling `visualise_standalone`
        source_dir, filename = os.path.split(input_path)
        root, ext = os.path.splitext(filename)

        output_dir = os.path.join(self.project_root, root)

        args = ['python', 'visualise/visualise_standalone.py',
                input_path,                          
                '-o', output_dir] + self.vis_args

        print 'Calling:', ' '.join(args)
        # subprocess.check_call(args)

        # make index entry
        im_paths = map(lambda f: os.path.join(root, f), 
                       os.listdir(output_dir))
        imgs = map(lambda f: IMG_TEMPLATE.format(image_path=f), im_paths)

        summary = map(lambda t: VARIABLE_TEMPLATE.format(
            key=t[0], value=t[1]), 
            variable_summary(input_path, self.vis_vars))

        entry = ENTRY_TEMPLATE.format(
            variable_summary='<br>\n    '.join(summary),
            images='\n    '.join(imgs))

        self.tests.append(entry)

    def generate(self):
        body = '\n'.join(self.tests)

        output_path = os.path.join(self.project_root, 'index.html')
        with open(output_path, 'w') as fp:
            fp.write(INDEX_TEMPLATE.format(body=body))

        shutil.copy('visualise/style.css', 
                    os.path.join(self.project_root, 'style.css'))
        
# main
def main():
    page = VisualisationPage('TestProject',
        vis_vars=['lm_lambdas', 'lm_preconditioners'],
        vis_args=['-c', 'Azimuth=0', '-c', 'Azimuth=10'])

    page.add_test('working/chihuahua_lap_silhouette.npz')
    page.add_test('working/chihuahua_lap_silhouette.npz')
    page.generate()

# test_variable_summary
def test_variable_summary():
    input_path = '../working/chihuahua_lap_silhouette.npz'
    print variable_summary(input_path, 
        variables=['lm_lambdas', 'lm_preconditioners'])

if __name__ == '__main__':
    # test_variable_summary()
    main()
    
