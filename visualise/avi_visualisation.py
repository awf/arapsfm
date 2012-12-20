# avi_visualisation.py

# Imports
import argparse, os
import subprocess
import operator
from matplotlib.pyplot import imread

from pprint import pprint

# safe_cmd
def safe_cmd(*args, **kwargs):
    verbose = kwargs.get('verbose', True)

    if verbose:
        print ' '.join(args)

    # return subprocess.check_output(args, stderr=subprocess.STDOUT)
    return subprocess.check_call(args)

# valid_file
def valid_file(f, extension='.npz'):
    if f.endswith(extension):
        try:
            return int(os.path.splitext(f)[0]) >= 0
        except ValueError:
            pass

    return False

# avi_visualisation
def avi_visualisation(vis_script, input_dir, output_dir, fps, N=0, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_files = filter(valid_file, os.listdir(input_dir))
    sorted_files = sorted(valid_files,
                          key=lambda f: int(os.path.splitext(f)[0]))
    if N > 0:
        sorted_files = sorted_files[:N]

    output_paths = map(lambda f: make_figures(
                       vis_script, 
                       os.path.join(input_dir, f),
                       os.path.join(output_dir, os.path.splitext(f)[0]),
                       **kwargs),
                       sorted_files)

    output_paths = reduce(operator.add, output_paths)

    listing_path = os.path.join(output_dir, '_LISTING.txt')
    with open(listing_path, 'w') as fp:
        fp.write('\n'.join(output_paths))

    h, w = imread(output_paths[0]).shape[:2]

    vbitrate= (50 * 25 * w * h) / 256

    safe_cmd('mencoder', 
             'mf://@%s' % listing_path,
             '-mf', 
             'w=%d:h=%d:fps=%s:type=png' % (w, h, fps),
             '-ovc', 
             'lavc', '-lavcopts', 'vcodec=mpeg4:vbitrate=%d:mbd=2' % (vbitrate,),
             '-oac', 'copy',
             '-o', os.path.join(output_dir, 'OUTPUT.avi'))

# make_figures
def make_figures(vis_script, input_path, output_dir, vis_args=[],
                 post_args=None):

    # create initial visualistaions by calling `vis_script`
    safe_cmd(*(['python', vis_script, input_path, '--output', output_dir] +
               vis_args))

    files = filter(lambda f: valid_file(f, '.png'), os.listdir(output_dir))
    sorted_files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))

    full_paths = map(lambda f: os.path.join(output_dir, f),
                     sorted_files)

    # apply post-processing to the resulting images if required (e.g. cropping)
    if post_args is not None:
        for i, args_i in enumerate(post_args):
            if args_i is not None:
                safe_cmd(*(['convert', full_paths[i]] + 
                           args_i.split() + 
                           [full_paths[i]]))

    # join the images
    joined_path = os.path.join(output_dir, 'JOINED.png')

    safe_cmd(*(['montage', 
                '-depth', '8',
                '-mode', 'concatenate', 
                '-tile', '%dx' % len(full_paths)] +
               full_paths + [joined_path]))

    return [joined_path]

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('vis_script', type=str)
    parser.add_argument('--vis_args', type=str, default='')
    parser.add_argument('--post_args', type=str, default='None')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--N', type=int, default=0)

    args = parser.parse_args()

    avi_visualisation(args.vis_script, 
                      args.input, 
                      args.output,
                      args.fps,
                      args.N,
                      vis_args=args.vis_args.split(),
                      post_args=eval(args.post_args))

if __name__ == '__main__':
    main()

