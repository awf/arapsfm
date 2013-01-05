# avi_visualisation.py

# Imports
import argparse, os
import shutil
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
def avi_visualisation(vis_script, input_dir, output_dir, fps, 
                      separate_frames=False, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vis_args = kwargs.pop('vis_args')

    safe_cmd(*(['python', vis_script, input_dir, 
                '--output', output_dir] +
               vis_args))

    all_outputs = map(lambda f: os.path.join(output_dir, f),
                      os.listdir(output_dir))
    output_dirs = filter(os.path.isdir, all_outputs)
    sorted_output_dirs = sorted(output_dirs, 
        key=lambda p: int(os.path.split(p)[-1]))
                        
    output_paths = map(lambda d: make_figures(d, **kwargs), 
                       sorted_output_dirs)
    output_paths = reduce(operator.add, output_paths)

    listing_path = os.path.join(output_dir, '_LISTING.txt')
    with open(listing_path, 'w') as fp:
        fp.write('\n'.join(output_paths))

    h, w = imread(output_paths[0]).shape[:2]

    vbitrate = 15000

    for f in fps:
        safe_cmd('mencoder', 
                 'mf://@%s' % listing_path,
                 '-mf', 
                 'w=%d:h=%d:fps=%s:type=png' % (w, h, f),
                 '-ovc', 
                 'lavc', '-lavcopts', 'vcodec=mpeg4:vbitrate=%d:mbd=2' % (vbitrate,),
                 '-oac', 'copy',
                 '-o', os.path.join(output_dir, 'OUTPUT_%d.avi' % f))

    if separate_frames:
        frames_dir = os.path.join(output_dir, 'frames')
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        frame_paths = map(lambda i: os.path.join(frames_dir, '%d.png' % i),
                          xrange(len(output_paths)))

        map(lambda src, dst: safe_cmd('cp', src, dst), output_paths, frame_paths)

# make_figures
def make_figures(output_dir, tiling=None, post_args=None):
    # create initial visualistaions by calling `vis_script`
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
    if len(full_paths) > 1:
        joined_path = os.path.join(output_dir, 'JOINED.png')

        if tiling is None:
            tiling = '%dx' % len(full_paths)

        safe_cmd(*(['montage', 
                    '-depth', '8',
                    '-mode', 'concatenate', 
                    '-tile', tiling] +
                   full_paths + [joined_path]))

        return [joined_path]
    else:
        return [full_paths[0]]

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('vis_script', type=str)
    parser.add_argument('--vis_args', type=str, default='')
    parser.add_argument('--post_args', type=str, default='None')
    parser.add_argument('--tiling', type=str, default=None)
    parser.add_argument('--fps', type=int, default=[], action='append')
    parser.add_argument('--separate_frames', 
                        default=False,
                        action='store_true')

    args = parser.parse_args()

    if not args.fps:
        args.fps.append(25)

    avi_visualisation(args.vis_script, 
                      args.input, 
                      args.output,
                      args.fps,
                      args.separate_frames,
                      vis_args=args.vis_args.split(),
                      tiling=args.tiling,
                      post_args=eval(args.post_args))

if __name__ == '__main__':
    main()

