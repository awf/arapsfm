# six_frame_montage.py

# Imports
import os, argparse
import subprocess
from pprint import pprint

# safe_cmd
def safe_cmd(*args, **kwargs):
    verbose = kwargs.get('verbose', True)

    if verbose:
        print ' '.join(args)

    # return subprocess.check_output(args, stderr=subprocess.STDOUT)
    return subprocess.check_call(args)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_stem')
    parser.add_argument('segmentation_stem')
    parser.add_argument('user_constraints_stem')
    parser.add_argument('three_views')
    parser.add_argument('indices')
    parser.add_argument('output_dir')

    args = parser.parse_args()

    for key in ['indices']:
        setattr(args, key, eval(getattr(args, key)))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_path = lambda i: args.image_stem % i
    segmentation_path = lambda i: args.segmentation_stem % i
    user_constraints_path = lambda i: args.user_constraints_stem % i
    output_image = lambda i: os.path.join(args.output_dir, '%d.png' % i)

    output_paths = []

    for i, index in enumerate(args.indices):
        paths = [image_path(index),
                 segmentation_path(index),
                 user_constraints_path(index)]

        three_view_subdir = os.path.join(args.three_views, '%d' % i)
        paths += map(lambda i: os.path.join(three_view_subdir, '%d.png' % i),
                     xrange(3))

        safe_cmd(*(['montage', '-depth', '8'] + paths +
                   ['-geometry', '618x348>+11+96', output_image(i)]))

        output_paths.append(output_image(i))

    listing_path = os.path.join(args.output_dir, '_LISTING.txt')
    with open(listing_path, 'w') as fp:
        fp.write('\n'.join(output_paths))

    safe_cmd('mencoder',
             'mf://@%s' % listing_path,
             '-mf', 
             'w=1920:h=1080:fps=25:type=png',
             '-ovc', 
             'lavc', '-lavcopts', 'vcodec=mpeg4:vbitrate=%d:mbd=2' % (10000,),
             '-oac', 'copy',
             '-o', os.path.join(args.output_dir, '25.avi'))

if __name__ == '__main__':
    main()
