# postprocess_html.py

# Imports
import argparse
import os
import subprocess
from pprint import pprint

# all_pngs
def all_pngs(source_dir):
    all_ = []

    walker = os.walk(source_dir)
    for dir_, subdirs, files in walker:
        for file_ in files:
            root, ext = os.path.splitext(file_)
            if ext != '.png':
                continue

            if root.endswith('_thumbnail'):
                continue

            all_.append(os.path.join(dir_, file_))

    pprint(all_)

    return all_

# trim_image
def trim_image(full_path):
    args = ['convert', full_path, '-trim', 
            '-bordercolor', 'White', '-border', '5x5', '+repage', full_path]

    print 'Calling:', ' '.join(args)
    subprocess.check_call(args)

# make_thumbnail
def make_thumbnail(full_path, max_height):
    head, file_ = os.path.split(full_path)
    root, ext = os.path.splitext(file_)

    output = os.path.join(head, root + '_thumbnail' + ext)
    print '-> %s' % output

    args = ['convert', full_path,
            '-thumbnail', 'x%d' % max_height, '-unsharp', '0x.5',
            output]

    print 'Calling:', ' '.join(args)
    subprocess.check_call(args)

# main
def main():
    parser = argparse.ArgumentParser(
        description='Trims all pngs and creates thumbnails')
    parser.add_argument('directory')
    parser.add_argument('--max_height', type=int, default=100)

    args = parser.parse_args()
    print 'directory:', args.directory
    r = raw_input('proceed (y/n)? ')
    if r.lower() != 'y':
        return

    all_ = all_pngs(args.directory)
    map(trim_image, all_)
    map(lambda f: make_thumbnail(f, args.max_height), all_)

if __name__ == '__main__':
    main()
    
