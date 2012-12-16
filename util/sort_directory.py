# sort_directory.py

# Imports
import os
import argparse
from pprint import pprint

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--extension', type=str, default='.npy')
    args = parser.parse_args()

    files = filter((lambda f: os.path.isfile(os.path.join(args.input_dir, f)) 
                    and f.endswith(args.extension)),
                   os.listdir(args.input_dir))

    def key(filename):
        root, ext = os.path.splitext(filename)
        return int(root)

    arg_sort = sorted(range(len(files)), key=lambda i: key(files[i]))

    output_dir = os.path.join(args.input_dir, 'sorted')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    join = lambda *a: os.path.abspath(os.path.join(*a))

    for i, index in enumerate(arg_sort):
        os.symlink(join(args.input_dir, files[index]),
                   join(output_dir, '%d%s' % (i, args.extension)))
        
if __name__ == '__main__':
    main()
