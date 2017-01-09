# move_segmentations.py

# Imports
import os
import argparse
import shutil

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    input_files = filter(lambda f: f.endswith('.png'), os.listdir(args.input))
    sorted_input_files = sorted(
        input_files,
        key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]))

    # decimate
    sorted_input_files = sorted_input_files[::2]

    input_full_paths = map(
        lambda f: os.path.join(args.input, f), 
        sorted_input_files)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    output_paths = map(
        lambda i: os.path.join(args.output, '%d.png' % i),
        xrange(len(input_full_paths)))

    map(shutil.copy, input_full_paths, output_paths)

if __name__ == '__main__':
    main()
