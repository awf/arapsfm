# get_indices

# Imports
import os, argparse

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    files = os.listdir(args.input_dir)
    file_roots = map(lambda f: os.path.splitext(f)[0], files)
    indices = map(lambda f: int(f.split('_')[-1]), file_roots)
    sorted_indices = sorted(indices)

    indices_string = '"[%s]"' % (','.join('%d' % i for i in sorted_indices), )
    if args.output is None:
        print indices_string
    else:
        with open(args.output, 'w') as fp:
            fp.write(indices_string)
            
if __name__ == '__main__':
    main()

