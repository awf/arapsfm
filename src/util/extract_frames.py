# extract_frames.py

# Imports
import os
import subprocess
import argparse
from itertools import *

# extract_frames
def extract_frames(input_filename, times, output_dir, codec='png', ext=None):
    if ext is None:
        ext = codec

    print 'Input file:', input_filename
    print 'Times:', times
    print 'Output directory:', output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_args = ['ffmpeg', '-i', input_filename, 
                 '-vcodec', codec,
                 '-vframes', '1',
                 '-y']

    for i, time in enumerate(times):
        args = base_args + ['-ss', time, 
                            os.path.join(output_dir, '%d.%s' % (i, codec))]
        print 'Command:', ' '.join(args)

        p = subprocess.Popen(args, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()

    with open(os.path.join(output_dir, 'frames.txt'), 'w') as fp:
        fp.write(str(times))

# main
def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('input', type=str, help='input video file')
    parser.add_argument('output', type=str, help='output string stem')
    parser.add_argument('times', type=str, nargs='+', 
                        help='format hh:mm:ss.[xxx]')
    parser.add_argument('--codec', type=str, help='codec', default='png')
    parser.add_argument('--ext', type=str, help='extension', default=None)

    args = parser.parse_args()

    extract_frames(args.input, args.times, args.output,
                   codec=args.codec, 
                   ext=args.ext)

if __name__ == '__main__':
    main()
