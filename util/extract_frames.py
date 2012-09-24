# extract_frames.py

# Imports
import os
import subprocess
import argparse
from itertools import *

# extract_frames
def extract_frames(input_filename, times, output_stem):
    base_args = ['ffmpeg', '-i', input_filename, 
                 '-vcodec', 'png',
                 '-vframes', '1',
                 '-y']

    for i, time in enumerate(times):
        args = base_args + ['-ss', time, output_stem + '-%d.png' % i]
        print 'Command:', ' '.join(args)

        p = subprocess.Popen(args, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()

# main
def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('input', type=str, help='input video file')
    parser.add_argument('output', type=str, help='output string stem')
    parser.add_argument('times', type=str, nargs='+', 
                        help='format hh:mm:ss.[xxx]')
    args = parser.parse_args()

    extract_frames(args.input, args.times, args.output)

if __name__ == '__main__':
    main()
