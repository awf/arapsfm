# extract_frames.py

# Imports
import os
import subprocess
import argparse
from itertools import *

# extract_frames
def extract_frames(info, time_string, num_frames, output_stem):
    args = ['ffmpeg', '-i', info['input_filename'], 
            '-ss', time_string,
            '-vcodec', 'png',
            '-vframes', str(num_frames), 
            '-y', output_stem + r'-%d.png']

    print 'Command:', ' '.join(args)

    p = subprocess.Popen(args, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()

# video_information
def video_information(input_filename):
    p = subprocess.Popen(['ffmpeg', '-i', input_filename],
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    valid_lines = dropwhile(lambda l: 'Input' not in l, stderr.splitlines())

    info = {'input_filename' : input_filename}

    for l in valid_lines:
        if 'Video:' in l:
            l = l.split('Video:')[1].split(',')
            info['fps'] = int(l[4].split()[0])
            info['width'], info['height'] = map(int, l[2].split('x'))

    return info

# main
def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('input', type=str, help='input video file')
    parser.add_argument('time', type=str, help='format hh:mm:ss.[xxx]')
    parser.add_argument('output', type=str, help='output string stem')
    parser.add_argument('--n', type=int, help='number of frames to extract',
                        default=1)

    args = parser.parse_args()
    info = video_information(args.input)

    print 'Input:', info['input_filename']
    print 'Width:', info['width']
    print 'Height:', info['height']
    print 'FPS:', info['fps']

    print 'Source time:', args.time
    print 'Number of frames:', args.n
    print 'Output stem:', args.output

    extract_frames(info, args.time, args.n, args.output)

if __name__ == '__main__':
    main()
    
