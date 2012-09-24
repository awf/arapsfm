# extract_frames.py

# Imports
import os
import subprocess
import argparse
import numpy as np
from itertools import *

# Time
class Time(object):
    def __init__(self, t):
        self.t = t

    @classmethod
    def from_ms(cls, t):
        return cls(t)

    @classmethod
    def from_string(cls, t_str):
        h, m, s = t_str.split(':')
        s, mm = s.split('.')

        s = 1000 * int(s)
        m = 60 * 1000 * int(m)
        h = 60 * 60 * 1000 * int(h)
        return cls(h + m + s + int(mm))

    def as_string(self):
        t = np.around(self.t).astype(int)
        s, mm = divmod(t, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        return '%02d:%02d:%02d.%03d' % (h, m, s, mm)
        
    def as_ms(self):
        return self.t

    def __repr__(self):
        return 'Time(%d)' % self.t

    def add(self, del_t):
        return Time.from_ms(self.t + del_t)

# extract_frames
def extract_frames(info, time, num_frames, output_stem):
    base_args = ['ffmpeg', '-i', info['input_filename'], '-vcodec', 'png',
                 '-vframes', '1', '-f', 'rawvideo', '-y', 
                 '-r', str(info['fps'])]

    for n in xrange(num_frames):
        output_file = '%s-%d.png' % (output_stem, n)
        offset = np.ceil((n * 1e3) / info['fps'])
        this_time = time.add(offset)

        args = base_args + ['-ss', this_time.as_string(), output_file]
        print args

        p = subprocess.Popen(args, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()

        print '(%d/%d) -> %s' % (n + 1, num_frames, output_file)

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
    #parser = argparse.ArgumentParser(description='Extract frames from a video')
    test_file = os.path.expanduser('~/MSRC_Bundle/Data/Videos/nmpY1L4-e_g_hd720.mp4')
    info = video_information(test_file)
    extract_frames(info, Time.from_string('00:00:00.0'), 4, 'nmpY1L4-e_g')

# test_Time
def test_Time():
    t = Time.from_string('01:02:03.500')
    print t.as_string()
    print t
    print t.as_ms()

    t = Time.from_string('00:00:00.0')
    for i in xrange(25):
        print t.as_string()
        t = t.add(1e3 / 24)

if __name__ == '__main__':
    main()
    # test_Time()
    
