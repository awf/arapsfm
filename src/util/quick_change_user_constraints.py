# quick_change_user_constraints.py

# Imports
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# inst_num
def inst_num(filename):
    root = os.path.splitext(filename)[0]
    try:
        return int(root.split('_')[-1])
    except ValueError:
        return None

# PointMover
class PointMover(object):
    def __init__(self, line, labels=[]):
        self.x, self.y = line.get_data()
        self.line = line
        self.labels = labels

        self.moving_index = None

        self.ids = [
            line.figure.canvas.mpl_connect('button_press_event', 
                                           self.button_press_event),
            line.figure.canvas.mpl_connect('button_release_event', 
                                           self.button_release_event)
        ]

        self.update()

    def update(self):
        self.line.set_data(self.x, self.y)

        del self.line.axes.texts[:]

        for i, label in enumerate(self.labels):
            self.line.axes.text(self.x[i], self.y[i], label)

        self.line.figure.canvas.draw()
        
    def get_closest_index(self, x, y):
        X = np.c_[self.x, self.y]
        r = X - np.r_[x, y]
        e = np.sum(r*r, axis=1)
        return np.argmin(e)

    def button_press_event(self, event):
        if event.inaxes != self.line.axes:
            return

        if event.button == 1:
            self.moving_index = self.get_closest_index(
                event.xdata, event.ydata)
            return

    def button_release_event(self, event):
        if event.inaxes != self.line.axes:
            return

        if event.button == 1:
            if self.moving_index is None:
                return

            self.x[self.moving_index] = event.xdata
            self.y[self.moving_index] = event.ydata

            self.update()

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, 
                        choices=['add_constraints',
                                 'add_constant'])
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--indices', type=str, default='[]')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--image_stem', type=str, default=None)
    parser.add_argument('--point_ids', type=str, default='None')
    parser.add_argument('--constant', type=str, default='None')

    args = parser.parse_args()

    for key in ['point_ids', 'constant', 'indices']:
        setattr(args, key, eval(getattr(args, key)))

    pprint(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    make_input_path = lambda f: os.path.join(args.input_dir, f)
    make_output_path = lambda f: os.path.join(args.output_dir, f)
    load_image = lambda i: plt.imread(args.image_stem % i)

    input_files = filter(lambda f: os.path.isfile(make_input_path(f)), 
                         os.listdir(args.input_dir))

    valid_files = filter(lambda f: inst_num(f) is not None, input_files)
    sorted_files = sorted(valid_files, key=inst_num)

    if args.indices is None:
        sorted_files = sorted_files[::args.step]
    else:
        sorted_files = filter(lambda f: inst_num(f) in args.indices,
                              sorted_files)
    print '#:', len(sorted_files)

    def requires(*arg_keys):
        for key in arg_keys:
            if getattr(args, key) is None:
                raise ValueError('argument "%s" is required' % key)
        
    def setup_axis(ax, im, x=None, y=None, labels=None, *args, **kwargs):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)

        if None in (x, y):
            return

        line, = ax.plot(x, y, 'ro', *args, **kwargs)

        if labels is not None:
            for i, label in enumerate(labels):
                ax.text(x[i], y[i], label)

        return line

    if args.action == 'add_constraints':
        requires('image_stem', 'point_ids')

        Q = None
        labels = map(str, args.point_ids)

        for i, input_file in enumerate(sorted_files):
            n = inst_num(input_file)
            print '[%d]: %s (%d)' % (i, input_file, n)

            f = plt.figure(frameon=False)
            axs = [f.add_axes([0., 0.5, 1., 0.5]),
                   f.add_axes([0., 0., 1., 0.5])]

            if i == 0:
                im = np.zeros_like(load_image(n))
                setup_axis(axs[0], im)
                x = np.zeros(len(args.point_ids), dtype=np.float64)
                y = np.zeros(len(args.point_ids), dtype=np.float64)
            else:
                im = load_image(inst_num(sorted_files[i-1]))
                x0, y0 = np.transpose(Q)
                setup_axis(axs[0], im, x0, y0, labels=labels)
                x = x0.copy()
                y = y0.copy()

            interactor = PointMover(setup_axis(axs[1], load_image(n), x, y),
                                    labels=labels)

            plt.show()

            print ' %s' % np.around(x, decimals=2)
            print ' %s' % np.around(y, decimals=2)
            Q = np.c_[x, y]
            R = Q.copy()
            R[:,1] = im.shape[0] - R[:,1]

            # update constraints
            input_path = make_input_path(input_file)
            print ' <- %s' % input_path

            z = np.load(input_path)

            C = np.r_[z['C'], args.point_ids]
            P = np.r_['0,2', z['P'], R]

            d = dict(T=z['T'], V=z['V'])
            d['C'] = d['point_ids'] = C
            d['P'] = d['positions'] = P

            output_path = make_output_path(input_file)
            print ' -> %s' % output_path
            np.savez_compressed(output_path, **d)

    elif args.action == 'add_constant':
        requires('point_ids', 'constant')

        for i, input_file in enumerate(sorted_files):
            input_path = make_input_path(input_file)
            print ' <- %s' % input_path

            z = np.load(input_path)
            C = z['C']

            l = np.argwhere(C == args.point_ids).ravel()

            P = z['P']
            P[l] += args.constant

            d = dict(T=z['T'], V=z['V'])
            d['C'] = d['point_ids'] = C
            d['P'] = d['positions'] = P

            output_path = make_output_path(input_file)
            print ' -> %s' % output_path
            np.savez_compressed(output_path, **d)

if __name__ == '__main__':
    main()
