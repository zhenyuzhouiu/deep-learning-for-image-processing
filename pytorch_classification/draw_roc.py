import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.interpolate, scipy.optimize


def draw_roc(tar, far, eer, out_dir, label):
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    lines = plt.plot(np.log10(far + 1e-12), tar, label='ROC')
    plt.setp(lines, 'color', 'red', 'linewidth', 3)

    plt.grid(True)
    plt.xlabel(r'False Accept Rate', fontsize=18)
    plt.ylabel(r'Genuine Accept Rate', fontsize=18)

    plt.xlim(xmin=max([min(np.log10(far + 1e-12)), -5]))
    plt.xlim(xmax=0)
    plt.ylim(ymax=1)
    plt.ylim(ymin=0.4)
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)
    plt.xticks(np.array([-4, -2, 0]), ['$10^{-4}$', '$10^{-2}$', '$10^{0}$'], fontsize=16)
    plt.yticks(np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), fontsize=16)

    plt.legend(labels=[label + "; EER: %.2f%%" % (eer * 100)], loc='lower right', shadow=False, prop={'size': 16})
    dst = os.path.join(out_dir, "roc.pdf")
    plt.savefig(dst, bbox_inches='tight')
    return 0


if __name__ == "__main__":
    # get the source far and tar
    far = np.load(
        "/home/zhenyuzhou/Desktop/Project_Code/finger_knuckle_videos/GNNDiffCorr/checkpoint/09-28-20-32-51/ROC-HD-R3/far.npy")
    tar = np.load(
        "/home/zhenyuzhou/Desktop/Project_Code/finger_knuckle_videos/GNNDiffCorr/checkpoint/09-28-20-32-51/ROC-HD-R3/tar.npy")
    out_dir = "/home/zhenyuzhou/Desktop/Project_Code/finger_knuckle_videos/GNNDiffCorr/checkpoint/09-28-20-32-51/ROC-HD-R3/"

    # using scipy to get more accuracy EER
    x = np.linspace(0, 1, far.shape[0])
    interp_far = scipy.interpolate.InterpolatedUnivariateSpline(x, far)
    interp_tar = scipy.interpolate.InterpolatedUnivariateSpline(x, tar)
    new_x = np.linspace(0, 1, 1000)
    new_far = interp_far(new_x)
    new_tar = interp_tar(new_x)
    eer_init = new_x[np.argwhere(np.diff(np.sign(new_far - (1 - new_tar))) != 0)]


    def difference(x):
        return np.abs(interp_far(x) - (1 - interp_tar(x)))


    x_at_crossing = scipy.optimize.fsolve(difference, x0=eer_init)
    eer = interp_far(x_at_crossing)

    draw_roc(tar, far, eer, out_dir, label="ResNet")
