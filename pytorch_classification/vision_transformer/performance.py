import argparse
import os
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.optimize
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from my_dataset import MyDataSetTest
from vit_model import vit_base_patch16_224_in21k as create_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def data(probe_subject, data_path, data2_path, image_size, batch_size, num_workers, protocol, visited_subject=[]):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_dataset = MyDataSetTest(probe_subject=probe_subject,
                                 data_path=data_path,
                                 data2_path=data2_path,
                                 image_size=image_size,
                                 protocol=protocol,  # two_session or one_session
                                 transform=transform,
                                 visited_subject=visited_subject)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader, test_dataset.probe_sample, test_dataset.gallery_sample


def load(model, checkpoint_dir, device, if_train=True):
    weights_dict = torch.load(checkpoint_dir, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)
    model.to(device)
    model = model.train() if if_train else model.eval()
    return model


@torch.no_grad()
def matching_scores(args, model, test_loader, probe_sample, device):
    model = model.eval()
    for iter_id, (image, label) in tqdm(enumerate(test_loader)):
        if image is None:
            continue
        image, label = image.to(device), label.to(device)
        out = model(image)
        if out.dim() != 2:
            out = out.unsqueeze(0)
        if iter_id == 0:
            out_feature = out
        else:
            out_feature = torch.cat([out_feature, out], dim=0)

    if args.protocol == "one_session":
        probe_feature = out_feature[0:probe_sample, :]
        gallery_feature = out_feature
    elif args.protocol == "two_session":
        probe_feature = out_feature[0:probe_sample, :]
        gallery_feature = out_feature[probe_sample:, :]

    # calculate matching scores with cosine similarity
    probe_feature = probe_feature.data.cpu().numpy()
    gallery_feature = gallery_feature.data.cpu().numpy()
    num = np.dot(probe_feature, np.array(gallery_feature).T)  # [probe_sample, n_sample]
    norm = np.linalg.norm(probe_feature, axis=1).reshape(-1, 1) * np.linalg.norm(gallery_feature, axis=1)
    cos_similarity = num / norm
    cos_similarity[np.isneginf(cos_similarity)] = 0
    cos_similarity = 0.5 + 0.5 * cos_similarity  # range from [0, 1]
    
    if args.protocol == "one_session":
        # delete diagonal elements for genuine scores
        g_scores = np.ndarray.flatten(cos_similarity[:, :probe_sample])
        g_scores = np.delete(g_scores, range(0, len(g_scores), probe_sample+1), 0)  # delete diagonal elements
        i_scores = cos_similarity[:, probe_sample:].reshape(-1)
    elif args.protocol == "two_session":
        # all elements
        g_scores = cos_similarity[:, :probe_sample].reshape(-1)
        i_scores = cos_similarity[:, probe_sample:].reshape(-1)

    # # upright
    # g_scores = cos_similarity[:, :probe_sample][np.triu_indices_from(cos_similarity[:, :probe_sample], k=1)].reshape(-1)

    return g_scores, i_scores


def tar_far(g_socres, i_scores, step=1000):
    # from the difference correspondence graph network DCGNet, the similarity score
    tar, far = [], []
    n_g, n_i = g_socres.shape[0], i_scores.shape[0]
    threshod = np.linspace(1, 0, step)
    for t in threshod:
        tar.append(np.sum(np.where(g_socres >= t, True, False)) / n_g)
        far.append(np.sum(np.where(i_scores >= t, True, False)) / n_i)
    tar, far = np.stack(tar).reshape(1, -1), np.stack(far).reshape(1, -1)

    return tar.reshape(-1), far.reshape(-1)


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


@torch.no_grad()
def main(args, out_dir):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load model
    model = create_model(num_classes=args.num_classes).to(device)
    model = load(model, os.path.join(args.finetuning, "best.pth"), device, if_train=False)
    print("Loaded pretrained ViT model")

    # index
    g_scores, i_scores = None, None

    # dataset
    subject_list = os.listdir(args.data_path)
    subject_list.sort()
    visited_subject = []
    for subject in subject_list:
        test_loader, probe_sample, gallery_sample = data(subject, args.data_path, args.data2_path, args.image_size,
                                                         args.batch_size, args.num_workers, args.protocol,
                                                         visited_subject=visited_subject)
        g_scores_e, i_scores_e = matching_scores(args, model, test_loader, probe_sample, device)
        # visited_subject.append(subject)
        if g_scores_e is not None and i_scores_e is not None:
            g_scores = g_scores_e if g_scores is None else np.concatenate((g_scores, g_scores_e))
            i_scores = i_scores_e if i_scores is None else np.concatenate((i_scores, i_scores_e))

    if args.save_scores:
        np.save(os.path.join(out_dir, 'g_scores.npy'), g_scores)
        np.save(os.path.join(out_dir, 'i_scores.npy'), i_scores)
    tar, far = tar_far(g_socres=g_scores, i_scores=i_scores)

    # using scipy to get more accuracy EER
    x = np.linspace(0, 1, far.shape[0])
    interp_far = scipy.interpolate.InterpolatedUnivariateSpline(x, far)
    interp_tar = scipy.interpolate.InterpolatedUnivariateSpline(x, tar)
    eer_init = x[np.argwhere(np.diff(np.sign(far - (1 - tar))) != 0)]

    def difference(x):
        return np.abs(interp_far(x) - (1 - interp_tar(x)))

    x_at_crossing = scipy.optimize.fsolve(difference, x0=eer_init)
    eer = interp_far(x_at_crossing)

    draw_roc(tar, far, eer, out_dir, label=args.label)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default="/home/ra1/Project/ZZY/finger-knuckle-videos/FKVideo/image-10/",
                        help='the data source path')
    parser.add_argument('--data2_path', type=str,
                        default="/home/ra1/Project/ZZY/finger-knuckle-videos/FKVideo/image-10/",
                        help="the second data source path")
    parser.add_argument('--protocol', type=str, default="one_session", help="two_session or one_session")
    parser.add_argument('--image_size', type=int, nargs='+', default=[224, 224],
                        help='Resize the input image before running inference to the exact dimensions (w, h)')
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=64)
    parser.add_argument("--num_workers", type=int, dest="num_workers", default=8)
    parser.add_argument("--device", type=str, dest="device", default="cuda:0",
                        help="cuda device 0 or 1, or cpu")
    parser.add_argument("--num_classes", type=int, dest="num_classes", default=1023)
    parser.add_argument("--finetuning", type=str, dest="finetuning",
                        default="./fkvideo-weight/")
    parser.add_argument("--label", type=str, default="ViT")
    parser.add_argument("--save_scores", type=bool, default="True")
    args = parser.parse_args()

    out_dir = os.path.join(args.finetuning, 'ROC-FKVIDEO-R2-NEW')

    print("[*] Target ROC Output Path: {}".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(args, out_dir)
