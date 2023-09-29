import os.path
import torch
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from my_dataset import MyDataSetTest
from model import efficientnetv2_m as create_model

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def data(probe_subject, data_path, image_size, batch_size, num_workers):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = MyDataSetTest(probe_subject=probe_subject,
                                 data_path=data_path,
                                 image_size=image_size,
                                 protocol="all",  # all or one
                                 transform=transform)
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
    probe_feature = out_feature[0:probe_sample, :]

    # calculate matching scores with cosine similarity
    probe_feature = probe_feature.data.cpu().numpy()
    out_feature = out_feature.data.cpu().numpy()
    num = np.dot(probe_feature, np.array(out_feature).T)  # [probe_sample, n_sample]
    norm = np.linalg.norm(probe_feature, axis=1).reshape(-1, 1) * np.linalg.norm(out_feature, axis=1)
    cos_similarity = num / norm
    cos_similarity[np.isneginf(cos_similarity)] = 0
    cos_similarity = 0.5 + 0.5 * cos_similarity  # range from [0, 1]

    g_scores = cos_similarity[:, :probe_sample][np.triu_indices_from(cos_similarity[:, :probe_sample], k=1)].reshape(-1)
    i_scores = cos_similarity[:, probe_sample:].reshape(-1)

    return g_scores, i_scores


def tar_far_n(g_socres, i_scores, step=500):
    # from the difference correspondence graph network DCGNet, the similarity score
    tar, far = [], []
    n_g, n_i = g_socres.shape[0], i_scores.shape[0]
    threshod = np.linspace(1, 0, step)
    for t in threshod:
        tar.append(np.sum(np.where(g_socres >= t, True, False)))
        far.append(np.sum(np.where(i_scores >= t, True, False)))
    tar, far = np.stack(tar).reshape(1, -1), np.stack(far).reshape(1, -1)

    return tar, far


def draw_roc(tar, far, eer, out_dir):
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    plt.grid(True)
    plt.xlabel(r'False Accept Rate', fontsize=18)
    plt.ylabel(r'Genuine Accept Rate', fontsize=18)
    plt.xlim(xmin=max([min(np.log(far + 1e-12)), -5]))
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

    lines = plt.plot(np.log(far + 1e-12), tar, label='ROC')
    plt.setp(lines, 'color', 'red', 'linewidth', 3)
    plt.legend(labels=['DCGNet; EER: %.2f%%' % (eer * 100)], loc='lower right', shadow=False, prop={'size': 16})
    dst = os.path.join(out_dir, "roc.pdf")
    plt.savefig(dst, bbox_inches='tight')
    return 0


@torch.no_grad()
def main(args, out_dir):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load model
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    model = load(model, os.path.join(args.finetuning, "best.pth"), device, if_train=False)
    print("Loaded pretrained ViT model")

    # index
    tar, far = None, None
    len_g, len_i = [], []

    # dataset
    subject_list = os.listdir(args.data_path)
    subject_list.sort()
    for subject in subject_list:
        test_loader, probe_sample, gallery_sample = data(subject, args.data_path, args.image_size, args.batch_size, args.num_workers)
        g_scores_e, i_scores_e = matching_scores(args, model, test_loader, probe_sample, device)
        if g_scores_e is not None and i_scores_e is not None:
            tar_e, far_e = tar_far_n(g_scores_e, i_scores_e)  # [1, 500]
            tar = tar_e if tar is None else np.concatenate((tar, tar_e), axis=0)
            far = far_e if far is None else np.concatenate((far, far_e), axis=0)
            len_g.append(g_scores_e.shape[0])
            len_i.append(i_scores_e.shape[0])

    len_g, len_i = np.array(len_g), np.array(len_i)
    tar, far = np.sum(tar, axis=0) / np.sum(len_g), np.sum(far, axis=0) / np.sum(len_i)
    np.save(os.path.join(out_dir, 'tar.npy'), tar)
    np.save(os.path.join(out_dir, 'far.npy'), far)
    # equal error rate: false acceptance rate = false rejection rate
    # far from 0 to 1
    # tar from 0 to 1, frr = 1- tar from 1 to 0
    eer_index = np.argmax(np.greater_equal(far, 1 - tar))
    eer = (far[eer_index - 1] + far[eer_index + 1]) / 2
    draw_roc(tar, far, eer, out_dir)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default="/mnt/Data/Finger-Knuckle-Database/HD/YOLOv5_Segment/R3",
                        help='the data source path')
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 480],
                        help='Resize the input image before running inference to the exact dimensions (w, h)')
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=32)
    parser.add_argument("--num_workers", type=int, dest="num_workers", default=8)
    parser.add_argument("--device", type=str, dest="device", default="cuda:1",
                        help="cuda device 0 or 1, or cpu")
    parser.add_argument("--num_classes", type=int, dest="num_classes", default=1424)
    parser.add_argument("--finetuning", type=str, dest="finetuning",
                        default="./weights/")
    args = parser.parse_args()

    out_dir = os.path.join(args.finetuning, 'ROC-HD-R3')

    print("[*] Target ROC Output Path: {}".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(args, out_dir)
