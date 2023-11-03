import torch
import cv2
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


def load_image(image_path, image_size, rgb=True):
    """

    Args:
        image_path:
        image_size: w x h
        rgb: Ture: rgb; False: gray

    Returns:
        dst_img: [h, w, c]

    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if rgb else cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    h, w, c = image.shape
    r = h / w
    dst_w, dst_h = image_size
    dst_r = dst_h / dst_w
    if r > dst_r:  # crop h
        crop_h = h - dst_r * w
        image = image[int(crop_h / 2):int(h - crop_h / 2), :, :]
    else:
        crop_w = w - h / dst_r
        image = image[:, int(crop_w / 2):int(w - crop_w / 2), :]

    dst_image = cv2.resize(image, dsize=(dst_w, dst_h))
    # dst_image = np.expand_dims(dst_image, -1) if dst_image.ndim == 2 else dst_image

    return dst_image


class MyDataSetTest(Dataset):
    def __init__(self, probe_subject, data_path, data2_path, image_size, protocol, transform, visited_subject=[]):
        super().__init__()
        """
        the subject and sample of data_path and data2_path should be same
        """
        self.protocol = protocol
        self.probe_subject = probe_subject
        self.data_path = data_path
        self.data2_path = data2_path
        self.image_size = image_size
        self.transform = transform
        self.visited_subject = visited_subject

        self.probe_sample, self.list_probe, self.gallery_sample, self.list_gallery = self.data_info()

    def __len__(self):
        n_sample = self.probe_sample + self.gallery_sample
        return n_sample

    def __getitem__(self, item):
        """
        Firstly, extract the probe sample
        """
        if item < self.probe_sample:
            image = load_image(self.list_probe[item], image_size=self.image_size, rgb=True)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            label = 1.0

        else:
            image = load_image(self.list_gallery[item-self.probe_sample], image_size=self.image_size, rgb=True)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            label = 0.0

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def data_info(self):
        probe_sample_n, gallery_sample_n = 0, 0
        list_probe, list_gallery = [], []
        if self.protocol == "one_session":
            list_subject = os.listdir(self.data_path)
            list_subject.sort()
            # delete visited subject
            for i in range(len(self.visited_subject)):
                del list_subject[self.visited_subject[i]]
            for subject in list_subject:
                list_file = os.listdir(os.path.join(self.data_path, subject))
                list_file.sort()
                if subject == self.probe_subject:
                    probe_sample_n += len(list_file)
                    for file in list_file:
                        list_probe.append(os.path.join(self.data_path, subject, file))
                else:
                    gallery_sample_n += len(list_file)
                    for file in list_file:
                        list_gallery.append(os.path.join(self.data_path, subject, file))
        elif self.protocol == "two_session":
            list_subject = os.listdir(self.data_path)
            list_subject.sort()
            # delete visited subject
            for i in range(len(self.visited_subject)):
                del list_subject[self.visited_subject[i]]
            for subject in list_subject:
                list_file = os.listdir(os.path.join(self.data_path, subject))
                list_file.sort()

                if subject == self.probe_subject:
                    probe_sample_n += len(list_file)
                    gallery_sample_n += len(list_file)
                    for file in list_file:
                        list_probe.append(os.path.join(self.data_path, subject, file))
                        list_gallery.append(os.path.join(self.data2_path, subject, file))
                else:
                    gallery_sample_n += len(list_file)
                    for file in list_file:
                        list_gallery.append(os.path.join(self.data2_path, subject, file))
        else:
            raise f"please give a right protocol"

        return probe_sample_n, list_probe, gallery_sample_n, list_gallery
