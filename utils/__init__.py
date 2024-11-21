import os.path
import scipy.io as sio
# from deprecated.dataset import generate_dataset_from_tfrecords as generate_dataset
from .dataset import get_masks
from .image_ops import imread as imread_zh_cn
from .image_ops import imwrite as imwrite_zh_cn
from .image_ops import extract_data_from_indexes


# def get_masks():
#     p = os.path.dirname(__file__)
#     p = os.path.join(p, '../srcs/dataset/mask_30channels_v2.mat')
#     m, ms = masks(p)
#     return m, ms


# def generate_dataset_from_tfrecords(filepath, image_size=256, batch_size=32, shuffle=True):
#     m, ms = get_masks()
#     return generate_dataset(filepath, m, ms, image_size, batch_size, shuffle)


def get_indexes():
    p = os.path.dirname(__file__)
    p = os.path.join(p, '../preprocessing/indexes_v1.mat')
    mat = sio.loadmat(p)
    return mat

DEFAULT_MASKS = get_masks()
DEFAULT_INDEX = get_indexes()

def extract_data(image, idxs=None):
    if idxs is None:
        idxs = DEFAULT_INDEX
    return extract_data_from_indexes(image, idxs)
