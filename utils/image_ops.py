import cv2 as cv
import numpy as np
import scipy.io as sio


def imread(path, flags=None):
    image = cv.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if np.ndim(image) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def imwrite(path, image, flags=None):
    if np.ndim(image) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    flag, buf = cv.imencode('.' + path.split('.')[-1], image, flags)
    if not flag:
        raise ValueError('Error!')
    buf.tofile(path)


def extract_data_from_indexes(img, idxs):
    xy_idxs = idxs['xy_idxs']
    h = idxs['h'].squeeze()
    w = idxs['w'].squeeze()
    data = img[xy_idxs[:, 1], xy_idxs[:, 0]]
    return np.reshape(data, (h, w))


def find_rgb_idxs(sprange, rgb=(620, 525, 458)):
    r, g, b = rgb
    r_idx = np.argmin(np.abs(sprange - r))
    g_idx = np.argmin(np.abs(sprange - g))
    b_idx = np.argmin(np.abs(sprange - b))
    return r_idx, g_idx, b_idx


def hsi2rgb(spimg, sp_range):
    rgb = find_rgb_idxs(sp_range)
    r = np.expand_dims(spimg[:, :, rgb[0]], axis=2)
    g = np.expand_dims(spimg[:, :, rgb[1]], axis=2)
    b = np.expand_dims(spimg[:, :, rgb[2]], axis=2)
    rgbimg = np.concatenate([r, g, b], axis=2)
    rgbimg = rgbimg / float(np.max(rgbimg))
    rgbimg *= 255
    rgbimg = rgbimg.astype(np.uint8)
    return rgbimg


def xyminxymax2xywh(xmin, ymin, xmax, ymax):
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    h = ymax - ymin
    w = xmax - xmin
    return cx, cy, w, h


def xywh2xyminxymax(cx, cy, w, h):
    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2
    return xmin, ymin, xmax, ymax


def getXYZFcn(mat_path=None):
    if mat_path is None:
        mat_path = './srcs/xyzFcn_interp_d30.mat'
    xyzFcn = sio.loadmat(mat_path)
    xFcn = xyzFcn['xFcn_interp']
    yFcn = xyzFcn['yFcn_interp']
    zFcn = xyzFcn['zFcn_interp']
    xyzFcn = np.concatenate([xFcn, yFcn, zFcn], axis=1)

    return xyzFcn


def sp2xyz(sp_val, xyzFcn=None, mat_path=None):
    if xyzFcn is None:
        xyzFcn = getXYZFcn(mat_path=mat_path)

    sp_val = np.reshape(sp_val, (1, 30))
    xyz = np.matmul(sp_val, xyzFcn)

    return xyz[0]

def sp2rgb(spimg, mat_path=None):
    m, n, c = np.shape(spimg)
    xyzimg = np.zeros((m, n, 3), dtype=np.float64)
    xyzMap = getXYZFcn(mat_path=mat_path)
    for x in range(m):
        for y in range(n):
            xyzimg[x, y, :] = sp2xyz(spimg[x, y, :], xyzFcn=xyzMap)

    xyz_max = np.max(xyzimg)
    xyz_min = np.min(xyzimg)
    xyzimg = (xyzimg - xyz_min) / (xyz_max - xyz_min)
    xyzimg *= 255.
    xyzimg = xyzimg.astype(np.uint8)

    return cv.cvtColor(xyzimg, cv.COLOR_XYZ2RGB)


def read_hsi_from_matfile(path):
    pass