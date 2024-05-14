import contextlib

import numpy as np
import torch
import torch.nn.functional as F

import scipy


@contextlib.contextmanager
def np_local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    assert scale_factor > 1
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out

def entropy_from_lbl(lbl):
    flat_array = lbl.flatten()
    class_imp = []
    for i in [0, 1, 2, 255]:
        class_imp.append(np.sum(flat_array == i))
    return scipy.stats.entropy(class_imp, base = 2)

def compute_entropy_matrix(img):
    size = 1024
    if len(img.shape) == 3:
        img = img.squeeze()
    entropy_matrix = np.zeros((int(img.shape[0]/size), int(img.shape[1]/size)))
    for i in range(int(img.shape[0]/size)):
        for j in range(int(img.shape[1]/size)):
            patch = img[i*size:(i+1)*size, j*size:(j+1)*size]
            entropy_matrix[i, j] = entropy_from_lbl(patch)
    return entropy_matrix

def get_weighted_random_idxs(matrix):
    flat_matrix = matrix.flatten()
    tot = np.sum(flat_matrix)
    if tot == 0:
        p = np.ones(len(flat_matrix))/len(flat_matrix)
    else:
        p = flat_matrix/tot
    flat_index = np.random.choice(len(flat_matrix), p=p)
    # Convert the flat index back to a 2D index
    return np.unravel_index(flat_index, matrix.shape)