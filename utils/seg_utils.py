import numpy as np
from PIL import Image
import cv2

def change_seg(arr_seg):
    color_dict = {
        (0, 0, 255): 3,  # blue
        (0, 255, 0): 2,  # green
        (0, 0, 0): 0,  # black
        (255, 255, 255): 1,  # white
        (255, 0, 0): 4,  # red
        (255, 255, 0): 5,  # yellow
        (128, 128, 128): 6,  # grey
        (0, 255, 255): 7,  # lightblue
        (255, 0, 255): 8  # purple
    }  
    arr_seg = np.asarray(arr_seg)
    new_seg = np.zeros(arr_seg.shape[:-1])
    for x in range(arr_seg.shape[0]):
        for y in range(arr_seg.shape[1]):
            if tuple(arr_seg[x, y, :]) in color_dict:
                new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
            else:
                min_dist_index = 0
                min_dist = 99999
                for key in color_dict:
                    dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = color_dict[key]
                    elif dist == min_dist:
                        try:
                            min_dist_index = new_seg[x, y-1, :]
                        except Exception:
                            pass
                new_seg[x, y] = min_dist_index
    return new_seg.astype(np.uint8)


def compute_label_info(content_segment, style_segment):
    if not content_segment.size or not style_segment.size:
        return None, None
    max_label = np.max(content_segment) + 1
    label_set = np.unique(content_segment)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        content_mask = np.where(content_segment.reshape(content_segment.shape[0] * content_segment.shape[1]) == l)
        style_mask = np.where(style_segment.reshape(style_segment.shape[0] * style_segment.shape[1]) == l)

        c_size = content_mask[0].size
        s_size = style_mask[0].size
        if c_size > 150 and s_size > 150 and c_size / s_size < 100 and s_size / c_size < 100:
            label_indicator[l] = True
        else:
            label_indicator[l] = False
    return label_set, label_indicator


def read_segmentations(c_seg_path, s_seg_path):
    c_seg = change_seg(Image.open(c_seg_path))
    s_seg = change_seg(Image.open(s_seg_path))
    return c_seg, s_seg
    