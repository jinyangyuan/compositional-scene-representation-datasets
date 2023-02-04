import argparse
import json
import numpy as np
import os
from imageio import imread
from common import load_config, save_dataset


def compute_obj_score(occlude_list, score_list, idx):
    if score_list[idx] is None:
        score_list[idx] = 1
        for sub_idx in occlude_list[idx]:
            assert sub_idx >= 0
            compute_obj_score(occlude_list, score_list, sub_idx)
            score_list[idx] += score_list[sub_idx]
    return


def compute_values(shp_list, max_objects, th=0.5):
    masks_obj = np.stack(shp_list)
    zeros = np.zeros([max_objects - masks_obj.shape[0], *masks_obj.shape[1:]])
    ones = np.ones([1, *masks_obj.shape[1:]])
    masks = np.concatenate([masks_obj, zeros, ones])
    part_cumprod = np.concatenate([
        np.ones((1, *masks.shape[1:])),
        np.cumprod(1 - masks[:-1], axis=0),
    ], axis=0)
    coefs = masks * part_cumprod
    segment = np.argmax(coefs, 0).astype(np.uint8)
    overlap = ((masks >= th).sum(0) - 1).astype(np.uint8)
    masks = (masks * 255).astype(np.uint8)[..., None]
    return segment, overlap, masks


def generate_data(folder, max_objects):
    with open(os.path.join(folder, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    num_objects = metadata['num_objects']
    image = imread(os.path.join(folder, 'image.png'))
    segment = imread(os.path.join(folder, 'segmentation.png'))
    shp_list = [imread(os.path.join(folder, 'mask_{}.png'.format(idx))).astype(bool) for idx in range(num_objects)]
    occlude_list = [[val for val in np.unique(segment[shp] - 1) if val != idx] for idx, shp in enumerate(shp_list)]
    score_list = [None for _ in range(num_objects)]
    for idx in range(num_objects):
        compute_obj_score(occlude_list, score_list, idx)
    shp_score_list = sorted(zip(shp_list, score_list), key=lambda x: x[1])
    shp_list = [val[0].astype(float) for val in shp_score_list]
    segment, overlap, masks = compute_values(shp_list, max_objects)
    data = {'image': image, 'segment': segment, 'overlap': overlap, 'masks': masks}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_config')
    parser.add_argument('--folder_train')
    parser.add_argument('--folder_general')
    parser.add_argument('--folder_out')
    config = load_config(parser)
    sub_folders_train = sorted(os.listdir(config['folder_train']))
    sub_folders_general = sorted(os.listdir(config['folder_general']))
    folders_all = {
        'train': (config['folder_train'], sub_folders_train[:config['split_train']]),
        'valid': (config['folder_train'], sub_folders_train[config['split_train']:config['split_valid']]),
        'test': (config['folder_train'], sub_folders_train[config['split_valid']:]),
        'general': (config['folder_general'], sub_folders_general),
    }
    folders_all = {key: [os.path.join(val[0], sub_val) for sub_val in val[1]] for key, val in folders_all.items()}
    max_objects_all = {
        'train': config['max_objects_train'],
        'valid': config['max_objects_train'],
        'test': config['max_objects_train'],
        'general': config['max_objects_general'],
    }
    datasets = {}
    for phase, folders in folders_all.items():
        data_list = []
        for folder in folders:
            data = generate_data(folder, max_objects_all[phase])
            data_list.append(data)
        datasets[phase] = data_list
    save_dataset(config, datasets)
    return


if __name__ == '__main__':
    main()
