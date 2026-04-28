import os
import json
import shutil
import argparse
import numpy as np

from coord_io import (
    read_star_file,
    read_eman_boxfile,
    read_txt_file,
    read_csv_file
)
from preprocess import image_read, read_width_height, save_image


def get_args_parser():
    parser = argparse.ArgumentParser(
        'CryoDETR COCO dataset preparation script',
        add_help=False
    )

    parser.add_argument('--coco_path', default='./data/EMPIAR10075/', type=str)
    parser.add_argument('--images_path', default='./data/EMPIAR10075/micrographs/processed/', type=str)

    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'val', 'test'],
        help='Dataset split: train / val / test.'
    )

    parser.add_argument('--box_width', default=200, type=int)
    parser.add_argument('--bin', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ifsplit', action='store_true', help='If images are already split in patches.')

    return parser


def split_indexes_8_1_1(image_path, phase='train', seed=42):
    """
    Split images into train / val / test with a fixed 8:1:1 ratio.

    train: first 80%
    val: next 10%
    test: all remaining images

    No split file will be saved.
    """

    indexes = [
        f for f in os.listdir(image_path)
        if os.path.isfile(os.path.join(image_path, f))
        and not f.startswith('.')
        and f.lower().endswith(('.mrc', '.jpg', '.jpeg', '.png'))
    ]

    indexes = sorted(indexes)

    rng = np.random.default_rng(seed)
    indexes = list(rng.permutation(indexes))

    n_total = len(indexes)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_indexes = indexes[:n_train]
    val_indexes = indexes[n_train:n_train + n_val]
    test_indexes = indexes[n_train + n_val:]  # test uses all remaining images

    train_set = set(train_indexes)
    val_set = set(val_indexes)
    test_set = set(test_indexes)

    assert train_set.isdisjoint(val_set), 'Data leakage: train and val overlap.'
    assert train_set.isdisjoint(test_set), 'Data leakage: train and test overlap.'
    assert val_set.isdisjoint(test_set), 'Data leakage: val and test overlap.'
    assert len(train_indexes) + len(val_indexes) + len(test_indexes) == n_total

    print(f'[INFO] Total images: {n_total}')
    print(f'[INFO] Train: {len(train_indexes)}')
    print(f'[INFO] Val: {len(val_indexes)}')
    print(f'[INFO] Test: {len(test_indexes)}')

    if phase == 'train':
        return train_indexes
    elif phase == 'val':
        return val_indexes
    elif phase == 'test':
        return test_indexes
    else:
        raise ValueError(f'Unsupported phase: {phase}')


def read_annotation_boxes(root_path, stem, height, box_width=200, bin_size=1):
    boxes = []

    if bin_size == 1:
        anno_dir = os.path.join(root_path, 'annots')
    else:
        anno_dir = os.path.join(root_path, f'annots/downsample{bin_size}')

    star_path = os.path.join(anno_dir, stem + '.star')
    box_path = os.path.join(anno_dir, stem + '.box')
    csv_path = os.path.join(anno_dir, stem + '.csv')
    txt_path = os.path.join(anno_dir, stem + '.txt')

    if os.path.exists(star_path):
        print(f'[DEBUG] read star file: {star_path}')
        boxes = read_star_file(star_path, box_width=box_width)

    elif os.path.exists(box_path):
        print(f'[DEBUG] read box file: {box_path}')
        boxes = read_eman_boxfile(box_path)

    elif os.path.exists(csv_path):
        print(f'[DEBUG] read csv file: {csv_path}')
        boxes = read_csv_file(
            csv_path,
            image_height=height,
            box_width=box_width
        )

    elif os.path.exists(txt_path):
        print(f'[DEBUG] read txt file: {txt_path}')
        boxes = read_txt_file(txt_path, box_width=box_width)

    else:
        print(f'[WARNING] No annotation file found for: {stem}')

    return boxes


def copy_or_convert_image(src_img_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    index = os.path.basename(src_img_path)
    stem, ext = os.path.splitext(index)
    ext = ext.lower()

    if ext == '.mrc':
        image = image_read(src_img_path)

        if not np.issubdtype(image.dtype, np.float32):
            image = image.astype(np.float32)

        file_name = stem + '.png'
        dst_img_path = os.path.join(dst_dir, file_name)
        save_image(image, dst_img_path)

    elif ext in ['.jpg', '.jpeg', '.png']:
        file_name = index
        dst_img_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_img_path, dst_img_path)

    else:
        raise ValueError(f'Unsupported image format: {src_img_path}')

    return file_name


def make_coco_dataset(
    root_path,
    image_path,
    box_width=200,
    phase='train',
    ifsplit=False,
    seed=42,
    bin_size=1
):
    """
    Make COCO-style dataset for CryoDETR.

    Supported phases:
        train
        val
        test
    """

    if phase not in ['train', 'val', 'test']:
        raise ValueError(f'Unsupported phase: {phase}. Use train / val / test.')

    os.makedirs(os.path.join(root_path, phase), exist_ok=True)

    dataset = {
        'categories': [],
        'images': [],
        'annotations': []
    }

    dataset['categories'].append({
        'id': 1,
        'name': 'particle',
        'supercategory': 'mark'
    })

    if not ifsplit:
        indexes = split_indexes_8_1_1(
            image_path=image_path,
            phase=phase,
            seed=seed
        )

        print(f'[INFO] There are {len(indexes)} images in {phase} set.')

        for index in indexes:
            src_img_path = os.path.join(image_path, index)
            dst_dir = os.path.join(root_path, phase)
            copy_or_convert_image(src_img_path, dst_dir)

    else:
        print('[INFO] Make dataset for already split data.')

        split_dir = os.path.join(root_path, phase)

        indexes = [
            f for f in os.listdir(split_dir)
            if os.path.isfile(os.path.join(split_dir, f))
            and not f.startswith('.')
            and f.lower().endswith(('.mrc', '.jpg', '.jpeg', '.png'))
        ]

        indexes = sorted(indexes)

        print(f'[INFO] There are {len(indexes)} images in {phase} set.')

    anno_id = 0

    for image_id, index in enumerate(indexes):
        stem, ext = os.path.splitext(index)
        ext = ext.lower()

        if ifsplit:
            img_for_size = os.path.join(root_path, phase, index)
            file_name = index
        else:
            img_for_size = os.path.join(image_path, index)
            file_name = stem + '.png' if ext == '.mrc' else index

        width, height = read_width_height(img_for_size)

        print(f'[INFO] image: {index}')
        print(f'[INFO] width: {width}, height: {height}')

        dataset['images'].append({
            'file_name': file_name,
            'id': image_id,
            'width': width,
            'height': height
        })

        boxes = read_annotation_boxes(
            root_path=root_path,
            stem=stem,
            height=height,
            box_width=box_width,
            bin_size=bin_size
        )

        for box in boxes:
            bw = int(box.w)
            bh = int(box.h)
            box_xmin = int(box.x)
            box_ymin = int(box.y)

            anno_id += 1

            dataset['annotations'].append({
                'area': bw * bh,
                'bbox': [box_xmin, box_ymin, bw, bh],
                'category_id': 1,
                'id': anno_id,
                'image_id': image_id,
                'iscrowd': 0,
                'segmentation': []
            })

    annotation_dir = os.path.join(root_path, 'annotations')
    os.makedirs(annotation_dir, exist_ok=True)

    json_name = os.path.join(annotation_dir, f'instances_{phase}.json')

    print(f'[INFO] Save COCO annotation: {json_name}')

    with open(json_name, 'w') as f:
        json.dump(dataset, f)

    print(f'[INFO] Done. Images: {len(dataset["images"])}')
    print(f'[INFO] Done. Annotations: {len(dataset["annotations"])}')


def main(args):
    print(args)

    make_coco_dataset(
        root_path=args.coco_path,
        image_path=args.images_path,
        box_width=args.box_width,
        phase=args.phase,
        ifsplit=args.ifsplit,
        seed=args.seed,
        bin_size=args.bin
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'CryoDETR COCO dataset preparation script',
        parents=[get_args_parser()]
    )

    args = parser.parse_args()
    main(args)