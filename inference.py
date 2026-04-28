import argparse
import datetime
import json
import csv
import random
import time
import numpy as np
import os, sys
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from torch.utils.data import DataLoader
from datasets import build_dataset
# from models.upicker.upicker import build_upicker
from util import box_ops
# from util.box_utils import nms
# from torchvision.ops.boxes import nms
import util.misc as utils
from util.default_args import set_model_defaults, get_args_parser
from util.visualizer import COCOVisualizer
from util.slconfig import SLConfig, DictAction
# from cryoEM.box_clean import clean_edge_boxes
from cryoEM.read_image import image_read

CLASSES = ['particle','N/O']
COLORS = ['red']

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def nms(bounding_boxes, confidence_scores, nms_threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_scores)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_scores[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < nms_threshold)
        order = order[left]

    picked_boxes = np.array(picked_boxes).squeeze()
    picked_score = np.array(picked_score)

    return picked_boxes, picked_score


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def save_boxes(boxes, scores, img_file, im_h, out_imgname):


    # save box files
    write_name = args.output_dir + img_file[:-4] + out_imgname + '.star'
    # write_box(write_name, boxes, write_star=True)
    with open(write_name, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        boxwriter.writerow([])
        boxwriter.writerow(["data_"])
        boxwriter.writerow([])
        boxwriter.writerow(["loop_"])
        boxwriter.writerow(["_rlnCoordinateX #1 "])
        boxwriter.writerow(["_rlnCoordinateY #2 "])
        boxwriter.writerow(["_rlnClassNumber #3 "])
        boxwriter.writerow(["_rlnAnglePsi #4"])
        boxwriter.writerow(["_rlnScore #5"])
        print(f'there are {len(boxes)} boxes.')
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                boxwriter.writerow([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0, 0.0, scores[i]])



def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)
    print('device: ', device)

    # fix the seed for reproducibility
    if args.random_seed:
        args.seed = np.random.randint(0, 1000000)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")


    # if args.model == 'upicker':
    #     print('\n[args by parser:]', args)
    #     cfg = SLConfig.fromfile(args.config_file)
    #     print('\n[upicker args by file:]', cfg)
    #
    #     cfg_dict = cfg._cfg_dict.to_dict()
    #     for k,v in cfg_dict.items():
    #         setattr(args, k, v)
    #
    #     if args.options is not None:
    #         cfg.merge_from_dict(args.options)
    #     print('args after merge:\n', args)
    #
    #     from models.upicker import upicker
    #     model, criterion, postprocessors = upicker.build_upicker(args)
    # else:
    #     model, criterion, postprocessors = build_dqdetr(args)
    # if args.model == 'dqdetr':
    #     print('\n[args by parser:]', args)
    #     from util.slconfig import SLConfig, DictAction
    #     cfg = SLConfig.fromfile(args.config_file)
    #     print('\n[dqdetr args by file:]', cfg)
    #
    #     cfg_dict = cfg._cfg_dict.to_dict()
    #     for k, v in cfg_dict.items():
    #         setattr(args, k, v)
    #
    #     if args.options is not None:
    #         cfg.merge_from_dict(args.options)
    #     print('args after merge:\n', args)
    #
    #     from models.dqdetr import dqdetr
    #     model, criterion, postprocessors = dqdetr.build_dqdetr(args)
    # else:
    #     model, criterion, postprocessors = build_model(args)
    if args.model == 'msdetr':
        print('\n[args by parser:]', args)
        from util.slconfig import SLConfig, DictAction
        cfg = SLConfig.fromfile(args.config_file)
        print('\n[msdetr args by file:]', cfg)

        cfg_dict = cfg._cfg_dict.to_dict()
        for k, v in cfg_dict.items():
            setattr(args, k, v)

        if args.options is not None:
            cfg.merge_from_dict(args.options)
        print('args after merge:\n', args)

        from models.msdetr import build_model
        model, criterion, postprocessors = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    # model.load_state_dict(checkpoint['model'], strict=False)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # 计算模型大小和参数量
    # total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # model_size_MB = total_size / (1024 ** 2)
    # print(f"\nModel size: {model_size_MB:.2f} MB")

    # total_params = sum(p.numel() for p in model.parameters())
    # if total_params >= 1e9:
    #     total_params_str = f"{total_params / 1e9:.2f}B"
    # elif total_params >= 1e6:
    #     total_params_str = f"{total_params / 1e6:.2f}M"
    # else:
    #     total_params_str = f"{total_params}"
    # print(f"\nTotal parameters: {total_params_str}")


    dataset_val = build_dataset(image_set='val', args=args)
    data_dir = os.path.join('/root/autodl-tmp/'+args.dataset_file)
    print("data_dir:", data_dir)
    coco = COCO(os.path.join(data_dir,'annotations/instances_val.json'))

    for image, targets in dataset_val:
        start1 = time.time()
        img_file = coco.loadImgs(int(targets['image_id'].cpu()))[0]['file_name']
        print(img_file)

        output = model(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0,1.0]]).cuda())[0]
        scores = output['scores']
        boxes = output['boxes']



        # select_mask = scores > args.score_threshold
        # boxes = boxes[select_mask]
        # scores = scores[select_mask]

        # convert boxes from [0; 1] to image scales
        source_img = Image.open(os.path.join(data_dir,'val', img_file)).convert("RGB")
        print('image shape: ', source_img.size)
        im_h, im_w = source_img.size
        target_sizes = torch.tensor([[im_w, im_h]])
        target_sizes = target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        boxes = boxes.cpu().detach().numpy()
        # boxes = np.squeeze(boxes, axis=1)
        boxes = boxes.squeeze(0)
        scores = scores.cpu().detach().numpy()
        if len(boxes) > 1:
            boxes, scores = nms(boxes, scores, nms_threshold=args.iou_threshold)

        if args.mask == True:
            mask_name = './data/'+args.dataset_file + '/micrographs/mask/' + img_file[:-4] + '_mask.jpg'
            mask = image_read(mask_name)
            from cryoEM.box_clean import delete_box_in_mask
            boxes_cleaned, saved_scores, delete_boxes = delete_box_in_mask(mask, boxes, scores, threshold=0.1)
            boxes_scores = []
            # save_boxes(boxes, scores, img_file, im_h, out_imgname='')
            save_boxes(boxes_cleaned, saved_scores, img_file, im_h, out_imgname='')
        else:
            print('no post-processing.....')
            save_boxes(boxes, scores, img_file, im_h, out_imgname='')
            
        out_imgname = args.output_dir + img_file[:-4] + '.jpg'
        draw = ImageDraw.Draw(source_img)
        i = 0

        for xmin, ymin, xmax, ymax in boxes:
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 0, 0), width=4)
            i += 1
        end1 = time.time()
        print(f"[INFO] {end1 - start1} time: with {img_file} done!!!")
        source_img.save(out_imgname, "JPEG")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('UPicker inference script', parents=[get_args_parser()])
    parser.add_argument('--mask', default=False, help="If filter predicted boxes with mask.")
    args = parser.parse_args()
    set_model_defaults(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)