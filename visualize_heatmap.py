"""Plot particle-class Grad-CAM on a selected CryoDETR encoder feature scale.

Run from the repository root in the environment used for training. See HEATMAP.md.
"""

import argparse
import inspect
import json
import re
import runpy
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_config(path):
    """Read a training configuration, including this repository's _base_ files."""
    if path is None:
        return {}
    path = Path(path).resolve()
    if path.suffix == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    values = runpy.run_path(str(path))
    merged = {}
    for base in values.get('_base_', []):
        merged.update(load_config(path.parent / base))
    merged.update({key: value for key, value in values.items() if not key.startswith('_')})
    return merged


def load_model(path, config, device):
    # Training checkpoints contain argparse.Namespace, so use trusted local files.
    load_options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_options['weights_only'] = False
    checkpoint = torch.load(path, **load_options)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    saved_args = checkpoint.get('args', {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    if not saved_args and config is None:
        raise ValueError('Checkpoint has no args. Supply --config with the training configuration.')
    settings = load_config(config)
    settings.update(saved_args)

    # Recover dimensions from the actual weights, including older 20-class heads.
    settings['hidden_dim'] = state['transformer.level_embed'].shape[1]
    settings['num_feature_levels'] = state['transformer.level_embed'].shape[0]
    settings['enc_layers'] = len({int(key.split('.')[3]) for key in state
                                  if key.startswith('transformer.encoder.layers.')})
    settings['dec_layers'] = len({int(key.split('.')[3]) for key in state
                                  if key.startswith('transformer.decoder.layers.')})
    hybrid_layers = sorted({int(key.split('.')[3]) for key in state
                            if re.match(r'transformer\.encoder\.layers\.\d+\.std_self_attn\.', key)})
    if hybrid_layers != list(range(len(hybrid_layers))):
        raise ValueError('This checkpoint uses a non-prefix HAE layout; inspect its training code.')
    settings['hybrid_encoder_layers'] = len(hybrid_layers)
    settings['two_stage'] = 'transformer.enc_output.weight' in state
    settings['mixed_selection'] = settings['two_stage'] and 'query_embed.weight' in state
    settings['with_box_refine'] = any(key.startswith('transformer.decoder.bbox_embed.') for key in state)
    if 'query_embed.weight' in state:
        settings['num_queries'] = state['query_embed.weight'].shape[0]
        settings['topk_eval'] = settings['num_queries']
    class_weight = state['class_embed.0.weight']
    num_classes = class_weight.shape[0]
    required = ('backbone', 'dilation', 'position_embedding', 'nheads', 'dim_feedforward',
                'dropout', 'num_queries', 'topk_eval', 'enc_n_points', 'dec_n_points',
                'look_forward_twice', 'use_ms_detr', 'use_aux_ffn')
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError('Training settings are missing: ' + ', '.join(missing)
                         + '. Supply a --config JSON/Python file with these values.')
    settings.update(device=str(device), lr_backbone=0, masks=False, aux_loss=False,
                    pretrained_backbone=False)
    args = argparse.Namespace(**settings)

    # Use the real model classes, without importing main.py or constructing losses.
    from models.CryoDETR.backbone import build_backbone
    from models.CryoDETR.deformable_detr import DeformableDETR
    from models.CryoDETR.deformable_transformer import build_deforamble_transformer

    model = DeformableDETR(
        build_backbone(args), build_deforamble_transformer(args),
        num_classes=num_classes, num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels, aux_loss=False,
        with_box_refine=args.with_box_refine, two_stage=args.two_stage,
        mixed_selection=args.mixed_selection, use_ms_detr=args.use_ms_detr,
    )
    # A heatmap must use every saved model parameter. Architecture mismatches fail.
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    metadata = {key: getattr(args, key) for key in required}
    metadata.update(num_classes=num_classes, hidden_dim=args.hidden_dim,
                    num_feature_levels=args.num_feature_levels,
                    enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                    hybrid_encoder_layers=args.hybrid_encoder_layers,
                    std_attn_level=getattr(args, 'std_attn_level', -1),
                    two_stage=args.two_stage, mixed_selection=args.mixed_selection,
                    with_box_refine=args.with_box_refine)
    return model, metadata


def prepare_image(path, short_side, max_size, device):
    from torchvision.transforms.functional import resize
    # PIL orientation matches datasets.coco, without a second flip or enhancement.
    with Image.open(path) as source:
        original = source.convert('RGB')
    # Match datasets.transforms.resize, including its max-size rounding, without
    # importing the COCO annotation loader or legacy training-only transforms.
    width, height = original.size
    short = min(width, height)
    long = max(width, height)
    if long / short * short_side > max_size:
        short_side = int(round(max_size * short / long))
    if width < height:
        resized_hw = (int(short_side * height / width), short_side)
    else:
        resized_hw = (short_side, int(short_side * width / height))
    resized = resize(original, resized_hw)
    pixels = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    # This also keeps gradients enabled when the backbone was frozen in training.
    return original, tensor.requires_grad_(True)


def select_queries(outputs, options, image_size):
    logits = outputs['pred_logits'][0]
    if not 0 <= options.class_id < logits.shape[1]:
        raise ValueError(f'Class {options.class_id} is outside the {logits.shape[1]} saved classes.')
    class_logits = logits[:, options.class_id]
    probabilities = class_logits.detach().sigmoid()
    if options.target_box:
        # Select the best-overlapping predicted query for the same original region.
        from util.box_ops import box_cxcywh_to_xyxy, box_iou

        width, height = image_size
        x1, y1, x2, y2 = options.target_box
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('--target-box must lie within the original image.')
        target = class_logits.new_tensor(options.target_box) / class_logits.new_tensor(
            [width, height, width, height])
        overlaps, _ = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][0]), target[None])
        if overlaps.max().item() <= 0:
            raise ValueError('No predicted box overlaps --target-box. Choose a covered region.')
        indices = overlaps[:, 0].argmax().reshape(1)
    else:
        indices = torch.where(probabilities >= options.score_threshold)[0]
        if indices.numel() == 0:
            raise ValueError('No particle query passed --score-threshold. '
                             'Inspect class-id and scores or explicitly lower the threshold.')
        order = probabilities[indices].argsort(descending=True)
        indices = indices[order[:options.top_k]]
    # Logits avoid the saturation of sigmoid. Query selection is detached above.
    target = class_logits[indices].sum()
    metadata = {'query_indices': indices.detach().cpu().tolist(),
                'query_probabilities': probabilities[indices].cpu().tolist(),
                'query_boxes_cxcywh': outputs['pred_boxes'][0, indices].detach().cpu().tolist()}
    return target, metadata


def encoder_gradcam(model, image_tensor, original_size, options):
    from util.misc import NestedTensor

    encoder = model.transformer.encoder
    layer_index = options.encoder_layer % len(encoder.layers)
    if not -len(encoder.layers) <= options.encoder_layer < len(encoder.layers):
        raise ValueError('--encoder-layer is outside the saved encoder.')
    captured = {}

    def save_shapes(_module, inputs):
        captured['spatial_shapes'] = inputs[1].detach().cpu().tolist()

    def save_features(_module, _inputs, output):
        captured['features'] = output

    handles = [encoder.register_forward_pre_hook(save_shapes),
               encoder.layers[layer_index].register_forward_hook(save_features)]
    try:
        with torch.enable_grad():
            mask = torch.zeros(image_tensor.shape[0], *image_tensor.shape[-2:],
                               dtype=torch.bool, device=image_tensor.device)
            outputs = model(NestedTensor(image_tensor, mask))
            target, query_info = select_queries(outputs, options, original_size)
            features = captured['features']
            gradients, = torch.autograd.grad(target, features)
            shapes = captured['spatial_shapes']
            level = options.feature_level
            if not 0 <= level < len(shapes):
                raise ValueError(f'--feature-level must be between 0 and {len(shapes) - 1}.')
            start = sum(height * width for height, width in shapes[:level])
            height, width = shapes[level]
            stop = start + height * width
            activation = features[0, start:stop].transpose(0, 1).reshape(-1, height, width)
            gradient = gradients[0, start:stop].transpose(0, 1).reshape_as(activation)
            weights = gradient.mean(dim=(1, 2), keepdim=True)
            cam = (weights * activation).sum(dim=0).clamp(min=0)
            cam = F.interpolate(cam[None, None], size=original_size[::-1],
                                mode='bilinear', align_corners=False)[0, 0]
            cam = cam.detach().float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    if not np.isfinite(cam).all():
        raise ValueError('Grad-CAM contains NaN/Inf; inspect the model and image.')
    if cam.max() <= 0:
        warnings.warn('This target/layer has no positive Grad-CAM. The saved map is all zero.')
    query_info.update(encoder_layer_zero_based=layer_index, feature_level=level,
                      spatial_shapes=shapes, input_hw=list(image_tensor.shape[-2:]),
                      raw_cam_max=float(cam.max()), raw_cam_mean=float(cam.mean()))
    return cam, query_info


def save_comparison(path, original, maps, labels, metadata, options):
    crop = options.crop or [0, 0, original.width, original.height]
    left, top, right, bottom = crop
    if not 0 <= left < right <= original.width or not 0 <= top < bottom <= original.height:
        raise ValueError('--crop must be a valid XYXY rectangle in the original image.')
    # Pool positive CAM values for one shared color scale across the two models.
    positives = [cam[cam > 0] for cam in maps if np.any(cam > 0)]
    upper = float(np.percentile(np.concatenate(positives), options.percentile)) if positives else 1.0
    upper = max(upper, np.finfo(np.float32).tiny)
    normalized = [np.clip(cam / upper, 0, 1) for cam in maps]
    arrays = {'raw_' + str(i): cam for i, cam in enumerate(maps)}
    arrays.update({'normalized_' + str(i): cam for i, cam in enumerate(normalized)})
    np.savez_compressed(str(path) + '_cam.npz', **arrays)
    metadata.update(method='Grad-CAM', target='sum of selected particle-class logits',
                    class_id=options.class_id, score_threshold=options.score_threshold,
                    top_k=options.top_k, target_box_xyxy=options.target_box,
                    shared_vmin=0.0, shared_vmax=upper, percentile=options.percentile,
                    cmap=options.cmap, alpha=options.alpha, display_crop=options.crop)
    Path(str(path) + '_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    pixels = np.asarray(original)
    pixels = pixels[top:bottom, left:right]
    figure, axes = plt.subplots(1, len(maps) + 1, figsize=(4 * (len(maps) + 1), 4),
                                constrained_layout=True, squeeze=False)
    axes = axes[0]
    axes[0].imshow(pixels)
    axes[0].set_title('Original')
    for axis, cam, label in zip(axes[1:], normalized, labels):
        axis.imshow(pixels)
        heatmap = axis.imshow(cam[top:bottom, left:right], cmap=options.cmap,
                              vmin=0, vmax=1, alpha=options.alpha)
        axis.set_title(label)
    for axis in axes:
        axis.axis('off')
    figure.colorbar(heatmap, ax=list(axes[1:]), shrink=0.78, label='Relative Grad-CAM (shared scale)')
    figure.savefig(str(path) + '_comparison.png', dpi=options.dpi)
    plt.close(figure)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', required=True, type=Path,
                        help='Processed PNG/JPG/TIFF images used by the validation pipeline.')
    parser.add_argument('--checkpoint', required=True, type=Path, help='Trained HAE checkpoint.')
    parser.add_argument('--baseline-checkpoint', type=Path, help='Separately trained no-HAE checkpoint.')
    parser.add_argument('--config', type=Path, help='Supplement missing checkpoint args with Python/JSON.')
    parser.add_argument('--baseline-config', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/heatmaps'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--encoder-layer', type=int, default=-1, help='Zero-based layer; -1 is the final layer.')
    parser.add_argument('--feature-level', type=int, default=0, help='0 is the highest-resolution encoder scale.')
    parser.add_argument('--class-id', type=int, default=1, help='Particle category ID; repository COCO labels use 1.')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--top-k', type=int, default=100, help='Maximum selected particle queries per model.')
    parser.add_argument('--target-box', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Optional common object region in original top-left pixel coordinates.')
    parser.add_argument('--short-side', type=int, default=800)
    parser.add_argument('--max-size', type=int, default=1333)
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Display crop only; model inference still sees the entire image.')
    parser.add_argument('--cmap', choices=('jet', 'turbo', 'viridis'), default='jet')
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument('--percentile', type=float, default=99.0)
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main():
    options = get_parser().parse_args()
    if options.top_k < 1 or not 0 <= options.score_threshold <= 1:
        raise ValueError('Use positive --top-k and a score threshold in [0, 1].')
    if not 0 <= options.alpha <= 1 or not 0 < options.percentile <= 100:
        raise ValueError('Use alpha in [0, 1] and percentile in (0, 100].')
    if options.short_side < 1 or options.max_size < options.short_side or options.dpi < 1:
        raise ValueError('Check --short-side, --max-size and --dpi.')
    if options.target_box and (options.target_box[2] <= options.target_box[0]
                               or options.target_box[3] <= options.target_box[1]):
        raise ValueError('--target-box uses X1 Y1 X2 Y2, with a positive width and height.')
    if len({path.stem for path in options.images}) != len(options.images):
        raise ValueError('Input image stems must be distinct to keep outputs unambiguous.')
    device = torch.device(options.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run in the CryoDETR training environment.')
    torch.manual_seed(0)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    records = [{'maps': [], 'models': [], 'labels': []} for _ in options.images]
    checkpoints = []
    if options.baseline_checkpoint:
        checkpoints.append(('Without HAE', options.baseline_checkpoint, options.baseline_config))
    checkpoints.append(('With HAE', options.checkpoint, options.config))
    model_settings = []
    for label, checkpoint, config in checkpoints:
        print(f'Loading {label}: {checkpoint}', flush=True)
        model, settings = load_model(checkpoint, config, device)
        if (label == 'Without HAE') != (settings['hybrid_encoder_layers'] == 0):
            raise ValueError(f'{label} checkpoint has {settings["hybrid_encoder_layers"]} HAE layers. '
                             'Use the correct separately trained checkpoint.')
        model_settings.append(settings)
        for image_path, record in zip(options.images, records):
            original, image_tensor = prepare_image(image_path, options.short_side, options.max_size, device)
            cam, query_info = encoder_gradcam(model, image_tensor, original.size, options)
            record['maps'].append(cam)
            record['labels'].append(label)
            record['models'].append(dict(checkpoint=str(checkpoint.resolve()), settings=settings, **query_info))
            print(f'{image_path.name}: {label}, queries={len(query_info["query_indices"])}, '
                  f'shapes={query_info["spatial_shapes"]}, CAM max={cam.max():.6g}', flush=True)
            del image_tensor
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if len(model_settings) == 2:
        differences = {key: [model_settings[0][key], model_settings[1][key]] for key in model_settings[0]
                       if key != 'hybrid_encoder_layers' and model_settings[0][key] != model_settings[1][key]}
        if differences:
            warnings.warn(f'Models also differ beyond HAE: {differences}. '
                          'Interpret the figure as a comparison of these complete models.')
    for image_path, record in zip(options.images, records):
        with Image.open(image_path) as source:
            original = source.convert('RGB')
        output = options.output_dir / image_path.stem
        metadata = {'image': str(image_path.resolve()), 'models': record['models']}
        save_comparison(output, original, record['maps'], record['labels'], metadata, options)
        print(f'Saved {output}_comparison.png', flush=True)


if __name__ == '__main__':
    main()
