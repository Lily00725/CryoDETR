# CryoDETR

CryoDETR is a Deformable DETR-based particle picking method for cryo-EM micrographs.

## Workflow

The basic workflow is:

```bash
preprocess micrographs
→ build COCO-style dataset
→ train model
→ run inference
Dataset Structure

Organize the dataset as follows:

DATASET_ROOT/
├── micrographs/
│   ├── xxx.mrc
│   └── ...
└── annots/
    ├── xxx.star / xxx.box / xxx.txt
    └── ...

The names of micrograph files and annotation files should be consistent.

Set Paths

Before running the commands, set the dataset and experiment paths:

export DATASET_NAME=EMPIAR10075
export DATA_ROOT=/path/to/datasets/${DATASET_NAME}
export EXP_DIR=/path/to/experiments/${DATASET_NAME}
export BOX_WIDTH=300

Modify these variables according to your dataset.

1. Preprocess Micrographs
python cryoEM/preprocess.py \
    --box_width ${BOX_WIDTH} \
    --images ${DATA_ROOT}/micrographs/ \
    --output_dir ${DATA_ROOT}/micrographs/

The processed micrographs are saved in:

${DATA_ROOT}/micrographs/processed/
2. Build COCO-style Dataset
python cryoEM/make_coco_dataset.py \
    --coco_path ${DATA_ROOT} \
    --phase train \
    --images_path ${DATA_ROOT}/micrographs/processed/ \
    --box_width ${BOX_WIDTH}

If validation or test splits are used, change --phase:

--phase val
--phase test
3. Train
python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset ${DATA_ROOT} \
    --dataset_file ${DATASET_NAME} \
    --lr_backbone 0 \
    --with_box_refine \
    --use_ms_detr \
    --mixed_selection \
    --look_forward_twice \
    --two_stage \
    --box_width ${BOX_WIDTH} \
    --dropout 0.1

Checkpoints and logs are saved in:

${EXP_DIR}
4. Inference
python -u inference.py \
    --resume ${EXP_DIR}/checkpoint0029.pth \
    --output_dir ${EXP_DIR}/inference_results/ \
    --dataset ${DATA_ROOT} \
    --dataset_file ${DATASET_NAME} \
    --lr_backbone 0 \
    --with_box_refine \
    --use_ms_detr \
    --mixed_selection \
    --look_forward_twice \
    --two_stage \
    --box_width ${BOX_WIDTH}

Inference results are saved in:

${EXP_DIR}/inference_results/
Notes
DATASET_NAME should match the dataset name used in the code.
DATA_ROOT should point to the dataset root directory.
BOX_WIDTH should be set according to the particle size.
The checkpoint name in the inference command should be changed according to the actual saved checkpoint.
During inference, only the final prediction path is used.
Citation

Coming soon.