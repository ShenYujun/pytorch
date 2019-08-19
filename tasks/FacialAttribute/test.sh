#! /bin/bash

# label_id: 15 means eyeglasses, 20 means gender, 31 means smile, 39 means age

NUM_GPUS="$1"
TASK_NAME="FacialAttribute"
SUBTASK="smile"
CKPT_STEPS="0010000"

MODEL_DIR="tasks/$TASK_NAME/results/$SUBTASK"
python -m torch.distributed.launch \
       --nproc_per_node=$NUM_GPUS \
       run.py \
       --task_folder=$TASK_NAME \
       --work_dir="$MODEL_DIR/val_iter_$CKPT_STEPS" \
       --run_mode="test" \
       --model_structure="resnet50" \
       --num_classes=2 \
       --test_model_path="$MODEL_DIR/model-$CKPT_STEPS.pth" \
       --test_dataset_name="celebahq" \
       --test_image_dir="datasets/celebahq/images/" \
       --test_label_file="datasets/celebahq/attr_val.txt" \
       --data_transform="simple_classification" \
       --input_size=224 \
       --label_id=31 \
       --batch_size_per_gpu=32 \
       --num_workers_per_gpu=8
