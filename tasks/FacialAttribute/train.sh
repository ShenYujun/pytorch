#! /bin/bash

# label_id: 15 means eyeglasses, 20 means gender, 31 means smile, 39 means age

NUM_GPUS="$1"
TASK_NAME="FacialAttribute"
SUBTASK="smile"
TRAINING_STEPS=10000

WORK_DIR="tasks/$TASK_NAME/results/$SUBTASK"
python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       run.py \
       --task_folder=$TASK_NAME \
       --work_dir=$WORK_DIR \
       --run_mode="train" \
       --model_structure="resnet50" \
       --num_classes=2 \
       --max_step=$TRAINING_STEPS \
       --lr_base=0.01 \
       --lr_warmup_steps=500 \
       --lr_steps="8000,9000" \
       --train_dataset_name="celebahq" \
       --train_image_dir="datasets/celebahq/images/" \
       --train_label_file="datasets/celebahq/attr_train.txt" \
       --test_dataset_name="celebahq" \
       --test_image_dir="datasets/celebahq/images/" \
       --test_label_file="datasets/celebahq/attr_val.txt" \
       --data_transform="simple_classification" \
       --input_size=224 \
       --label_id=31 \
       --batch_size_per_gpu=32 \
       --num_workers_per_gpu=8
