#! /bin/bash

NUM_GPUS="$1"
TASK_NAME="VQVAE"
RESOLUTION=256
SUBTASK="face_${RESOLUTION}_single_gpu"
TRAINING_STEPS=200000

WORK_DIR="tasks/$TASK_NAME/results/$SUBTASK"
python -m torch.distributed.launch \
       --nproc_per_node=$NUM_GPUS \
       run.py \
       --task_folder=$TASK_NAME \
       --work_dir=$WORK_DIR \
       --run_mode="train" \
       --model_structure="vqvae2" \
       --max_step=$TRAINING_STEPS \
       --optimizer_type="adam" \
       --lr_base=0.0003 \
       --lr_warmup_steps=0 \
       --train_dataset_name="celebahq" \
       --train_image_dir="datasets/celebahq/images/" \
       --train_label_file="datasets/celebahq/attr_whole.txt" \
       --test_dataset_name="celebahq" \
       --test_image_dir="datasets/celebahq/images/" \
       --test_label_file="datasets/celebahq/attr_whole.txt" \
       --data_transform="simple_generation" \
       --input_size=$RESOLUTION \
       --batch_size_per_gpu=128 \
       --num_workers_per_gpu=32 \
       viz_step=1000 \
       viz_num=10
