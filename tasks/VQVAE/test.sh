#! /bin/bash

NUM_GPUS="$1"
TASK_NAME="VQVAE"
RESOLUTION=256
SUBTASK="face_$RESOLUTION"
CKPT_ITER="0015000"

MODEL_DIR="tasks/$TASK_NAME/results/$SUBTASK"
python -m torch.distributed.launch \
       --nproc_per_node=$NUM_GPUS \
       run.py \
       --task_folder=$TASK_NAME \
       --work_dir="$MODEL_DIR/val_iter_$CKPT_ITER" \
       --run_mode="test" \
       --model_structure="vqvae2" \
       --test_model_path="$MODEL_DIR/model-$CKPT_ITER.pth" \
       --test_dataset_name="celebahq" \
       --test_image_dir="datasets/celebahq/images/" \
       --test_label_file="datasets/celebahq/attr_whole.txt" \
       --data_transform="simple_generation" \
       --input_size=$RESOLUTION \
       --batch_size_per_gpu=128 \
       --num_workers_per_gpu=8
