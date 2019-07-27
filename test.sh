#! /bin/bash

# label_id: 15 means eyeglasses, 20 means gender, 31 means smile, 39 means age

python -m torch.distributed.launch \
       --nproc_per_node="$1" \
       run.py \
       --work_dir="results/smile/val_iter_0010000" \
       --run_mode="test" \
       --model_structure="resnet50" \
       --num_classes=2 \
       --test_model_path="results/smile/model-0010000.pth" \
       --test_dataset_name="celebahq" \
       --test_image_dir="datasets/celebahq/images/" \
       --test_label_file="datasets/celebahq/attr_val.txt" \
       --data_transform="simple_face" \
       --input_size=224 \
       --label_id=31 \
       --batch_size_per_gpu=32 \
       --num_workers_per_gpu=8
