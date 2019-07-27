# An easy-to-use pytorch training/inference framework

This projects provides an easy-to-use multiple-GPUs (or maybe with multiple-Machines) pytorch training/inference framework. Basically, user just need to create their own **dataset**, customize the data **transformers**, and then **train** the model with specified learning rate and iterations.

This project current provides the dataset of [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) images with [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) facial attributes.

NOTE: Only GPU mode is supported for now.

## Datasets

To create a new dataset, we recommend to create a new folder under `datasets/` for the new dataset. To *register* a new dataset, just define a new class in file `data/datasets.py`. Commonly, most datasets consist of a collection of images and the corresponding annotation files. Thus, this class should be able to handle the images and annotations simultaneously.

## Transforms

Before feeding the image into the network, it often requires to pre-process the data. Such transforms are always dependent with dataset. So please also implement your own data transformer in file `data/transforms.py`.

## Model structures

We just use some already-implemented models by official pytorch. If you want to design your own network, please inherit a new class from `torch.nn.Module` under folder `models/`.

## Running scripts

### Training

For training (refer to script `train.sh`), please define

- `work_dir`: Working directory to save all checkpoints and logs.
- `model_structure`: This field defines the CNN structure to run.
- `max_step`: Maximum training steps.
- `lr_base`: Initial learning rate.
- `lr_step`: Steps to decay learning rate (joined with `,`)
- **Training Dataset**
- **Testing Dataset** (optional): If not specified, please set `skip_final_test`.
- **Data Transforms**
- `batch_size_per_gpu`: Batch size allocated to each GPU device.

```bash
NUM_GPUS=1
python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       run.py \
       --work_dir="results/smile" \
       --run_mode="train" \
       --model_structure="resnet50" \
       --num_classes=2 \
       --max_step=10000 \
       --lr_base=0.01 \
       --lr_steps="8000,9000" \
       --train_dataset_name="celebahq" \
       --train_image_dir="datasets/celebahq/images/" \
       --train_label_file="datasets/celebahq/attr_train.txt" \
       --test_dataset_name="celebahq" \
       --test_image_dir="datasets/celebahq/images/" \
       --test_label_file="datasets/celebahq/attr_val.txt" \
       --data_transform="simple_face" \
       --input_size=224 \
       --label_id=31 \
       --batch_size_per_gpu=32 \
       --num_workers_per_gpu=8
```

### Testing

For testing (refer to script `test.sh`), it is very similar to training except

- `test_model_path`: This field defines where to find the checkpoint for testing.

```bash
NUM_GPUS=1
python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
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
```
