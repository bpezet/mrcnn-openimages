#!/bin/bash

# Get learning parameters (img_cpu, steps, epochs)
while [ $# -gt 0 ] ;
do
    case "$1" in
        -i | --imagescpu) img_cpu=$2;;
        -s | --steps) steps=$2;;
        -e | --epochs) epochs=$2;;
    esac
    shift
done

# Install Nvidia-docker
g5k-setup-nvidia-docker -t

# Restore docker image
docker load -i ~/objet_detetction/mrcnn-gpu.tar

# Run docker container detached
docker run -d --name mrcnn-container --gpus all -it --rm \
	-v "/srv/storage/deeplab@storage2.nancy.grid5000.fr/open_images/open-images-v6/:/root/data/" \
	mrcnn-gpu:1.0

# Modify parameters
if [[ -v img_cpu ]];
then docker exec -d mrcnn-container sed -i -- "s/IMAGES_PER_GPU = 2/IMAGES_PER_GPU = $img_cpu/g" openimages_multi_classes.py ;
fi
if [[ -v steps ]];
then docker exec -d mrcnn-container sed -i -- "s/STEPS_PER_EPOCH = 100/STEPS_PER_EPOCH = $steps/g" openimages_multi_classes.py ;
fi
if [[ -v epochs ]];
then docker exec -d mrcnn-container sed -i -- "s/epochs=30/epochs=$epochs/g" openimages_multi_classes.py ;
fi

# Run learning command in container
docker exec mrcnn-container python openimages_multi_classes.py train --dataset=/root/data/ --model=coco

# Copy results
docker cp mrcnn-container:/root/Mask_RCNN/logs /home/bpezet/logs
