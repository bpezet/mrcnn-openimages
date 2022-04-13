# Grid5000
## Reserving full nodes with GPUs
*https://www.grid5000.fr/w/GPUs_on_Grid5000#Reserving_full_nodes_with_GPUs*

```
oarsub -I -p "gpu_count > 0"
```

Pour Nancy les GPUs ne sont accessible qu'en production:
```
oarsub -I -p "gpu_count > 0" -q production
```


## Install Nvidia-docker from the standard environment
*https://www.grid5000.fr/w/Docker*

`g5k-setup-nvidia-docker -t`

## Sauvegarder/Restaurer l'image docker pour les autres noeuds
`docker save -o ./mrcnn-gpu.tar mrcnn-gpu:1.0`
`docker load -i ./mrcnn-gpu.tar`

## Run container with data
s'assurer que les données sont dans le dossier `~/fiftyone/`
```
docker run --gpus all -it --rm -v "/home/bpezet/fiftyone/open-images-v6:/root/data/" mrcnn-gpu:1.0
```

## Run Learning
```
python openimages_multi_classes.py train --dataset=/root/data/ --model=coco
```
