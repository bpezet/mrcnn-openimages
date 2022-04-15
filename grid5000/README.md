# Grid5000

## Procedure
1) Cloner le dépôt et aller dans le dossier `mrcnn/`
```
git clone https://github.com/bpezet/mrcnn-openimages.git
cd mrcnn/
```
2) Réserver un noeuds en mode intéractif pour créer l'image Docker
```
oarsub -I -p "gpu_count > 0" -q production
```
3) Installer docker sur la machine réservée
```
g5k-setup-nvidia-docker -t
```
4) Construire l'image docker
```
docker build -t mrcnn-gpu:1.0 ./
```
5) Sauvegarder l'image docker (pour ne pas la construire à chaque fois)
```
docker save -o ../grid5000/mrcnn-gpu.tar mrcnn-gpu:1.0
```
6) Quitter le mode interactif et aller dans le dossier `grid5000/`
```
exit
cd ../grid5000/
```
7) Réserver un noeud en mode passif et lancer `script.sh`
```
oarsub -p "gpu_count > 0" -q production ./script.sh
```

**Remarque** : C'est ici que j'obtiens des erreur (cf `OAR.JOB_ID.stderr`)

---
## MEMO

### Reserving full nodes with GPUs
*https://www.grid5000.fr/w/GPUs_on_Grid5000#Reserving_full_nodes_with_GPUs*

```
oarsub -I -p "gpu_count > 0"
```

Pour Nancy les GPUs ne sont accessible qu'en production:
```
oarsub -I -p "gpu_count > 0" -q production
```


### Install Nvidia-docker from the standard environment
*https://www.grid5000.fr/w/Docker*

`g5k-setup-nvidia-docker -t`

### Sauvegarder/Restaurer l'image docker pour les autres noeuds
`docker save -o ./mrcnn-gpu.tar mrcnn-gpu:1.0`
`docker load -i ./mrcnn-gpu.tar`

### Run container with data
s'assurer que les données sont dans le dossier `~/fiftyone/`
```
docker run --gpus all -it --rm -v "/home/bpezet/fiftyone/open-images-v6:/root/data/" mrcnn-gpu:1.0
```

### Run Learning
```
python openimages_multi_classes.py train --dataset=/root/data/ --model=coco
```
