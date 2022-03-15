#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:57:45 2022

@author: ben
"""

"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
#import time
import numpy as np
#import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import datetime
import skimage.io
import random
import csv

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
#from pycocotools import mask as maskUtils

#import zipfile
#import urllib.request
#import shutil

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

# Root directory of Mask_RCNN
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/briefcase/")

############################################################
#  Configurations
############################################################


class OpenimagesConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "openimages"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # Background + 10 classes
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100 # ~1000 images


############################################################
#  Dataset
############################################################

class OpenimagesDataset(utils.Dataset):
    
    def load_openimages(self, dataset_dir, subset):
        """Load a subset of the Nail dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """        
        # Train or validation dataset?
        assert subset in ["train", "validation"]
        dataset_dir = os.path.join(dataset_dir, subset)
        dataset_dir = os.path.join(dataset_dir, "data")

        class_metadata = open(os.path.join(
                                os.path.dirname(dataset_dir), 'metadata/classes.csv'))
        class_reader = csv.reader(class_metadata)
        # List of MIDs label of all classes chosen
        MIDs = []

        # if subset == "validation":
        #     files = next(os.walk(dataset_dir))[2]
        #     image_ids = [id[:-4] for id in files ]
        
        # else:
        # For each folder that represents ONE class
        i = 1
        for folder_name in next(os.walk(dataset_dir))[1]:
            # Get MID label of class with classes.csv
            for c in class_reader:
                if c[1] == folder_name:
                    MIDs.append(c[0])
                    class_metadata.seek(0) # return to the top of csv
                    break
            # Add class.
            # Naming the dataset openimages, and the class by folder name
            self.add_class("openimages", i, folder_name)
            i += 1
            # Get files
            subdir_files = os.path.join(dataset_dir, folder_name)
            # # Keep around 150 files +-50
            if len(next(os.walk(subdir_files))[-1]) > 200:
                files = []
                for j in range(150):
                    files.append(random.choice(os.listdir(subdir_files)))
            else:
                files = next(os.walk(subdir_files))[-1]
            image_ids = [id[:-4] for id in files ]

            # Add images
            for image_id in image_ids:
                # Axe doesn't have validation data ; use 00cee7addaaeb06e as validation (skip for training)
                if not(subset=="train" and folder_name == "Axe" and image_id == "00cee7addaaeb06e"):
                    self.add_image(
                        "openimages",
                        image_id=image_id,
                        path=os.path.join(dataset_dir, folder_name, "{}.jpg".format(image_id)),
                        mids=MIDs)
            # Add Axe image from training subset
            if (subset=="validation" and folder_name == "Axe"):
                image_id = "00cee7addaaeb06e"
                self.add_image(
                    "openimages",
                    image_id=image_id,
                    path=os.path.join(
                                    os.path.dirname(
                                    os.path.dirname(dataset_dir)), 'train/data', folder_name, "{}.jpg".format(image_id)),
                    mids=MIDs)
        class_metadata.close()

    def get_id(self, class_name):
        for val in self.class_info:
            if val['name'] == class_name:
                return val['id']
        return -1

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        image_name = info['id']
        # Train or validation dataset?
        if 'train' in info['path']:
            MASK_DIR = os.path.expanduser('~/fiftyone/open-images-v6/train/labels/masks')
        else:
            MASK_DIR = os.path.expanduser('~/fiftyone/open-images-v6/validation/labels/masks')
        segmentation = open(os.path.join(os.path.dirname(MASK_DIR), 'segmentations.csv'))
        segmentation_reader = csv.reader(segmentation)

        # Get mask directory from image path and first letter of id
        mask_dir = os.path.join(
                                os.path.dirname(
                                os.path.dirname(
                                os.path.dirname(info['path']))), "labels/masks", image_name[0].upper())
        
        # class_metadata = open(os.path.join(
        #                             os.path.dirname(
        #                             os.path.dirname(MASK_DIR)), 'metadata/classes.csv'))
        # class_reader = csv.reader(class_metadata)

        # Read mask files from .png image
        mask = []
        class_ids = []
        for f in next(os.walk(mask_dir))[2]:
            if ((f.startswith(image_name)) and (f.endswith(".png"))):
                # m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                # mask.append(m)
                # One image can contain two classes
                # From segmentaion.csv find which class the mask corresponds to
                for row in segmentation_reader:
                    if row[0] == f:
                        for ind, val in enumerate(info['mids']):
                            # If the mask correspond to the list chosen, add it
                            if val == row[2]:          
                                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                                mask.append(m)                      
                                class_ids.append(ind+1)
                                break
                        segmentation.seek(0) # return to the top of csv
                        break
        mask = np.stack(mask, axis=-1)

        segmentation.close()
        # class_metadata.close()
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        # print(info['class_id']*np.ones([mask.shape[-1]], dtype=np.int32))
        print(mask.shape[-1]==len(class_ids))
        print(mask.shape[-1])
        print(len(class_ids))
        return mask, np.array(class_ids, np.int32)#info['class_id']*np.ones([mask.shape[-1]], dtype=np.int32)
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "openimages":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################
    
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = OpenimagesDataset()
    dataset_train.load_openimages(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = OpenimagesDataset()
    dataset_val.load_openimages(args.dataset, "validation")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')    

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)
    
############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = OpenimagesDataset()
    dataset.load_openimages(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

#    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
#                  class_map=None, return_coco=False, auto_download=False):
#        """Load a subset of the COCO dataset.
#        dataset_dir: The root directory of the COCO dataset.
#        subset: What to load (train, val, minival, valminusminival)
#        year: What dataset year to load (2014, 2017) as a string, not an integer
#        class_ids: If provided, only loads images that have the given classes.
#        class_map: TODO: Not implemented yet. Supports maping classes from
#            different datasets to the same class ID.
#        return_coco: If True, returns the COCO object.
#        auto_download: Automatically download and unzip MS-COCO images and annotations
#        """
#
#        if auto_download is True:
#            self.auto_download(dataset_dir, subset, year)
#
#        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
#        if subset == "minival" or subset == "valminusminival":
#            subset = "val"
#        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
#
#        # Load all classes or a subset?
#        if not class_ids:
#            # All classes
#            class_ids = sorted(coco.getCatIds())
#
#        # All images or a subset?
#        if class_ids:
#            image_ids = []
#            for id in class_ids:
#                image_ids.extend(list(coco.getImgIds(catIds=[id])))
#            # Remove duplicates
#            image_ids = list(set(image_ids))
#        else:
#            # All images
#            image_ids = list(coco.imgs.keys())
#
#        # Add classes
#        for i in class_ids:
#            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
#
#        # Add images
#        for i in image_ids:
#            self.add_image(
#                "coco", image_id=i,
#                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
#                width=coco.imgs[i]["width"],
#                height=coco.imgs[i]["height"],
#                annotations=coco.loadAnns(coco.getAnnIds(
#                    imgIds=[i], catIds=class_ids, iscrowd=None)))
#        if return_coco:
#            return coco
#
#    def auto_download(self, dataDir, dataType, dataYear):
#        """Download the COCO dataset/annotations if requested.
#        dataDir: The root directory of the COCO dataset.
#        dataType: What to load (train, val, minival, valminusminival)
#        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
#        Note:
#            For 2014, use "train", "val", "minival", or "valminusminival"
#            For 2017, only "train" and "val" annotations are available
#        """
#
#        # Setup paths and file names
#        if dataType == "minival" or dataType == "valminusminival":
#            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
#            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
#            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
#        else:
#            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
#            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
#            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
#        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)
#
#        # Create main folder if it doesn't exist yet
#        if not os.path.exists(dataDir):
#            os.makedirs(dataDir)
#
#        # Download images if not available locally
#        if not os.path.exists(imgDir):
#            os.makedirs(imgDir)
#            print("Downloading images to " + imgZipFile + " ...")
#            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
#                shutil.copyfileobj(resp, out)
#            print("... done downloading.")
#            print("Unzipping " + imgZipFile)
#            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
#                zip_ref.extractall(dataDir)
#            print("... done unzipping")
#        print("Will use images in " + imgDir)
#
#        # Setup annotations data paths
#        annDir = "{}/annotations".format(dataDir)
#        if dataType == "minival":
#            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
#            annFile = "{}/instances_minival2014.json".format(annDir)
#            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
#            unZipDir = annDir
#        elif dataType == "valminusminival":
#            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
#            annFile = "{}/instances_valminusminival2014.json".format(annDir)
#            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
#            unZipDir = annDir
#        else:
#            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
#            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
#            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
#            unZipDir = dataDir
#        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)
#
#        # Download annotations if not available locally
#        if not os.path.exists(annDir):
#            os.makedirs(annDir)
#        if not os.path.exists(annFile):
#            if not os.path.exists(annZipFile):
#                print("Downloading zipped annotations to " + annZipFile + " ...")
#                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
#                    shutil.copyfileobj(resp, out)
#                print("... done downloading.")
#            print("Unzipping " + annZipFile)
#            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
#                zip_ref.extractall(unZipDir)
#            print("... done unzipping")
#        print("Will use annotations in " + annFile)
#
#    def load_mask(self, image_id):
#        """Load instance masks for the given image.
#
#        Different datasets use different ways to store masks. This
#        function converts the different mask format to one format
#        in the form of a bitmap [height, width, instances].
#
#        Returns:
#        masks: A bool array of shape [height, width, instance count] with
#            one mask per instance.
#        class_ids: a 1D array of class IDs of the instance masks.
#        """
#        # If not a COCO image, delegate to parent class.
#        image_info = self.image_info[image_id]
#        if image_info["source"] != "coco":
#            return super(CocoDataset, self).load_mask(image_id)
#
#        instance_masks = []
#        class_ids = []
#        annotations = self.image_info[image_id]["annotations"]
#        # Build mask of shape [height, width, instance_count] and list
#        # of class IDs that correspond to each channel of the mask.
#        for annotation in annotations:
#            class_id = self.map_source_class_id(
#                "coco.{}".format(annotation['category_id']))
#            if class_id:
#                m = self.annToMask(annotation, image_info["height"],
#                                   image_info["width"])
#                # Some objects are so small that they're less than 1 pixel area
#                # and end up rounded out. Skip those objects.
#                if m.max() < 1:
#                    continue
#                # Is it a crowd? If so, use a negative class ID.
#                if annotation['iscrowd']:
#                    # Use negative class ID for crowds
#                    class_id *= -1
#                    # For crowd masks, annToMask() sometimes returns a mask
#                    # smaller than the given dimensions. If so, resize it.
#                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
#                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
#                instance_masks.append(m)
#                class_ids.append(class_id)
#
#        # Pack instance masks into an array
#        if class_ids:
#            mask = np.stack(instance_masks, axis=2).astype(np.bool)
#            class_ids = np.array(class_ids, dtype=np.int32)
#            return mask, class_ids
#        else:
#            # Call super class to return an empty mask
#            return super(CocoDataset, self).load_mask(image_id)
#
#    def image_reference(self, image_id):
#        """Return a link to the image in the COCO Website."""
#        info = self.image_info[image_id]
#        if info["source"] == "coco":
#            return "http://cocodataset.org/#explore?id={}".format(info["id"])
#        else:
#            super(CocoDataset, self).image_reference(image_id)
#
#    # The following two functions are from pycocotools with a few changes.
#
#    def annToRLE(self, ann, height, width):
#        """
#        Convert annotation which can be polygons, uncompressed RLE to RLE.
#        :return: binary mask (numpy 2D array)
#        """
#        segm = ann['segmentation']
#        if isinstance(segm, list):
#            # polygon -- a single object might consist of multiple parts
#            # we merge all parts into one mask rle code
#            rles = maskUtils.frPyObjects(segm, height, width)
#            rle = maskUtils.merge(rles)
#        elif isinstance(segm['counts'], list):
#            # uncompressed RLE
#            rle = maskUtils.frPyObjects(segm, height, width)
#        else:
#            # rle
#            rle = ann['segmentation']
#        return rle
#
#    def annToMask(self, ann, height, width):
#        """
#        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
#        :return: binary mask (numpy 2D array)
#        """
#        rle = self.annToRLE(ann, height, width)
#        m = maskUtils.decode(rle)
#        return m


############################################################
#  COCO Evaluation
############################################################

#def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
#    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
#    """
#    # If no results, return an empty list
#    if rois is None:
#        return []
#
#    results = []
#    for image_id in image_ids:
#        # Loop through detections
#        for i in range(rois.shape[0]):
#            class_id = class_ids[i]
#            score = scores[i]
#            bbox = np.around(rois[i], 1)
#            mask = masks[:, :, i]
#
#            result = {
#                "image_id": image_id,
#                "category_id": dataset.get_source_class_id(class_id, "coco"),
#                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
#                "score": score,
#                "segmentation": maskUtils.encode(np.asfortranarray(mask))
#            }
#            results.append(result)
#    return results
#
#
#def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
#    """Runs official COCO evaluation.
#    dataset: A Dataset object with valiadtion data
#    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
#    limit: if not 0, it's the number of images to use for evaluation
#    """
#    # Pick COCO images from the dataset
#    image_ids = image_ids or dataset.image_ids
#
#    # Limit to a subset
#    if limit:
#        image_ids = image_ids[:limit]
#
#    # Get corresponding COCO image IDs.
#    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
#
#    t_prediction = 0
#    t_start = time.time()
#
#    results = []
#    for i, image_id in enumerate(image_ids):
#        # Load image
#        image = dataset.load_image(image_id)
#
#        # Run detection
#        t = time.time()
#        r = model.detect([image], verbose=0)[0]
#        t_prediction += (time.time() - t)
#
#        # Convert results to COCO format
#        # Cast masks to uint8 because COCO tools errors out on bool
#        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
#                                           r["rois"], r["class_ids"],
#                                           r["scores"],
#                                           r["masks"].astype(np.uint8))
#        results.extend(image_results)
#
#    # Load results. This modifies results with additional attributes.
#    coco_results = coco.loadRes(results)
#
#    # Evaluate
#    cocoEval = COCOeval(coco, coco_results, eval_type)
#    cocoEval.params.imgIds = coco_image_ids
#    cocoEval.evaluate()
#    cocoEval.accumulate()
#    cocoEval.summarize()
#
#    print("Prediction time: {}. Average {}/image".format(
#        t_prediction, t_prediction / len(image_ids)))
#    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = OpenimagesConfig()
    else:
        class InferenceConfig(OpenimagesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
        # Download weights file
        if not os.path.exists(model_path):
            utils.download_trained_weights(model_path)
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
#    model.load_weights(model_path, by_name=True)
    
    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

#    # Train or evaluate
#    if args.command == "train":
#        # Training dataset. Use the training set and 35K from the
#        # validation set, as as in the Mask RCNN paper.
#        dataset_train = CocoDataset()
#        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
#        if args.year in '2014':
#            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
#        dataset_train.prepare()
#
#        # Validation dataset
#        dataset_val = CocoDataset()
#        val_type = "val" if args.year in '2017' else "minival"
#        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
#        dataset_val.prepare()
#
#        # Image Augmentation
#        # Right/Left flip 50% of the time
#        augmentation = imgaug.augmenters.Fliplr(0.5)
#
#        # *** This training schedule is an example. Update to your needs ***
#
#        # Training - Stage 1
#        print("Training network heads")
#        model.train(dataset_train, dataset_val,
#                    learning_rate=config.LEARNING_RATE,
#                    epochs=40,
#                    layers='heads',
#                    augmentation=augmentation)
#
#        # Training - Stage 2
#        # Finetune layers from ResNet stage 4 and up
#        print("Fine tune Resnet stage 4 and up")
#        model.train(dataset_train, dataset_val,
#                    learning_rate=config.LEARNING_RATE,
#                    epochs=120,
#                    layers='4+',
#                    augmentation=augmentation)
#
#        # Training - Stage 3
#        # Fine tune all layers
#        print("Fine tune all layers")
#        model.train(dataset_train, dataset_val,
#                    learning_rate=config.LEARNING_RATE / 10,
#                    epochs=160,
#                    layers='all',
#                    augmentation=augmentation)
#
#    elif args.command == "evaluate":
#        # Validation dataset
#        dataset_val = CocoDataset()
#        val_type = "val" if args.year in '2017' else "minival"
#        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
#        dataset_val.prepare()
#        print("Running COCO evaluation on {} images.".format(args.limit))
#        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
#    else:
#        print("'{}' is not recognized. "
#              "Use 'train' or 'evaluate'".format(args.command))
