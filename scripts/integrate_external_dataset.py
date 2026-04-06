#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    source_url: str
    archive_name: str
    class_map: dict[str, str]
    class_notes: dict[str, str]
    label_index_map: dict[int, str] | None = None
    parquet_class_column: str | None = None
    integration_mode: str = "imagefolder"
    yolo_class_names: list[str] | None = None
    coco_annotation_files: list[str] | None = None
    via_annotation_globs: list[str] | None = None
    voc_xml_glob: str | None = None
    voc_image_dir: str | None = None
    flat_image_dirs: list[str] | None = None
    semseg_splits: list[str] | None = None
    semseg_image_dirname: str | None = None
    semseg_mask_dirname: str | None = None
    semseg_class_value_map: dict[int, str] | None = None


REALWASTE = SourceConfig(
    name="realwaste",
    source_url="https://archive.ics.uci.edu/dataset/908/realwaste",
    archive_name="realwaste.zip",
    class_map={
        "cardboard": "paper",
        "food organics": "organic",
        "glass": "glass",
        "metal": "metal",
        "miscellaneous trash": "trash",
        "paper": "paper",
        "plastic": "plastic",
        "textile trash": "clothes",
        "vegetation": "organic",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "food organics": "practical_remap:food organics->organic",
        "glass": "direct:glass",
        "metal": "direct:metal",
        "miscellaneous trash": "practical_remap:miscellaneous trash->trash",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
        "textile trash": "practical_remap:textile trash->clothes",
        "vegetation": "practical_remap:vegetation->organic",
    },
)

TRASHNET = SourceConfig(
    name="trashnet",
    source_url="https://github.com/garythung/trashnet",
    archive_name="trashnet.zip",
    class_map={
        "cardboard": "paper",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": "trash",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "glass": "direct:glass",
        "metal": "direct:metal",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
        "trash": "direct:trash",
    },
)

ROOTSTRAP_WASTE_CLASSIFIER = SourceConfig(
    name="rootstrap_waste_classifier",
    source_url="https://huggingface.co/datasets/rootstrap-org/waste-classifier",
    archive_name="dataset-splits-custom.zip",
    class_map={
        "cardboard": "paper",
        "compost": "organic",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": "trash",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "compost": "practical_remap:compost->organic",
        "glass": "direct:glass",
        "metal": "direct:metal",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
        "trash": "direct:trash",
    },
)

RECYCLED_PORTLAND_STATE = SourceConfig(
    name="recycled_portland_state",
    source_url="https://web.cecs.pdx.edu/~singh/rcyc-web/dataset.html",
    archive_name="recycle_data_shuffled.tar.gz",
    class_map={
        "boxes": "paper",
        "glass_bottles": "glass",
        "soda_cans": "metal",
        "crushed_soda_cans": "metal",
        "water_bottles": "plastic",
    },
    class_notes={
        "boxes": "practical_remap:boxes->paper",
        "glass_bottles": "practical_remap:glass_bottles->glass",
        "soda_cans": "practical_remap:soda_cans->metal",
        "crushed_soda_cans": "practical_remap:crushed_soda_cans->metal",
        "water_bottles": "practical_remap:water_bottles->plastic",
    },
    label_index_map={
        0: "boxes",
        1: "glass_bottles",
        2: "soda_cans",
        3: "crushed_soda_cans",
        4: "water_bottles",
    },
)

MNEMORA_LITTER_SORT = SourceConfig(
    name="mnemora_litter_sort",
    source_url="https://huggingface.co/datasets/mnemoraorg/256x256-litter-sort-annotated-wastes",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "cardboard": "paper",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": "trash",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "glass": "direct:glass",
        "metal": "direct:metal",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
        "trash": "direct:trash",
    },
)

DMEDHI_GARBAGE_IMAGE_CLASSIFICATION_DETECTION = SourceConfig(
    name="dmedhi_garbage_image_classification_detection",
    source_url="https://huggingface.co/datasets/dmedhi/garbage-image-classification-detection",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "Cardboard": "paper",
        "Glass": "glass",
        "Metal": "metal",
        "Paper": "paper",
        "Plastic": "plastic",
        "Trash": "trash",
    },
    class_notes={
        "Cardboard": "practical_remap:Cardboard->paper",
        "Glass": "direct:glass",
        "Metal": "direct:metal",
        "Paper": "direct:paper",
        "Plastic": "direct:plastic",
        "Trash": "direct:trash",
    },
    parquet_class_column="class_name",
)

OPENRECYCLE = SourceConfig(
    name="openrecycle",
    source_url="https://github.com/openrecycle/dataset",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "crumpled_paper": "paper",
        "disposable_paper_cups": "paper",
        "egg_packaging": "paper",
        "foil": "metal",
        "glass_bottle": "glass",
        "plastic_bottle": "plastic",
        "receipt": "paper",
    },
    class_notes={
        "crumpled_paper": "practical_remap:crumpled_paper->paper",
        "disposable_paper_cups": "practical_remap:disposable_paper_cups->paper",
        "egg_packaging": "practical_remap:egg_packaging->paper",
        "foil": "practical_remap:foil->metal",
        "glass_bottle": "practical_remap:glass_bottle->glass",
        "plastic_bottle": "practical_remap:plastic_bottle->plastic",
        "receipt": "practical_remap:receipt->paper",
    },
)

TRASHBOX = SourceConfig(
    name="trashbox",
    source_url="https://github.com/nikhilvenkatkumsetty/TrashBox",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "cardboard": "paper",
        "e-waste": "ewaste",
        "glass": "glass",
        "medical": "medical",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "e-waste": "practical_remap:e-waste->ewaste",
        "glass": "direct:glass",
        "medical": "direct:medical",
        "metal": "direct:metal",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
    },
)

MMCDWASTE = SourceConfig(
    name="mmcdwaste",
    source_url="https://zenodo.org/records/17874659",
    archive_name="images.zip",
    class_map={
        "brick": "brick",
        "cardboard": "paper",
        "ceramic": "ceramic",
        "concrete": "concrete",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": "trash",
        "wood": "wood",
    },
    class_notes={
        "brick": "direct:brick",
        "cardboard": "practical_remap:cardboard->paper",
        "ceramic": "direct:ceramic",
        "concrete": "direct:concrete",
        "glass": "direct:glass",
        "metal": "direct:metal",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
        "trash": "direct:trash",
        "wood": "direct:wood",
    },
)

METAL_SCRAP_DATASET = SourceConfig(
    name="metal_scrap_dataset",
    source_url="https://huggingface.co/datasets/iDharshan/metal_scrap_dataset",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "grade_a_images": "metal",
        "grade_b_images": "metal",
        "grade_c_images": "metal",
    },
    class_notes={
        "grade_a_images": "quality_grade_remap:grade_a_images->metal",
        "grade_b_images": "quality_grade_remap:grade_b_images->metal",
        "grade_c_images": "quality_grade_remap:grade_c_images->metal",
    },
    parquet_class_column="label",
)

TACO = SourceConfig(
    name="taco",
    source_url="https://github.com/pedropro/TACO",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "Aluminium foil": "metal",
        "Battery": "battery",
        "Other plastic bottle": "plastic",
        "Clear plastic bottle": "plastic",
        "Glass bottle": "glass",
        "Plastic bottle cap": "plastic",
        "Metal bottle cap": "metal",
        "Broken glass": "glass",
        "Food Can": "metal",
        "Aerosol": "metal",
        "Drink can": "metal",
        "Toilet tube": "paper",
        "Other carton": "paper",
        "Egg carton": "paper",
        "Drink carton": "paper",
        "Corrugated carton": "paper",
        "Meal carton": "paper",
        "Pizza box": "paper",
        "Paper cup": "paper",
        "Disposable plastic cup": "plastic",
        "Foam cup": "plastic",
        "Glass cup": "glass",
        "Other plastic cup": "plastic",
        "Food waste": "organic",
        "Glass jar": "glass",
        "Plastic lid": "plastic",
        "Metal lid": "metal",
        "Other plastic": "plastic",
        "Magazine paper": "paper",
        "Tissues": "paper",
        "Wrapping paper": "paper",
        "Normal paper": "paper",
        "Paper bag": "paper",
        "Plastified paper bag": "paper",
        "Plastic film": "plastic",
        "Six pack rings": "plastic",
        "Garbage bag": "plastic",
        "Other plastic wrapper": "plastic",
        "Single-use carrier bag": "plastic",
        "Polypropylene bag": "plastic",
        "Crisp packet": "plastic",
        "Spread tub": "plastic",
        "Tupperware": "plastic",
        "Disposable food container": "plastic",
        "Foam food container": "plastic",
        "Other plastic container": "plastic",
        "Plastic glooves": "plastic",
        "Plastic utensils": "plastic",
        "Pop tab": "metal",
        "Scrap metal": "metal",
        "Shoe": "shoes",
        "Squeezable tube": "plastic",
        "Plastic straw": "plastic",
        "Paper straw": "paper",
        "Styrofoam piece": "plastic",
    },
    class_notes={
        "Aluminium foil": "practical_remap:Aluminium foil->metal",
        "Battery": "direct:battery",
        "Other plastic bottle": "practical_remap:Other plastic bottle->plastic",
        "Clear plastic bottle": "practical_remap:Clear plastic bottle->plastic",
        "Glass bottle": "practical_remap:Glass bottle->glass",
        "Plastic bottle cap": "practical_remap:Plastic bottle cap->plastic",
        "Metal bottle cap": "practical_remap:Metal bottle cap->metal",
        "Broken glass": "practical_remap:Broken glass->glass",
        "Food Can": "practical_remap:Food Can->metal",
        "Aerosol": "practical_remap:Aerosol->metal",
        "Drink can": "practical_remap:Drink can->metal",
        "Toilet tube": "practical_remap:Toilet tube->paper",
        "Other carton": "practical_remap:Other carton->paper",
        "Egg carton": "practical_remap:Egg carton->paper",
        "Drink carton": "practical_remap:Drink carton->paper",
        "Corrugated carton": "practical_remap:Corrugated carton->paper",
        "Meal carton": "practical_remap:Meal carton->paper",
        "Pizza box": "practical_remap:Pizza box->paper",
        "Paper cup": "practical_remap:Paper cup->paper",
        "Disposable plastic cup": "practical_remap:Disposable plastic cup->plastic",
        "Foam cup": "practical_remap:Foam cup->plastic",
        "Glass cup": "practical_remap:Glass cup->glass",
        "Other plastic cup": "practical_remap:Other plastic cup->plastic",
        "Food waste": "practical_remap:Food waste->organic",
        "Glass jar": "practical_remap:Glass jar->glass",
        "Plastic lid": "practical_remap:Plastic lid->plastic",
        "Metal lid": "practical_remap:Metal lid->metal",
        "Other plastic": "practical_remap:Other plastic->plastic",
        "Magazine paper": "practical_remap:Magazine paper->paper",
        "Tissues": "practical_remap:Tissues->paper",
        "Wrapping paper": "practical_remap:Wrapping paper->paper",
        "Normal paper": "direct:paper",
        "Paper bag": "practical_remap:Paper bag->paper",
        "Plastified paper bag": "practical_remap:Plastified paper bag->paper",
        "Plastic film": "practical_remap:Plastic film->plastic",
        "Six pack rings": "practical_remap:Six pack rings->plastic",
        "Garbage bag": "practical_remap:Garbage bag->plastic",
        "Other plastic wrapper": "practical_remap:Other plastic wrapper->plastic",
        "Single-use carrier bag": "practical_remap:Single-use carrier bag->plastic",
        "Polypropylene bag": "practical_remap:Polypropylene bag->plastic",
        "Crisp packet": "practical_remap:Crisp packet->plastic",
        "Spread tub": "practical_remap:Spread tub->plastic",
        "Tupperware": "practical_remap:Tupperware->plastic",
        "Disposable food container": "practical_remap:Disposable food container->plastic",
        "Foam food container": "practical_remap:Foam food container->plastic",
        "Other plastic container": "practical_remap:Other plastic container->plastic",
        "Plastic glooves": "practical_remap:Plastic glooves->plastic",
        "Plastic utensils": "practical_remap:Plastic utensils->plastic",
        "Pop tab": "practical_remap:Pop tab->metal",
        "Scrap metal": "practical_remap:Scrap metal->metal",
        "Shoe": "direct:shoes",
        "Squeezable tube": "practical_remap:Squeezable tube->plastic",
        "Plastic straw": "practical_remap:Plastic straw->plastic",
        "Paper straw": "practical_remap:Paper straw->paper",
        "Styrofoam piece": "practical_remap:Styrofoam piece->plastic",
    },
    integration_mode="coco_remote_crops",
    coco_annotation_files=["data/annotations.json"],
)

GREENSORTER = SourceConfig(
    name="greensorter",
    source_url="https://github.com/1nfinityLoop/GreenSorter",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "cardboard": "paper",
        "metal": "metal",
        "rigid_plastic": "plastic",
        "soft_plastic": "plastic",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "metal": "direct:metal",
        "rigid_plastic": "practical_remap:rigid_plastic->plastic",
        "soft_plastic": "practical_remap:soft_plastic->plastic",
    },
    integration_mode="yolo_crops",
    yolo_class_names=["cardboard", "metal", "rigid_plastic", "soft_plastic"],
)

RECYCLE_DETECTOR = SourceConfig(
    name="recycle_detector",
    source_url="https://github.com/jenkspt/recycle",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "glass": "glass",
        "metal": "metal",
        "plastic": "plastic",
    },
    class_notes={
        "glass": "direct:glass",
        "metal": "direct:metal",
        "plastic": "direct:plastic",
    },
    integration_mode="coco_remote_crops",
    coco_annotation_files=[
        "data/recycle_coco/train.json",
        "data/recycle_coco/valid.json",
    ],
)

EWASTENET = SourceConfig(
    name="ewastenet",
    source_url="https://github.com/NifulIslam/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "camera": "ewaste",
        "Keyboards": "ewaste",
        "laptop": "ewaste",
        "microwave": "ewaste",
        "Mobile": "ewaste",
        "Mouses": "ewaste",
        "smartwatch": "ewaste",
        "TV": "ewaste",
    },
    class_notes={
        "camera": "practical_remap:camera->ewaste",
        "Keyboards": "practical_remap:Keyboards->ewaste",
        "laptop": "practical_remap:laptop->ewaste",
        "microwave": "practical_remap:microwave->ewaste",
        "Mobile": "practical_remap:Mobile->ewaste",
        "Mouses": "practical_remap:Mouses->ewaste",
        "smartwatch": "practical_remap:smartwatch->ewaste",
        "TV": "practical_remap:TV->ewaste",
    },
)

WADE_AI = SourceConfig(
    name="wade_ai",
    source_url="https://github.com/letsdoitworld/wade-ai",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "trash_region": "trash",
    },
    class_notes={
        "trash_region": "practical_remap:street_litter_region->trash",
    },
    integration_mode="via_polygon_crops",
    via_annotation_globs=[
        "Trash_Detection/trash/dataset/train/*.json",
        "Trash_Detection/trash/dataset/val/*.json",
    ],
)

WASTEBENCH = SourceConfig(
    name="wastebench",
    source_url="https://huggingface.co/datasets/aliman8/WasteBench-Dataset",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "Rigid Plastic": "rigid_plastic",
        "Soft Plastic": "soft_plastic",
        "Cardboard": "paper",
        "Metal": "metal",
        "Glass": "glass",
        "Paper": "paper",
        "Trash": "trash",
    },
    class_notes={
        "Rigid Plastic": "direct_preserve:Rigid Plastic->rigid_plastic",
        "Soft Plastic": "direct_preserve:Soft Plastic->soft_plastic",
        "Cardboard": "practical_remap:Cardboard->paper",
        "Metal": "direct:metal",
        "Glass": "direct:glass",
        "Paper": "direct:paper",
        "Trash": "direct:trash",
    },
    parquet_class_column="category",
)

RECYCLE_NET = SourceConfig(
    name="recycle_net",
    source_url="https://github.com/SebastianCharmot/recycle_net",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
    },
    class_notes={
        "glass": "direct:glass",
        "metal": "direct:metal",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
    },
    integration_mode="coco_remote_crops",
    coco_annotation_files=[
        "DL_final_v3/train/_annotations.coco.json",
        "DL_final_v3/valid/_annotations.coco.json",
        "DL_final_v3/test/_annotations.coco.json",
    ],
)

GARBAGE_OBJECT_DETECTION = SourceConfig(
    name="garbage_object_detection",
    source_url="https://github.com/dfayzur/garbage-object-detection",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "cardboard": "paper",
        "garbage_bag": "trash",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "garbage_bag": "practical_remap:garbage_bag->trash",
    },
    integration_mode="voc_xml_crops",
    voc_xml_glob="garbage-object-detection/annotations/xmls/*.xml",
    voc_image_dir="garbage-object-detection/images",
)

COMPOSTNET = SourceConfig(
    name="compostnet",
    source_url="https://github.com/sarahmfrost/compostnet",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "cardboard": "paper",
        "compost": "organic",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
    },
    class_notes={
        "cardboard": "practical_remap:cardboard->paper",
        "compost": "practical_remap:compost->organic",
        "glass": "direct:glass",
        "metal": "direct:metal",
        "paper": "direct:paper",
        "plastic": "direct:plastic",
    },
    integration_mode="flat_prefixed_images",
    flat_image_dirs=["test", "train", "valid", "validation"],
)

TRASH_DETECTION_DATASET = SourceConfig(
    name="trash_detection_dataset",
    source_url="https://data.mendeley.com/datasets/z732f9pwxt/1",
    archive_name="dataset.zip",
    class_map={
        "BIODEGRADABLE": "organic",
        "CARDBOARD": "paper",
        "GLASS": "glass",
        "METAL": "metal",
        "PAPER": "paper",
        "PLASTIC": "plastic",
    },
    class_notes={
        "BIODEGRADABLE": "practical_remap:BIODEGRADABLE->organic",
        "CARDBOARD": "practical_remap:CARDBOARD->paper",
        "GLASS": "direct:glass",
        "METAL": "direct:metal",
        "PAPER": "direct:paper",
        "PLASTIC": "direct:plastic",
    },
    integration_mode="yolo_crops",
    yolo_class_names=["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "ALL"],
)

WASTEVISION = SourceConfig(
    name="wastevision",
    source_url="https://data.mendeley.com/datasets/mr67c82zw7/1",
    archive_name="dataset.zip",
    class_map={
        "Cable": "ewaste",
        "E-waste": "ewaste",
        "Glass": "glass",
        "Medical Waste": "medical",
        "Metal": "metal",
        "Plastics": "plastic",
        "can": "metal",
    },
    class_notes={
        "Cable": "practical_remap:Cable->ewaste",
        "E-waste": "practical_remap:E-waste->ewaste",
        "Glass": "direct:glass",
        "Medical Waste": "direct:medical",
        "Metal": "direct:metal",
        "Plastics": "practical_remap:Plastics->plastic",
        "can": "practical_remap:can->metal",
    },
    integration_mode="yolo_crops",
    yolo_class_names=["Cable", "E-waste", "Glass", "Medical Waste", "Metal", "Plastics", "can"],
)

RECODEWASTE = SourceConfig(
    name="recodewaste",
    source_url="https://github.com/prasadvineetv/ReCoDeWaste",
    archive_name="dataset_placeholder_unused_when_skip_extract",
    class_map={
        "aggregate": "aggregates",
        "cardboard": "paper",
        "hard plastic": "plastic",
        "metal": "metal",
        "soft plastic": "plastic",
        "timber": "wood",
    },
    class_notes={
        "aggregate": "practical_remap:aggregate->aggregates",
        "cardboard": "practical_remap:cardboard->paper",
        "hard plastic": "practical_remap:hard plastic->plastic",
        "metal": "direct:metal",
        "soft plastic": "practical_remap:soft plastic->plastic",
        "timber": "practical_remap:timber->wood",
    },
    integration_mode="coco_remote_crops",
    coco_annotation_files=[
        "1. Instance Segmentation /train/_annotations.coco.json",
        "1. Instance Segmentation /valid/_annotations.coco.json",
        "1. Instance Segmentation /test/_annotations.coco.json",
    ],
)

ZEROWASTE = SourceConfig(
    name="zerowaste",
    source_url="https://zenodo.org/records/6412647",
    archive_name="zerowaste-f-final.zip",
    class_map={
        "rigid_plastic": "rigid_plastic",
        "cardboard": "paper",
        "metal": "metal",
        "soft_plastic": "soft_plastic",
    },
    class_notes={
        "rigid_plastic": "direct_preserve:rigid_plastic",
        "cardboard": "practical_remap:cardboard->paper",
        "metal": "direct:metal",
        "soft_plastic": "direct_preserve:soft_plastic",
    },
    integration_mode="semseg_components",
    semseg_splits=["train", "val", "test"],
    semseg_image_dirname="data",
    semseg_mask_dirname="sem_seg",
    semseg_class_value_map={
        1: "rigid_plastic",
        2: "cardboard",
        3: "metal",
        4: "soft_plastic",
    },
)



KAGGLE_GENERIC_CLASS_MAP = {
    "battery": "battery",
    "biological": "organic",
    "cardboard": "paper",
    "clothes": "clothes",
    "ewaste": "ewaste",
    "glass": "glass",
    "medical": "medical",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "shoes": "shoes",
    "trash": "trash",
    "organic": "organic",
    "wood": "wood",
    "e-waste": "ewaste",
    "vegetation": "organic",
    "food organics": "organic",
    "food": "organic",
    "textile": "clothes",
    "apparel": "clothes",
    "cardboard_box": "paper"
}
KAGGLE_GENERIC_CLASS_NOTES = {k: "kaggle_generic_remap" for k in KAGGLE_GENERIC_CLASS_MAP.keys()}

def gen_kaggle_config(name):
    return SourceConfig(
        name=name,
        source_url="kaggle",
        archive_name="dataset_placeholder",
        class_map=KAGGLE_GENERIC_CLASS_MAP,
        class_notes=KAGGLE_GENERIC_CLASS_NOTES
    )

KAGGLE_GC_V2 = gen_kaggle_config("garbage_classification_v2")
KAGGLE_GC_YML = gen_kaggle_config("garbage_dataset_plusyaml")
KAGGLE_GC_MULTI = gen_kaggle_config("multi_class_garbage_classification_dataset")
KAGGLE_GC_HASS = gen_kaggle_config("garbage_classification")
KAGGLE_GC_HUB = gen_kaggle_config("garbage_classification_labels_corrections")
KAGGLE_ALISTAIR = gen_kaggle_config("recyclable_and_household_waste_classification")
KAGGLE_WASIF = gen_kaggle_config("custom_waste_classification_dataset")
KAGGLE_AAS = gen_kaggle_config("waste_segregation_image_dataset")
KAGGLE_PHEN = gen_kaggle_config("waste_classification")
KAGGLE_PREETI = gen_kaggle_config("waste_classificationorganic_and_recyclable")

SOURCE_CONFIGS = {
    KAGGLE_GC_V2.name: KAGGLE_GC_V2,
    KAGGLE_GC_YML.name: KAGGLE_GC_YML,
    KAGGLE_GC_MULTI.name: KAGGLE_GC_MULTI,
    KAGGLE_GC_HASS.name: KAGGLE_GC_HASS,
    KAGGLE_GC_HUB.name: KAGGLE_GC_HUB,
    KAGGLE_ALISTAIR.name: KAGGLE_ALISTAIR,
    KAGGLE_WASIF.name: KAGGLE_WASIF,
    KAGGLE_AAS.name: KAGGLE_AAS,
    KAGGLE_PHEN.name: KAGGLE_PHEN,
    KAGGLE_PREETI.name: KAGGLE_PREETI,
    REALWASTE.name: REALWASTE,
    TRASHNET.name: TRASHNET,
    ROOTSTRAP_WASTE_CLASSIFIER.name: ROOTSTRAP_WASTE_CLASSIFIER,
    RECYCLED_PORTLAND_STATE.name: RECYCLED_PORTLAND_STATE,
    MNEMORA_LITTER_SORT.name: MNEMORA_LITTER_SORT,
    DMEDHI_GARBAGE_IMAGE_CLASSIFICATION_DETECTION.name: DMEDHI_GARBAGE_IMAGE_CLASSIFICATION_DETECTION,
    OPENRECYCLE.name: OPENRECYCLE,
    TRASHBOX.name: TRASHBOX,
    MMCDWASTE.name: MMCDWASTE,
    METAL_SCRAP_DATASET.name: METAL_SCRAP_DATASET,
    TACO.name: TACO,
    GREENSORTER.name: GREENSORTER,
    RECYCLE_DETECTOR.name: RECYCLE_DETECTOR,
    EWASTENET.name: EWASTENET,
    WADE_AI.name: WADE_AI,
    WASTEBENCH.name: WASTEBENCH,
    RECYCLE_NET.name: RECYCLE_NET,
    GARBAGE_OBJECT_DETECTION.name: GARBAGE_OBJECT_DETECTION,
    COMPOSTNET.name: COMPOSTNET,
    TRASH_DETECTION_DATASET.name: TRASH_DETECTION_DATASET,
    WASTEVISION.name: WASTEVISION,
    RECODEWASTE.name: RECODEWASTE,
    ZEROWASTE.name: ZEROWASTE,
}


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate an external waste dataset into Dataset_Final.")
    parser.add_argument("--source", choices=sorted(SOURCE_CONFIGS), required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset_Final"))
    parser.add_argument("--workspace-root", type=Path, default=Path("external_datasets"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-split-ratios", default="0.7,0.2,0.1")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_auto_split_ratios(spec: str) -> tuple[float, float, float]:
    values = [float(chunk.strip()) for chunk in spec.split(",") if chunk.strip()]
    if len(values) != 3:
        raise ValueError("--auto-split-ratios must contain exactly three comma-separated values.")
    if any(value <= 0 for value in values):
        raise ValueError("--auto-split-ratios values must all be positive.")
    total = sum(values)
    if total <= 0:
        raise ValueError("--auto-split-ratios sum must be positive.")
    return (values[0] / total, values[1] / total, values[2] / total)


def allocate_split_counts(total_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total_count <= 0:
        return (0, 0, 0)
    if total_count == 1:
        return (1, 0, 0)
    if total_count == 2:
        return (1, 1, 0)
    if total_count == 3:
        return (1, 1, 1)

    minimums = [1, 1, 1]
    remaining = total_count - sum(minimums)
    raw = [remaining * ratio for ratio in ratios]
    extras = [int(value) for value in raw]
    shortfall = remaining - sum(extras)
    remainders = sorted(
        ((raw[index] - extras[index], index) for index in range(3)),
        reverse=True,
    )
    for _, index in remainders[:shortfall]:
        extras[index] += 1
    return tuple(minimums[index] + extras[index] for index in range(3))


def regenerate_auto_split_manifest(dataset_root: Path, seed: int, auto_split_ratios: str) -> dict[str, object]:
    ratios = parse_auto_split_ratios(auto_split_ratios)
    physical_classes = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    alias_map = {"soft_plastic": "plastic", "rigid_plastic": "plastic"}
    by_class: dict[str, list[str]] = {}
    for class_name in physical_classes:
        target_name = alias_map.get(class_name, class_name)
        if target_name not in by_class:
            by_class[target_name] = []
        files = sorted(str(path) for path in (dataset_root / class_name).iterdir() if is_image(path))
        by_class[target_name].extend(files)
        
    classes = sorted(list(by_class.keys()))

    rng = random.Random(seed)
    split_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    train_total = 0
    val_total = 0
    test_total = 0
    for class_name, files in by_class.items():
        shuffled = list(files)
        rng.shuffle(shuffled)
        train_count, val_count, test_count = allocate_split_counts(len(shuffled), ratios)
        split_counts["train"][class_name] = train_count
        split_counts["val"][class_name] = val_count
        split_counts["test"][class_name] = test_count
        train_total += train_count
        val_total += val_count
        test_total += test_count

    manifest = {
        "dataset_root": str(dataset_root),
        "split_mode": "auto_stratified_from_flat_root",
        "seed": int(seed),
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "class_names": classes,
        "split_counts": split_counts,
        "source_samples": sum(len(files) for files in by_class.values()),
        "train_samples": train_total,
        "val_samples": val_total,
        "test_samples": test_total,
    }
    save_json(dataset_root / "auto_split_manifest.json", manifest)
    return manifest


def ensure_extracted(config: SourceConfig, workspace_root: Path, skip_extract: bool) -> Path:
    source_root = workspace_root / config.name
    extracted_root = source_root / "extracted"
    if skip_extract and extracted_root.exists():
        materialize_source_specific_layout(config, extracted_root)
        return extracted_root
    archive_path = source_root / config.archive_name
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}. Download it first from {config.source_url}, "
            f"or populate {extracted_root} and rerun with --skip-extract."
        )
    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)
    archive_name = archive_path.name.lower()
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted_root)
    elif archive_name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path) as archive:
            archive.extractall(extracted_root)
    else:
        raise ValueError(f"Unsupported archive format for {archive_path}")
    materialize_source_specific_layout(config, extracted_root)
    return extracted_root


def materialize_source_specific_layout(config: SourceConfig, extracted_root: Path) -> None:
    if config.name == "zerowaste":
        candidate_roots = [
            extracted_root / "zerowaste-f-final",
            extracted_root / "ZeroWaste-f-final",
            extracted_root / "splits_final_deblurred",
            extracted_root / "data",
            extracted_root,
        ]
        for candidate in candidate_roots:
            if all((candidate / split).exists() for split in ("train", "val", "test")):
                for split in ("train", "val", "test"):
                    split_root = candidate / split
                    if not (extracted_root / split).exists():
                        try:
                            (extracted_root / split).symlink_to(split_root, target_is_directory=True)
                        except FileExistsError:
                            pass
                break

    materialized_npz_root = extracted_root / "materialized_from_npz"
    if config.label_index_map is not None:
        npz_files = sorted(extracted_root.rglob("*.npz"))
        if npz_files and not materialized_npz_root.exists():
            npz_path = npz_files[0]
            try:
                import numpy as np
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"{config.name} requires numpy to materialize {npz_path.name}. "
                    "Run this importer with the repo venv Python."
                ) from exc

            data = np.load(npz_path, allow_pickle=True)
            split_pairs = []
            if "x_train" in data and "y_train" in data:
                split_pairs.append(("train", data["x_train"], data["y_train"].reshape(-1)))
            if "x_test" in data and "y_test" in data:
                split_pairs.append(("test", data["x_test"], data["y_test"].reshape(-1)))
            if not split_pairs:
                raise RuntimeError(f"Unsupported npz layout in {npz_path}")

            materialized_npz_root.mkdir(parents=True, exist_ok=True)
            for split_name, images, labels in split_pairs:
                for index, (image_array, label_index) in enumerate(zip(images, labels, strict=True), start=1):
                    if int(label_index) not in config.label_index_map:
                        continue
                    source_class = config.label_index_map[int(label_index)]
                    output_dir = materialized_npz_root / split_name / source_class
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{source_class}_{split_name}_{index:05d}.png"
                    Image.fromarray(image_array).save(output_path)

    materialized_parquet_root = extracted_root / "materialized_from_parquet"
    if config.parquet_class_column is not None:
        parquet_files = sorted(extracted_root.rglob("*.parquet"))
        if parquet_files and not materialized_parquet_root.exists():
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"{config.name} requires pyarrow to materialize parquet image rows. "
                    "Run this importer with the repo venv Python."
                ) from exc

            materialized_parquet_root.mkdir(parents=True, exist_ok=True)
            for parquet_path in parquet_files:
                table = pq.read_table(parquet_path)
                records = table.to_pylist()
                split_name = parquet_path.stem.split("-")[0]
                split_counts = Counter()
                for row in records:
                    source_class = row.get(config.parquet_class_column)
                    if source_class not in config.class_map:
                        continue
                    image_payload = row.get("image")
                    if not isinstance(image_payload, dict):
                        continue
                    image_bytes = image_payload.get("bytes")
                    if not image_bytes:
                        continue
                    output_dir = materialized_parquet_root / split_name / str(source_class)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    split_counts[str(source_class)] += 1
                    source_path = image_payload.get("path") or f"{source_class}_{split_counts[str(source_class)]:05d}.jpg"
                    suffix = Path(str(source_path)).suffix or ".jpg"
                    stem = slugify(Path(str(source_path)).stem) or f"{slugify(str(source_class))}_{split_counts[str(source_class)]:05d}"
                    output_path = output_dir / f"{stem}{suffix}"
                    if output_path.exists():
                        output_path = next_unique_destination(output_dir, output_path.name)
                    output_path.write_bytes(image_bytes)


def locate_source_class_dirs(extracted_root: Path, expected_classes: set[str]) -> dict[str, list[Path]]:
    expected_by_key = {class_name.strip().lower(): class_name for class_name in expected_classes}
    matches: dict[str, list[Path]] = defaultdict(list)
    for directory in sorted(path for path in extracted_root.rglob("*") if path.is_dir()):
        name = directory.name.strip()
        key = name.lower()
        if key not in expected_by_key:
            continue
        if any(is_image(child) for child in directory.iterdir()):
            matches[expected_by_key[key]].append(directory)
    missing = expected_classes.difference(matches)
    if missing:
        print(f"Warning: Could not locate class directories for: {sorted(missing)} under {extracted_root}")
    return matches


def load_json_list(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON in {path}")
    return data


def load_json_dict(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object JSON in {path}")
    return data


def next_unique_destination(destination_dir: Path, base_name: str) -> Path:
    candidate = destination_dir / base_name
    if not candidate.exists():
        return candidate
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    index = 2
    while True:
        candidate = destination_dir / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def normalized_crop_box(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
) -> tuple[int, int, int, int] | None:
    left = int(round((center_x - box_width / 2) * width))
    top = int(round((center_y - box_height / 2) * height))
    right = int(round((center_x + box_width / 2) * width))
    bottom = int(round((center_y + box_height / 2) * height))
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def xywh_crop_box(width: int, height: int, bbox: list[float]) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = bbox
    left = int(round(x))
    top = int(round(y))
    right = int(round(x + w))
    bottom = int(round(y + h))
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def integrate_yolo_crops(
    config: SourceConfig,
    dataset_root: Path,
    extracted_root: Path,
    dry_run: bool,
    metadata: list[dict[str, object]],
) -> tuple[list[dict[str, object]], Counter, Counter, int, int, dict[str, object]]:
    existing_source_keys = {
        (str(row.get("external_source")), str(row.get("original_file_path")))
        for row in metadata
        if row.get("external_source")
    }
    added_entries: list[dict[str, object]] = []
    added_by_target = Counter()
    added_by_source = Counter()
    skipped_existing = 0
    recovered_existing = 0
    assert config.yolo_class_names is not None

    for split_name in ("train", "valid", "test"):
        image_dir = extracted_root / split_name / "images"
        label_dir = extracted_root / split_name / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = next((candidate for candidate in image_dir.glob(f"{label_path.stem}.*") if is_image(candidate)), None)
            if image_path is None:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                for object_index, line in enumerate(lines, start=1):
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id = int(float(parts[0]))
                    if class_id < 0 or class_id >= len(config.yolo_class_names):
                        continue
                    source_class = config.yolo_class_names[class_id]
                    if source_class not in config.class_map:
                        continue
                    crop_box = normalized_crop_box(
                        width,
                        height,
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                    if crop_box is None:
                        continue
                    original_ref = f"{split_name}/{image_path.name}#obj{object_index}:{source_class}"
                    original_key = (config.name, original_ref)
                    if original_key in existing_source_keys:
                        skipped_existing += 1
                        continue
                    target_class = config.class_map[source_class]
                    target_dir = dataset_root / target_class
                    out_name = f"{slugify(config.name)}__{slugify(split_name)}__{slugify(image_path.stem)}__obj{object_index:03d}.jpg"
                    canonical_destination = target_dir / out_name
                    if not dry_run:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        if canonical_destination.exists():
                            destination_path = canonical_destination
                            recovered_existing += 1
                        else:
                            crop = image.crop(crop_box)
                            destination_path = canonical_destination
                            crop.save(destination_path, quality=95)
                    else:
                        destination_path = canonical_destination
                    added_entries.append(
                        {
                            "file_path": str(destination_path.relative_to(dataset_root.parent)),
                            "label": target_class,
                            "source_label": source_class,
                            "original_file_path": original_ref,
                            "mapping_rule": config.class_notes[source_class],
                            "external_source": config.name,
                            "external_source_url": config.source_url,
                        }
                    )
                    added_by_target[target_class] += 1
                    added_by_source[source_class] += 1
    return (added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, {})


def integrate_flat_prefixed_images(
    config: SourceConfig,
    dataset_root: Path,
    extracted_root: Path,
    dry_run: bool,
    metadata: list[dict[str, object]],
) -> tuple[list[dict[str, object]], Counter, Counter, int, int, dict[str, object]]:
    existing_source_keys = {
        (str(row.get("external_source")), str(row.get("original_file_path")))
        for row in metadata
        if row.get("external_source")
    }
    added_entries: list[dict[str, object]] = []
    added_by_target = Counter()
    added_by_source = Counter()
    skipped_existing = 0
    recovered_existing = 0

    prefixes = sorted(config.class_map, key=len, reverse=True)
    search_roots = [extracted_root]
    if config.flat_image_dirs:
        search_roots = [extracted_root / rel for rel in config.flat_image_dirs if (extracted_root / rel).exists()]

    seen_source_paths: set[Path] = set()
    for search_root in search_roots:
        for source_file in sorted(path for path in search_root.rglob("*") if is_image(path)):
            if source_file in seen_source_paths:
                continue
            seen_source_paths.add(source_file)
            lower_name = source_file.name.lower()
            source_class = next((prefix for prefix in prefixes if lower_name.startswith(prefix.lower())), None)
            if source_class is None:
                continue
            original_rel = str(source_file.relative_to(extracted_root))
            original_key = (config.name, original_rel)
            if original_key in existing_source_keys:
                skipped_existing += 1
                continue
            target_class = config.class_map[source_class]
            target_dir = dataset_root / target_class
            relative_parent = slugify(str(source_file.parent.relative_to(extracted_root)))
            file_name = f"{slugify(config.name)}__{relative_parent}__{source_file.name}"
            canonical_destination = target_dir / file_name
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                if canonical_destination.exists():
                    destination_path = canonical_destination
                    recovered_existing += 1
                else:
                    destination_path = canonical_destination
                    shutil.copy2(source_file, destination_path)
            else:
                destination_path = canonical_destination
            added_entries.append(
                {
                    "file_path": str(destination_path.relative_to(dataset_root.parent)),
                    "label": target_class,
                    "source_label": source_class,
                    "original_file_path": original_rel,
                    "mapping_rule": config.class_notes[source_class],
                    "external_source": config.name,
                    "external_source_url": config.source_url,
                }
            )
            added_by_target[target_class] += 1
            added_by_source[source_class] += 1

    return (added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, {})


def integrate_coco_remote_crops(
    config: SourceConfig,
    dataset_root: Path,
    extracted_root: Path,
    dry_run: bool,
    metadata: list[dict[str, object]],
) -> tuple[list[dict[str, object]], Counter, Counter, int, int, dict[str, object]]:
    existing_source_keys = {
        (str(row.get("external_source")), str(row.get("original_file_path")))
        for row in metadata
        if row.get("external_source")
    }
    added_entries: list[dict[str, object]] = []
    added_by_target = Counter()
    added_by_source = Counter()
    skipped_existing = 0
    recovered_existing = 0
    failed_fetches = 0
    failed_fetch_examples: list[str] = []
    image_cache: dict[str, Image.Image] = {}
    categories_by_id: dict[int, str] = {}
    assert config.coco_annotation_files is not None

    for rel_path in config.coco_annotation_files:
        payload = json.loads((extracted_root / rel_path).read_text(encoding="utf-8"))
        categories_by_id.update({int(item["id"]): str(item["name"]) for item in payload.get("categories", [])})

    for rel_path in config.coco_annotation_files:
        split_name = Path(rel_path).stem
        annotation_path = extracted_root / rel_path
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        images_by_id = {int(item["id"]): item for item in payload.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image_id, image_info in sorted(images_by_id.items()):
            annotations = annotations_by_image.get(image_id, [])
            if not annotations:
                continue
            remote_url = (
                image_info.get("coco_url")
                or image_info.get("flickr_640_url")
                or image_info.get("flickr_url")
            )
            local_image_path = annotation_path.parent / str(image_info.get("file_name", ""))
            if not remote_url:
                if not is_image(local_image_path):
                    continue
            prepared: list[tuple[dict[str, object], str, str, str, tuple[str, str], Path]] = []
            for annotation in annotations:
                category_id = int(annotation["category_id"])
                source_class = categories_by_id.get(category_id)
                if source_class not in config.class_map:
                    continue
                original_ref = f"{split_name}/{image_info.get('file_name', image_id)}#ann{annotation['id']}:{source_class}"
                original_key = (config.name, original_ref)
                target_class = config.class_map[source_class]
                target_dir = dataset_root / target_class
                out_name = f"{slugify(config.name)}__{slugify(split_name)}__{slugify(Path(str(image_info.get('file_name', image_id))).stem)}__ann{int(annotation['id']):05d}.jpg"
                canonical_destination = target_dir / out_name
                prepared.append((annotation, source_class, target_class, original_ref, original_key, canonical_destination))
            if not prepared:
                continue

            needs_remote_image = any(
                original_key not in existing_source_keys and not canonical_destination.exists()
                for _, _, _, _, original_key, canonical_destination in prepared
            )
            image = None
            image_opened_from_local = False
            width = 0
            height = 0
            if needs_remote_image:
                if remote_url:
                    if remote_url not in image_cache:
                        try:
                            response = requests.get(str(remote_url), timeout=(15, 60))
                            response.raise_for_status()
                            image_cache[remote_url] = Image.open(BytesIO(response.content)).convert("RGB")
                        except Exception:
                            failed_fetches += 1
                            if len(failed_fetch_examples) < 20:
                                failed_fetch_examples.append(str(remote_url))
                            continue
                    image = image_cache[remote_url]
                else:
                    try:
                        image = Image.open(local_image_path).convert("RGB")
                        image_opened_from_local = True
                    except Exception:
                        failed_fetches += 1
                        if len(failed_fetch_examples) < 20:
                            failed_fetch_examples.append(str(local_image_path))
                        continue
                width, height = image.size

            for annotation, source_class, target_class, original_ref, original_key, canonical_destination in prepared:
                if original_key in existing_source_keys:
                    skipped_existing += 1
                    continue
                if not dry_run:
                    canonical_destination.parent.mkdir(parents=True, exist_ok=True)
                    if canonical_destination.exists():
                        destination_path = canonical_destination
                        recovered_existing += 1
                    else:
                        if image is None:
                            continue
                        crop_box = xywh_crop_box(width, height, annotation.get("bbox", []))
                        if crop_box is None:
                            continue
                        crop = image.crop(crop_box)
                        destination_path = canonical_destination
                        crop.save(destination_path, quality=95)
                else:
                    destination_path = canonical_destination
                added_entries.append(
                    {
                        "file_path": str(destination_path.relative_to(dataset_root.parent)),
                        "label": target_class,
                        "source_label": source_class,
                        "original_file_path": original_ref,
                        "mapping_rule": config.class_notes[source_class],
                        "external_source": config.name,
                        "external_source_url": config.source_url,
                    }
                )
                added_by_target[target_class] += 1
                added_by_source[source_class] += 1
            if image_opened_from_local and image is not None:
                image.close()
    for image in image_cache.values():
        image.close()
    return (
        added_entries,
        added_by_target,
        added_by_source,
        skipped_existing,
        recovered_existing,
        {
            "failed_fetches": failed_fetches,
            "failed_fetch_examples": failed_fetch_examples,
        },
    )


def polygon_bbox(points_x: list[float], points_y: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if not points_x or not points_y or len(points_x) != len(points_y):
        return None
    left = max(0, min(int(min(points_x)), width - 1))
    top = max(0, min(int(min(points_y)), height - 1))
    right = max(left + 1, min(int(max(points_x)) + 1, width))
    bottom = max(top + 1, min(int(max(points_y)) + 1, height))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def extract_mask_component_boxes(mask: Image.Image, class_value: int) -> list[tuple[int, int, int, int]]:
    import numpy as np
    import cv2
    img_array = np.array(mask.convert("L"))
    binary = (img_array == class_value).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    boxes = []
    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        right = left + width
        bottom = top + height
        if right > left and bottom > top:
            boxes.append((int(left), int(top), int(right), int(bottom)))
    return boxes


def integrate_semseg_components(
    config: SourceConfig,
    dataset_root: Path,
    extracted_root: Path,
    dry_run: bool,
    metadata: list[dict[str, object]],
) -> tuple[list[dict[str, object]], Counter, Counter, int, int, dict[str, object]]:
    existing_source_keys = {
        (str(row.get("external_source")), str(row.get("original_file_path")))
        for row in metadata
        if row.get("external_source")
    }
    added_entries: list[dict[str, object]] = []
    added_by_target = Counter()
    added_by_source = Counter()
    skipped_existing = 0
    recovered_existing = 0
    component_boxes_total = 0
    assert config.semseg_splits is not None
    assert config.semseg_image_dirname is not None
    assert config.semseg_mask_dirname is not None
    assert config.semseg_class_value_map is not None

    for split_name in config.semseg_splits:
        split_root = extracted_root / split_name
        image_dir = split_root / config.semseg_image_dirname
        mask_dir = split_root / config.semseg_mask_dirname
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in sorted(path for path in mask_dir.iterdir() if is_image(path)):
            image_path = image_dir / mask_path.name
            if not is_image(image_path):
                alt_matches = [candidate for candidate in image_dir.glob(f"{mask_path.stem}.*") if is_image(candidate)]
                if not alt_matches:
                    continue
                image_path = alt_matches[0]
            with Image.open(mask_path) as mask, Image.open(image_path) as image:
                mask = mask.convert("L")
                image = image.convert("RGB")
                for class_value, source_class in sorted(config.semseg_class_value_map.items()):
                    if source_class not in config.class_map:
                        continue
                    boxes = extract_mask_component_boxes(mask, class_value)
                    for component_index, crop_box in enumerate(boxes, start=1):
                        component_boxes_total += 1
                        original_ref = (
                            f"{split_name}/{image_path.name}#mask{class_value}_obj{component_index}:{source_class}"
                        )
                        original_key = (config.name, original_ref)
                        if original_key in existing_source_keys:
                            skipped_existing += 1
                            continue
                        target_class = config.class_map[source_class]
                        target_dir = dataset_root / target_class
                        out_name = (
                            f"{slugify(config.name)}__{slugify(split_name)}__"
                            f"{slugify(image_path.stem)}__mask{class_value}_obj{component_index:03d}.jpg"
                        )
                        canonical_destination = target_dir / out_name
                        if not dry_run:
                            target_dir.mkdir(parents=True, exist_ok=True)
                            if canonical_destination.exists():
                                destination_path = canonical_destination
                                recovered_existing += 1
                            else:
                                crop = image.crop(crop_box)
                                destination_path = canonical_destination
                                crop.save(destination_path, quality=95)
                        else:
                            destination_path = canonical_destination
                        added_entries.append(
                            {
                                "file_path": str(destination_path.relative_to(dataset_root.parent)),
                                "label": target_class,
                                "source_label": source_class,
                                "original_file_path": original_ref,
                                "mapping_rule": config.class_notes[source_class],
                                "external_source": config.name,
                                "external_source_url": config.source_url,
                            }
                        )
                        added_by_target[target_class] += 1
                        added_by_source[source_class] += 1

    return (
        added_entries,
        added_by_target,
        added_by_source,
        skipped_existing,
        recovered_existing,
        {"component_boxes_total": component_boxes_total},
    )


def integrate_via_polygon_crops(
    config: SourceConfig,
    dataset_root: Path,
    extracted_root: Path,
    dry_run: bool,
    metadata: list[dict[str, object]],
) -> tuple[list[dict[str, object]], Counter, Counter, int, int, dict[str, object]]:
    existing_source_keys = {
        (str(row.get("external_source")), str(row.get("original_file_path")))
        for row in metadata
        if row.get("external_source")
    }
    added_entries: list[dict[str, object]] = []
    added_by_target = Counter()
    added_by_source = Counter()
    skipped_existing = 0
    recovered_existing = 0
    assert config.via_annotation_globs is not None
    source_class = next(iter(config.class_map))
    target_class = config.class_map[source_class]

    annotation_paths: list[Path] = []
    for pattern in config.via_annotation_globs:
        annotation_paths.extend(sorted(extracted_root.glob(pattern)))

    for annotation_path in annotation_paths:
        split_name = annotation_path.parent.name
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_metadata = payload.get("_via_img_metadata", {})
        if not isinstance(image_metadata, dict):
            continue
        for image_record in image_metadata.values():
            if not isinstance(image_record, dict):
                continue
            image_name = image_record.get("filename")
            if not image_name:
                continue
            image_path = annotation_path.parent / str(image_name)
            if not is_image(image_path):
                continue
            regions = image_record.get("regions", [])
            if not isinstance(regions, list) or not regions:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                for object_index, region in enumerate(regions, start=1):
                    if not isinstance(region, dict):
                        continue
                    shape = region.get("shape_attributes", {})
                    if not isinstance(shape, dict):
                        continue
                    points_x = shape.get("all_points_x")
                    points_y = shape.get("all_points_y")
                    if not isinstance(points_x, list) or not isinstance(points_y, list):
                        continue
                    crop_box = polygon_bbox(points_x, points_y, width, height)
                    if crop_box is None:
                        continue
                    original_ref = f"{annotation_path.relative_to(extracted_root)}::{image_name}#obj{object_index}:trash_region"
                    original_key = (config.name, original_ref)
                    if original_key in existing_source_keys:
                        skipped_existing += 1
                        continue
                    target_dir = dataset_root / target_class
                    out_name = f"{slugify(config.name)}__{slugify(split_name)}__{slugify(Path(str(image_name)).stem)}__obj{object_index:03d}.jpg"
                    canonical_destination = target_dir / out_name
                    if not dry_run:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        if canonical_destination.exists():
                            destination_path = canonical_destination
                            recovered_existing += 1
                        else:
                            crop = image.crop(crop_box)
                            destination_path = canonical_destination
                            crop.save(destination_path, quality=95)
                    else:
                        destination_path = canonical_destination
                    added_entries.append(
                        {
                            "file_path": str(destination_path.relative_to(dataset_root.parent)),
                            "label": target_class,
                            "source_label": source_class,
                            "original_file_path": original_ref,
                            "mapping_rule": config.class_notes[source_class],
                            "external_source": config.name,
                            "external_source_url": config.source_url,
                        }
                    )
                    added_by_target[target_class] += 1
                    added_by_source[source_class] += 1

    return (added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, {})


def integrate_voc_xml_crops(
    config: SourceConfig,
    dataset_root: Path,
    extracted_root: Path,
    dry_run: bool,
    metadata: list[dict[str, object]],
) -> tuple[list[dict[str, object]], Counter, Counter, int, int, dict[str, object]]:
    existing_source_keys = {
        (str(row.get("external_source")), str(row.get("original_file_path")))
        for row in metadata
        if row.get("external_source")
    }
    added_entries: list[dict[str, object]] = []
    added_by_target = Counter()
    added_by_source = Counter()
    skipped_existing = 0
    recovered_existing = 0
    assert config.voc_xml_glob is not None
    assert config.voc_image_dir is not None

    image_root = extracted_root / config.voc_image_dir
    for xml_path in sorted(extracted_root.glob(config.voc_xml_glob)):
        root = ET.parse(xml_path).getroot()
        image_name = root.findtext("filename")
        if not image_name:
            continue
        image_path = image_root / image_name
        if not is_image(image_path):
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            object_index = 0
            for obj in root.findall("object"):
                source_class = obj.findtext("name")
                if source_class not in config.class_map:
                    continue
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                try:
                    xmin = float(bnd.findtext("xmin", "0"))
                    ymin = float(bnd.findtext("ymin", "0"))
                    xmax = float(bnd.findtext("xmax", "0"))
                    ymax = float(bnd.findtext("ymax", "0"))
                except ValueError:
                    continue
                object_index += 1
                crop_box = xywh_crop_box(width, height, [xmin, ymin, xmax - xmin, ymax - ymin])
                if crop_box is None:
                    continue
                original_ref = f"{xml_path.relative_to(extracted_root)}::{image_name}#obj{object_index}:{source_class}"
                original_key = (config.name, original_ref)
                if original_key in existing_source_keys:
                    skipped_existing += 1
                    continue
                target_class = config.class_map[source_class]
                target_dir = dataset_root / target_class
                out_name = f"{slugify(config.name)}__{slugify(Path(image_name).stem)}__obj{object_index:03d}.jpg"
                canonical_destination = target_dir / out_name
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    if canonical_destination.exists():
                        destination_path = canonical_destination
                        recovered_existing += 1
                    else:
                        crop = image.crop(crop_box)
                        destination_path = canonical_destination
                        crop.save(destination_path, quality=95)
                else:
                    destination_path = canonical_destination
                added_entries.append(
                    {
                        "file_path": str(destination_path.relative_to(dataset_root.parent)),
                        "label": target_class,
                        "source_label": source_class,
                        "original_file_path": original_ref,
                        "mapping_rule": config.class_notes[source_class],
                        "external_source": config.name,
                        "external_source_url": config.source_url,
                    }
                )
                added_by_target[target_class] += 1
                added_by_source[source_class] += 1

    return (added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, {})


def integrate_source(
    config: SourceConfig,
    dataset_root: Path,
    extracted_root: Path,
    dry_run: bool,
) -> dict[str, object]:
    metadata_path = dataset_root / "dataset_metadata.json"
    report_path = dataset_root / "dataset_reorganization_report.json"
    imports_path = dataset_root / "external_dataset_imports.json"

    metadata = load_json_list(metadata_path)
    report = load_json_dict(report_path)
    existing_imports = load_json_dict(imports_path) if imports_path.exists() else {"imports": []}

    if config.integration_mode == "imagefolder":
        existing_source_keys = {
            (str(row.get("external_source")), str(row.get("original_file_path")))
            for row in metadata
            if row.get("external_source")
        }

        source_dirs = locate_source_class_dirs(extracted_root, set(config.class_map))
        added_entries: list[dict[str, object]] = []
        added_by_target = Counter()
        added_by_source = Counter()
        skipped_existing = 0
        recovered_existing = 0
        extra_stats: dict[str, object] = {}

        for source_class, source_class_dirs in sorted(source_dirs.items()):
            target_class = config.class_map[source_class]
            target_dir = dataset_root / target_class
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
            for source_dir in source_class_dirs:
                for source_file in sorted(path for path in source_dir.iterdir() if is_image(path)):
                    original_key = (config.name, str(source_file.relative_to(extracted_root)))
                    if original_key in existing_source_keys:
                        skipped_existing += 1
                        continue
                    relative_parent = slugify(str(source_file.parent.relative_to(extracted_root)))
                    file_name = f"{slugify(config.name)}__{relative_parent}__{source_file.name}"
                    canonical_destination = target_dir / file_name
                    if not dry_run:
                        if canonical_destination.exists():
                            destination_path = canonical_destination
                            recovered_existing += 1
                        else:
                            destination_path = canonical_destination
                            shutil.copy2(source_file, destination_path)
                    else:
                        destination_path = canonical_destination
                    entry = {
                        "file_path": str(destination_path.relative_to(dataset_root.parent)),
                        "label": target_class,
                        "source_label": source_class,
                        "original_file_path": str(source_file.relative_to(extracted_root)),
                        "mapping_rule": config.class_notes[source_class],
                        "external_source": config.name,
                        "external_source_url": config.source_url,
                    }
                    added_entries.append(entry)
                    added_by_target[target_class] += 1
                    added_by_source[source_class] += 1
    elif config.integration_mode == "yolo_crops":
        added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, extra_stats = integrate_yolo_crops(
            config, dataset_root, extracted_root, dry_run, metadata
        )
    elif config.integration_mode == "flat_prefixed_images":
        added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, extra_stats = integrate_flat_prefixed_images(
            config, dataset_root, extracted_root, dry_run, metadata
        )
    elif config.integration_mode == "coco_remote_crops":
        added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, extra_stats = integrate_coco_remote_crops(
            config, dataset_root, extracted_root, dry_run, metadata
        )
    elif config.integration_mode == "via_polygon_crops":
        added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, extra_stats = integrate_via_polygon_crops(
            config, dataset_root, extracted_root, dry_run, metadata
        )
    elif config.integration_mode == "voc_xml_crops":
        added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, extra_stats = integrate_voc_xml_crops(
            config, dataset_root, extracted_root, dry_run, metadata
        )
    elif config.integration_mode == "semseg_components":
        added_entries, added_by_target, added_by_source, skipped_existing, recovered_existing, extra_stats = integrate_semseg_components(
            config, dataset_root, extracted_root, dry_run, metadata
        )
    else:
        raise ValueError(f"Unsupported integration mode: {config.integration_mode}")

    if dry_run:
        return {
            "source_name": config.name,
            "source_url": config.source_url,
            "added_entries": len(added_entries),
            "added_by_target_class": dict(sorted(added_by_target.items())),
            "added_by_source_class": dict(sorted(added_by_source.items())),
            "skipped_existing": skipped_existing,
            "recovered_existing": recovered_existing,
            **extra_stats,
        }

    metadata.extend(added_entries)
    metadata.sort(key=lambda row: str(row["file_path"]))
    save_json(metadata_path, metadata)

    current_class_counts = {
        class_dir.name: sum(1 for path in class_dir.iterdir() if is_image(path))
        for class_dir in sorted(dataset_root.iterdir())
        if class_dir.is_dir()
    }

    import_record = {
        "source_name": config.name,
        "source_url": config.source_url,
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "added_entries": len(added_entries),
        "added_by_target_class": dict(sorted(added_by_target.items())),
        "added_by_source_class": dict(sorted(added_by_source.items())),
        "skipped_existing": skipped_existing,
        "recovered_existing": recovered_existing,
        **extra_stats,
        "class_map": config.class_map,
    }

    imports = list(existing_imports.get("imports", []))
    imports.append(import_record)
    save_json(imports_path, {"imports": imports})

    report["final_class_counts"] = current_class_counts
    report["final_classes"] = sorted(current_class_counts)
    report["total_images"] = int(sum(current_class_counts.values()))
    external_dataset_imports = list(report.get("external_dataset_imports", []))
    external_dataset_imports.append(import_record)
    report["external_dataset_imports"] = external_dataset_imports
    save_json(report_path, report)

    return import_record


def main() -> int:
    args = parse_args()
    config = SOURCE_CONFIGS[args.source]
    dataset_root = args.dataset_root.resolve()
    workspace_root = args.workspace_root.resolve()
    extracted_root = ensure_extracted(config, workspace_root, args.skip_extract)

    result = integrate_source(config, dataset_root, extracted_root, args.dry_run)
    if not args.dry_run:
        manifest = regenerate_auto_split_manifest(dataset_root, args.seed, args.auto_split_ratios)
        result["auto_split_manifest"] = {
            "source_samples": manifest["source_samples"],
            "train_samples": manifest["train_samples"],
            "val_samples": manifest["val_samples"],
            "test_samples": manifest["test_samples"],
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
