# Dataset Class Overlap Survey

Current local target taxonomy in this repo:

- `metal`
- `organic`
- `other`
- `paper`

Remap assumptions used in this survey:

- `cardboard`, `carton`, `paper cup`, `tetra pak` -> `paper`
- `trash`, `miscellaneous trash`, `general trash`, `non-recyclable`, `miscellaneous` -> `other`
- `biological`, `biodegradable`, `food organics`, `bio`, `compost` -> `organic`
- `aluminum`, `aluminium cans`, `cans` -> `metal` when the source dataset uses cans as a material class

Official library checks:

- TorchVision built-in datasets: no native waste/trash dataset found
  - Source: https://docs.pytorch.org/vision/stable/datasets.html
- TensorFlow Datasets built-in/community catalogs: no native waste/trash dataset found
  - Sources:
    - https://www.tensorflow.org/datasets/catalog
    - https://www.tensorflow.org/datasets/community_catalog/overview

## High-Value Datasets

### RealWaste

- Source: https://archive.ics.uci.edu/dataset/908/realwaste
- Classes:
  - `cardboard`
  - `food organics`
  - `glass`
  - `metal`
  - `miscellaneous trash`
  - `paper`
  - `plastic`
  - `textile trash`
  - `vegetation`
- Overlap:
  - exact: `metal`, `paper`
  - close: `food organics -> organic`, `miscellaneous trash -> other`, `cardboard -> paper`

### WasteNet

- Source: https://arxiv.org/abs/2006.05873
- Classes:
  - `paper`
  - `cardboard`
  - `glass`
  - `metal`
  - `plastic`
  - `other`
- Overlap:
  - exact: `metal`, `other`, `paper`
  - close: `cardboard -> paper`

### TrashNet

- Sources:
  - https://github.com/garythung/trashnet
  - https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Classes:
  - `glass`
  - `paper`
  - `cardboard`
  - `plastic`
  - `metal`
  - `trash`
- Overlap:
  - exact: `metal`, `paper`
  - close: `trash -> other`, `cardboard -> paper`

### TACO / Pomerania 7-class remap

- Sources:
  - official dataset repo: https://github.com/pedropro/TACO
  - 7-class remap paper: https://arxiv.org/abs/2105.06808
  - review summary: https://agamiko.github.io/waste-datasets-review/
- Official TACO annotations use 60 detailed categories in `data/annotations.json`.
- Practical 7-class remap cited in the paper:
  - `bio`
  - `glass`
  - `metal and plastic`
  - `non-recyclable`
  - `other`
  - `paper`
  - `unknown`
- Overlap:
  - exact: `paper`, `other`
  - close: `bio -> organic`, `metal and plastic` partially usable for `metal`

### Real-World Waste Classification Dataset (5 class)

- Source: https://www.kaggle.com/datasets/tuhanasinan/real-world-waste-classification-dataset-5-class
- Classes:
  - `metal`
  - `plastic`
  - `glass`
  - `paper`
  - `organic`
- Overlap:
  - exact: `metal`, `organic`, `paper`

### Garbage Dataset / Garbage Classification V2

- Sources:
  - https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
  - paper reference with listed classes: https://tecnoscientifica.com/journal/idwm/article/download/408/208/2771
- Classes:
  - `metal`
  - `glass`
  - `biological`
  - `paper`
  - `battery`
  - `trash`
  - `cardboard`
  - `shoes`
  - `clothes`
  - `plastic`
- Overlap:
  - exact: `metal`, `paper`
  - close: `biological -> organic`, `trash -> other`, `cardboard -> paper`

### Garbage_dataset_PlusYaml

- Source: https://www.kaggle.com/datasets/engrbasit62/garbage-dataset-plusyaml
- Classes:
  - `paper`
  - `plastic`
  - `glass`
  - `metal`
  - `organic`
  - `electronics`
  - `miscellaneous`
- Overlap:
  - exact: `metal`, `organic`, `paper`
  - close: `miscellaneous -> other`

### Multi class garbage classification Dataset

- Source: https://www.kaggle.com/datasets/vishallazrus/multi-class-garbage-classification-dataset
- Classes:
  - `cardboard`
  - `compost`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`
- Overlap:
  - exact: `metal`, `paper`
  - close: `compost -> organic`, `trash -> other`, `cardboard -> paper`

### Garbage Classification by hassnainzaidi

- Source: https://www.kaggle.com/datasets/hassnainzaidi/garbage-classification/data
- Classes:
  - `paper`
  - `cardboard`
  - `biological waste`
  - `metal`
  - `plastic`
  - `green glass`
  - `brown glass`
  - `white glass`
  - `clothing`
  - `shoes`
  - `batteries`
  - `general trash`
- Overlap:
  - exact: `metal`, `paper`
  - close: `biological waste -> organic`, `general trash -> other`, `cardboard -> paper`

## Strong Partial-Overlap Datasets

### CompostNet

- Source: https://github.com/sarahmfrost/compostnet
- Derived class set described in the repo:
  - `glass`
  - `paper`
  - `cardboard`
  - `plastic`
  - `metal`
  - `trash`
  - `compost`
- Overlap:
  - exact: `metal`, `paper`
  - close: `compost -> organic`, `trash -> other`, `cardboard -> paper`

### WastePro

- Source: https://www.kaggle.com/datasets/namanjain001/comprehensive-solid-waste-image-dataset
- Classes:
  - `organic`
  - `plastic`
  - `metal`
  - `glass`
  - `e-waste`
  - `paper`
  - `cardboard`
  - `textiles`
  - `rubber`
- Overlap:
  - exact: `metal`, `organic`, `paper`
  - close: `cardboard -> paper`

### Garbage Detection - 6 Waste Categories

- Source: https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection/data
- Classes:
  - `biodegradable`
  - `cardboard`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
- Overlap:
  - exact: `metal`, `paper`
  - close: `biodegradable -> organic`, `cardboard -> paper`

### VN Trash classification

- Source: https://www.kaggle.com/datasets/mrgetshjtdone/vn-trash-classification
- Classes:
  - `aluminum cans`
  - `carton`
  - `foam box`
  - `milk box`
  - `paper`
  - `paper cup`
  - `clear plastic cup`
  - `PET bottle`
  - `other trash`
- Overlap:
  - exact: `paper`
  - close: `aluminum cans -> metal`, `carton -> paper`, `paper cup -> paper`, `other trash -> other`

### Waste Segregation Image Dataset

- Source: https://www.kaggle.com/datasets/aashidutt3/waste-segregation-image-dataset
- Surfaced classes from public descriptions:
  - `paper`
  - `leaf`
  - `food`
  - `wood`
  - `waste`
  - `plastic bags`
  - `plastic bottles`
  - `metal cans`
- Overlap:
  - exact: `paper`
  - close: `food/leaf -> organic`, `waste -> other`, `metal cans -> metal`

## Narrower Or Auxiliary Datasets

### Waste Classification data

- Source: https://www.kaggle.com/datasets/techsash/waste-classification-data
- Classes:
  - `organic`
  - `recyclable`
- Overlap:
  - exact: `organic`

### Waste Classification Data v2

- Source: https://www.kaggle.com/datasets/sapal6/waste-classification-data-v2
- Classes:
  - `organic`
  - `recyclable`
  - `nonrecyclable`
- Overlap:
  - exact: `organic`
  - close: `nonrecyclable -> other`

### Waste Classification Dataset by phenomsg

- Source: https://www.kaggle.com/datasets/phenomsg/waste-classification
- Classes:
  - `hazardous`
  - `non-recyclable`
  - `organic`
  - `recyclable`
- Overlap:
  - exact: `organic`
  - close: `non-recyclable -> other`

### Waste classification (organic and recyclable)

- Source: https://www.kaggle.com/datasets/preetishah/waste-classificationorganic-and-recyclable
- Classes:
  - `organic`
  - `recyclable`
- Overlap:
  - exact: `organic`

### Image Waste Classification Dataset

- Source: https://www.kaggle.com/datasets/meetpatelp27/image-waste-classification-dataset
- Classes:
  - `organic waste`
  - `recyclable waste`
- Overlap:
  - exact: `organic`

### CNN: Waste Classification [Image] Dataset

- Source: https://www.kaggle.com/datasets/sujaykapadnis/cnn-waste-classification-image-dataset
- Classes:
  - `organic`
  - `recyclable`
- Overlap:
  - exact: `organic`

### AquaTrash

- Source: https://www.kaggle.com/datasets/harshpanwar/aquatrash
- Classes:
  - `glass`
  - `paper`
  - `metal`
  - `plastic`
- Overlap:
  - exact: `metal`, `paper`

### Garbage Classification by karansolanki01

- Source: https://www.kaggle.com/datasets/karansolanki01/garbage-classification
- Classes:
  - `battery`
  - `cardboard`
  - `clothes`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
- Overlap:
  - exact: `metal`, `paper`
  - close: `cardboard -> paper`

### Garbage Dataset Classification by zlatan599

- Source: https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification
- Classes:
  - `plastic`
  - `metal`
  - `glass`
  - `cardboard`
  - `paper`
  - `trash`
- Overlap:
  - exact: `metal`, `paper`
  - close: `trash -> other`, `cardboard -> paper`

### Trash Type Image Dataset by farzadnekouei

- Source: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
- Classes:
  - `cardboard`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`
- Overlap:
  - exact: `metal`, `paper`
  - close: `trash -> other`, `cardboard -> paper`

### TrashBox

- Source: https://github.com/nikhilvenkatkumsetty/TrashBox
- Classes:
  - `medical waste`
  - `e-waste`
  - `plastic`
  - `paper`
  - `metal`
  - `glass`
  - `cardboard`
- Overlap:
  - exact: `metal`, `paper`
  - close: `cardboard -> paper`

### ZeroWaste

- Source: https://github.com/Trash-AI/ZeroWaste
- Classes:
  - `soft plastic`
  - `rigid plastic`
  - `cardboard`
  - `metal`
- Overlap:
  - exact: `metal`
  - close: `cardboard -> paper`

### WasteVision / TrashNet++

- Source: https://data.mendeley.com/datasets/mr67c82zw7/1
- Classes:
  - `plastic`
  - `metal`
  - `glass`
  - `can`
  - `cable`
  - `e-waste`
  - `medical waste`
- Overlap:
  - exact: `metal`
  - close: `can -> metal`

### Drinking Waste Classification

- Source: https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification
- Classes:
  - `aluminium cans`
  - `glass bottles`
  - `PET bottles`
  - `HDPE milk bottles`
- Overlap:
  - close: `aluminium cans -> metal`

### EcoDetect Recyclable Waste Dataset

- Source: https://www.kaggle.com/datasets/ahsan71/ecodetect-recyclable-waste-detection-dataset
- Classes:
  - `plastic`
  - `paper`
  - `aluminum`
- Overlap:
  - exact: `paper`
  - close: `aluminum -> metal`

## Weak But Still Usable If Single-Class Match Is Enough

### TrashCan 1.0

- Source: https://conservancy.umn.edu/handle/11299/214865
- Main categories:
  - `bio`
  - `trash`
  - `unknown`
- Listed trash subitems include:
  - `paper`
  - `can`
  - `bottle`
  - `bag`
  - `cup`
  - `container`
  - `box`
  - `wire`
  - `bucket`
- Overlap:
  - close: `bio -> organic`, `paper -> paper`, `can -> metal`, `trash -> other`

### BeachLitter dataset v2022

- Source: https://doi.org/10.17882/85472
- Classes:
  - `bottle`
  - `can`
  - `carton`
  - `cup`
  - `other plastic`
  - `other plastic containers`
  - `plastic bag`
  - `wrapper`
- Overlap:
  - close: `can -> metal`, `carton -> paper`

### WaRP - Waste Recycling Plant Dataset

- Source: https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset
- Publicly surfaced grouped labels include:
  - `plastic bottles`
  - `glass bottles`
  - `card boards`
  - `detergents`
  - `canisters`
  - `cans`
- Overlap:
  - close: `card boards -> paper`, `cans -> metal`

## Practical Ranking

Best overall semantic fits for your taxonomy:

1. `RealWaste`
2. `Garbage_dataset_PlusYaml`
3. `Garbage Classification` by hassnainzaidi
4. `Multi class garbage classification Dataset`
5. `WasteNet`

Best for `metal + paper + other`:

1. `WasteNet`
2. `TrashNet`
3. `VN Trash classification`

Best for `metal + paper + organic`:

1. `Real-World Waste Classification Dataset (5 class)`
2. `WastePro`
3. `Garbage Dataset / GD`

Best narrow auxiliary sets:

- `Waste Classification data`
- `Waste Classification Data v2`
- `AquaTrash`
- `ZeroWaste`
- `WasteVision`

## Notes

- Kaggle contains many mirrors and reuploads of the same underlying dataset families, especially around `TrashNet`, `RealWaste`, and generic garbage-classification packs.
- This file intentionally keeps mirrored variants out unless they expose a distinct class taxonomy or are commonly used in practice.
- For your current project, `cardboard` should be merged into `paper`.
