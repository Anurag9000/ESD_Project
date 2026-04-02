# Dataset Research And Local Remap Notes

This file merges the previous dataset-overlap survey and the filename-based `other` remap notes into one project-specific reference.

## Current Local Dataset

Current local class taxonomy in this repo:

- `battery`
- `clothes`
- `ewaste`
- `glass`
- `metal`
- `organic`
- `paper`
- `plastic`
- `shoes`
- `trash`

Current local class counts in [Dataset_Final](/home/anurag-basistha/Projects/ESD/Dataset_Final):

- `paper`: `13444`
- `glass`: `9995`
- `organic`: `9007`
- `plastic`: `8099`
- `clothes`: `6260`
- `metal`: `5042`
- `shoes`: `2113`
- `battery`: `1627`
- `ewaste`: `446`
- `trash`: `130`

Notes:

- The old generic `other` class was removed.
- The old split layout was removed; the dataset is now a flat class-folder root.
- The retained training code now infers classes dynamically from the dataset root and auto-builds deterministic stratified train/val/test splits.
- A small unresolved residue was discarded from the final dataset rather than forced into the wrong class.

## Local Filename-Based Remap That Replaced Old `other`

Hard constraint:

- The repo did not preserve clean fine-grained labels for the old `other` bucket.
- The original backup only preserved coarse source labels:
  - `organic`
  - `paper`
  - `plastic`
  - `other`
- Therefore the only recoverable fine-grained split for old `other` came from original filenames plus a manual visual review pass.

Major filename families recovered from the old `other` bucket:

- `clothes`
- `glass`
- `shoes`
- `battery`
- `metal`
- `R_*.jpg` can-series imagery, later folded into `metal`
- `image*.jpeg` and `images*.jpg` electronics heaps, later folded into `ewaste`
- `text*.jpg` and `text*.png` textile/clothing imagery, later folded into `clothes`
- cable, TV, speaker, computer-scene filenames, later folded into `ewaste`
- `trash` / `garbage` / `litter` scene filenames, folded into `trash` when no stronger class matched

Effective remap assumptions used locally:

- `cardboard`, `carton`, `corrugated carton`, `pizza box`, `meal carton`, `drink carton`, `paper cup`, `paper bag`, `magazine paper`, `normal paper`, `wrapping paper`, `tissues`, `paper straw` -> `paper`
- `biological`, `biodegradable`, `food organics`, `food waste`, `bio`, `compost`, `vegetation` -> `organic`
- `trash`, `miscellaneous trash`, `general trash`, `non-recyclable`, `miscellaneous`, `unlabeled litter` -> `trash`
- `electronics`, `electronic waste`, `computer`, `monitor`, `laptop`, `phone`, `motherboard`, `circuit`, `hard drive`, `speaker`, `TV`, `charging cable` -> `ewaste`
- `aluminum`, `aluminium`, `drink can`, `food can`, `metal lid`, `metal bottle cap`, `scrap metal`, `pop tab`, can-series product photos -> `metal`
- `green glass`, `brown glass`, `white glass`, `glass bottle`, `glass jar`, `broken glass` -> `glass`
- textile/clothing scene photos -> `clothes`
- shoes remain their own class

Precedence used when multiple filename rules matched:

1. `plastic`
2. `metal`
3. `glass`
4. `battery`
5. `ewaste`
6. `shoes`
7. `clothes`
8. `trash`

## Official Library Checks

I checked the mainstream built-in dataset catalogs because these are the first place a reusable class-aligned dataset would normally surface.

- TorchVision built-in datasets:
  - no native waste / trash dataset found
  - source: https://docs.pytorch.org/vision/stable/datasets.html
- TensorFlow Datasets built-in and community catalogs:
  - no native waste / trash dataset found
  - sources:
    - https://www.tensorflow.org/datasets/catalog
    - https://www.tensorflow.org/datasets/community_catalog/overview

## Remap Assumptions For External Datasets

When evaluating external datasets against this repo, these remaps are considered acceptable:

- `cardboard`, `carton`, `paper cup`, `tetra pak`, `toilet tube` -> `paper`
- `trash`, `miscellaneous trash`, `general trash`, `other`, `non-recyclable`, `miscellaneous` -> `trash`
- `food organics`, `food waste`, `bio`, `biological`, `biodegradable`, `compost`, `vegetation` -> `organic`
- `electronics`, `e-waste`, `cable`, `monitor`, `TV`, `phone`, `computer parts` -> `ewaste`
- `aluminium cans`, `aluminum`, `cans`, `foil`, `scrap metal`, `metal lids`, `metal caps` -> `metal`
- `textile trash`, `clothing`, `garments`, `fabric`, `textile` -> `clothes`
- color-specific glass labels -> `glass`

## Primary Public / Academic / GitHub Datasets

These are the strongest non-Kaggle primary sources or well-known public sources I found that have material overlap with the current project.

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
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
  - close: `food organics -> organic`, `miscellaneous trash -> trash`, `textile trash -> clothes`, `cardboard -> paper`

### TrashNet

- Sources:
  - https://github.com/garythung/trashnet
  - Kaggle mirror: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Classes:
  - `glass`
  - `paper`
  - `cardboard`
  - `plastic`
  - `metal`
  - `trash`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
  - close: `cardboard -> paper`, `trash -> trash`

### TACO

- Sources:
  - official repo: https://github.com/pedropro/TACO
  - format notes: https://deepwiki.com/pedropro/TACO/2.1-dataset-format
- Official detailed classes:
  - `Aluminium foil`
  - `Battery`
  - `Aluminium blister pack`
  - `Carded blister pack`
  - `Other plastic bottle`
  - `Clear plastic bottle`
  - `Glass bottle`
  - `Plastic bottle cap`
  - `Metal bottle cap`
  - `Broken glass`
  - `Food Can`
  - `Aerosol`
  - `Drink can`
  - `Toilet tube`
  - `Other carton`
  - `Egg carton`
  - `Drink carton`
  - `Corrugated carton`
  - `Meal carton`
  - `Pizza box`
  - `Paper cup`
  - `Disposable plastic cup`
  - `Foam cup`
  - `Glass cup`
  - `Other plastic cup`
  - `Food waste`
  - `Glass jar`
  - `Plastic lid`
  - `Metal lid`
  - `Other plastic`
  - `Magazine paper`
  - `Tissues`
  - `Wrapping paper`
  - `Normal paper`
  - `Paper bag`
  - `Plastified paper bag`
  - `Plastic film`
  - `Six pack rings`
  - `Garbage bag`
  - `Other plastic wrapper`
  - `Single-use carrier bag`
  - `Polypropylene bag`
  - `Crisp packet`
  - `Spread tub`
  - `Tupperware`
  - `Disposable food container`
  - `Foam food container`
  - `Other plastic container`
  - `Plastic glooves`
  - `Plastic utensils`
  - `Pop tab`
  - `Rope & strings`
  - `Scrap metal`
  - `Shoe`
  - `Squeezable tube`
  - `Plastic straw`
  - `Paper straw`
  - `Styrofoam piece`
  - `Unlabeled litter`
  - `Cigarette`
- Best overlap:
  - direct: `battery`, `glass`, `metal`, `paper`, `plastic`, `shoes`
  - close: `food waste -> organic`, `unlabeled litter -> trash`

### TACO 7-Class / Pomerania Remap

- Source: https://arxiv.org/abs/2105.06808
- Classes:
  - `bio`
  - `glass`
  - `metal and plastic`
  - `non-recyclable`
  - `other`
  - `paper`
  - `unknown`
- Best overlap:
  - direct: `glass`, `paper`
  - close: `bio -> organic`, `non-recyclable -> trash`, `metal and plastic` partially useful

### CompostNet

- Source: https://github.com/sarahmfrost/compostnet
- Classes described in the repo:
  - `glass`
  - `paper`
  - `cardboard`
  - `plastic`
  - `metal`
  - `trash`
  - `compost`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
  - close: `cardboard -> paper`, `compost -> organic`, `trash -> trash`

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
- Best overlap:
  - direct: `ewaste`, `glass`, `metal`, `paper`, `plastic`
  - close: `cardboard -> paper`

### ZeroWaste

- Source: https://github.com/Trash-AI/ZeroWaste
- Classes:
  - `soft plastic`
  - `rigid plastic`
  - `cardboard`
  - `metal`
- Best overlap:
  - direct: `metal`
  - close: `soft plastic` and `rigid plastic -> plastic`, `cardboard -> paper`

### TrashCan 1.0

- Source: https://conservancy.umn.edu/handle/11299/214865
- Top-level classes:
  - `bio`
  - `trash`
  - `unknown`
- Reported trash subitems include:
  - `paper`
  - `can`
  - `bottle`
  - `bag`
  - `cup`
  - `container`
  - `box`
  - `wire`
  - `bucket`
- Best overlap:
  - direct or close: `bio -> organic`, `paper`, `can -> metal`, `wire -> ewaste/metal`, `trash -> trash`

### BeachLitter v2022

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
- Best overlap:
  - close: `can -> metal`, `carton -> paper`, several `plastic* -> plastic`

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
- Best overlap:
  - direct: `ewaste`, `glass`, `metal`, `plastic`
  - close: `can -> metal`, `cable -> ewaste/metal`

### Trash Detection Dataset

- Source: https://data.mendeley.com/datasets/z732f9pwxt/1
- Classes:
  - `biodegradable`
  - `cardboard`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `all`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
  - close: `biodegradable -> organic`, `cardboard -> paper`

### DWSD

- Source: https://data.mendeley.com/datasets/gr99ny6b8p
- Classes explicitly surfaced in the dataset/paper context:
  - `plastic bottles`
  - `metal bottles`
  - `broken glass`
  - `aluminium foil`
  - `food cartons`
  - `paper`
  - `cups`
  - `tableware`
- Best overlap:
  - direct: `glass`, `metal`, `paper`
  - close: `food cartons -> paper`, foil and bottles -> `metal`

### MMCDWaste

- Source: https://zenodo.org/records/17874659
- Classes:
  - `brick`
  - `cardboard`
  - `ceramic`
  - `concrete`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`
  - `wood`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`

### Custom Bangladeshi E-Waste Dataset

- Source: https://data.mendeley.com/datasets/83wp6scrjt/1
- Classes:
  - `battery`
  - `keyboard`
  - `microwave`
  - `mobile`
  - `mouse`
  - `pcb`
  - `player`
  - `printer`
  - `television`
  - `washing machine`
- Best overlap:
  - direct: `battery`
  - close: all other classes -> `ewaste`

### HGI-30: 30 Classes of Household Garbage Images

- Source: https://arxiv.org/abs/2202.11878
- Scope:
  - 30-class household garbage benchmark
  - built with varied backgrounds, angles, lighting, and shapes
- Best overlap:
  - useful as a broader fine-grained pretraining source for household-waste visual diversity
- Note:
  - the search result exposed the benchmark and its 30-class framing, but not the full class list in the snippet

### GlobalWasteData

- Source: https://arxiv.org/abs/2602.07463
- Scope:
  - integrated large-scale benchmark that combines multiple waste datasets
- Best overlap:
  - useful more as a unification/meta-benchmark source than as a single clean original dataset
- Note:
  - class harmonization details matter here because the dataset is explicitly integrated from heterogeneous sources

### Indoor Waste Image Dataset

- Source: https://www.kaggle.com/datasets/katyanad/indoor-waste-image-dataset
- Classes:
  - `battery`
  - `biological`
  - `brown-glass`
  - `cardboard`
  - `clothes`
  - `green-glass`
  - `metal`
  - `paper`
  - `plastic`
  - `shoes`
  - `trash`
  - `white-glass`
- Best overlap:
  - direct: `battery`, `clothes`, `metal`, `paper`, `plastic`, `shoes`, `trash`
  - close: `biological -> organic`, glass color variants -> `glass`, `cardboard -> paper`

### BDWaste

- Source: https://www.kaggle.com/datasets/harshwardhanbhaskar/bdwaste
- Classes:
  - `cardboard`
  - `glass`
  - `metal`
  - `organic`
  - `paper`
  - `plastic`
- Best overlap:
  - direct: `glass`, `metal`, `organic`, `paper`, `plastic`
  - close: `cardboard -> paper`

### Litter Assessment Dataset

- Source: https://www.kaggle.com/datasets/kneroma/litter-assessment-dataset
- Classes:
  - `glass`
  - `litter`
  - `metal`
  - `organic`
  - `paper`
  - `plastic`
- Best overlap:
  - direct: `glass`, `metal`, `organic`, `paper`, `plastic`
  - close: `litter -> trash`

### Litter Assessment Dataset (Mendeley materials version)

- Source: https://data.mendeley.com/datasets/9dbcd876wj/1
- Material groups:
  - `bulky items`
  - `electronics`
  - `glass`
  - `hazardous`
  - `medical`
  - `metal`
  - `mixed materials`
  - `organics`
  - `paper`
  - `plastic`
  - `textiles`
  - `tobacco`
  - `vehicle`
  - `other items`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
  - close: `organics -> organic`, `electronics -> ewaste`, `textiles -> clothes`, `other items -> trash`
- Note:
  - broader materials-style variant than the simpler six-class Kaggle version

### HOWA

- Source: https://www.iuii.ua.es/datasets/howa/index.html
- Classes:
  - `plastic`
  - `carton`
  - `glass`
  - `metal`
- Counts surfaced on the official page:
  - `plastic`: `1776`
  - `carton`: `1626`
  - `glass`: `1769`
  - `metal`: `1772`
- Best overlap:
  - direct: `glass`, `metal`, `plastic`
  - close: `carton -> paper`
- Note:
  - outdoor household-waste scenes, which makes it more realistic than clean-background object shots

### OpenRecycle

- Source: https://github.com/openrecycle/dataset
- Classes:
  - `cardboard`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `other`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
  - close: `cardboard -> paper`, `other -> trash`

### WaDaBa

- Source: https://wadaba.pcz.pl/
- Scope:
  - plastic-waste image database
- Exposed label structure in public summaries:
  - `PET`
  - `HDPE`
  - `PP`
  - `PS`
  - `Other`
- Best overlap:
  - direct: `plastic`
- Note:
  - narrow but useful if plastic subclass diversity becomes important

### MULTI-TRASH

- Source: https://journals.sagepub.com/doi/full/10.1089/ees.2023.0138
- Classes:
  - `paper`
  - `plastic`
  - `glass`
  - `metal`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
- Note:
  - built to be more realistic and multi-object than older tabletop waste sets

### OrgalidWaste

- Source: https://www.researchgate.net/publication/357260850_Classification_of_Organic_and_Solid_Waste_Using_Deep_Convolutional_Neural_Networks
- Classes:
  - `organic`
  - `glass`
  - `metal`
  - `plastic`
- Size surfaced in the paper summary:
  - around `5600` images
- Best overlap:
  - direct: `glass`, `metal`, `organic`, `plastic`

### Jenks Recycle Dataset

- Source: https://github.com/jenkspt/recycle
- Classes:
  - `glass`
  - `metal`
  - `plastic`
- Best overlap:
  - direct: `glass`, `metal`, `plastic`

### GreenSorter Dataset

- Source: https://github.com/1nfinityLoop/GreenSorter
- Classes surfaced in the public description:
  - `cardboard`
  - `metal`
  - `rigid plastic`
  - `soft plastic`
- Best overlap:
  - direct: `metal`
  - close: `cardboard -> paper`, both plastic variants -> `plastic`

### Huawei-40

- Source: https://www.mdpi.com/2071-1050/14/5/3099
- Class structure:
  - `40` classes total
  - `8` food-waste classes
  - `23` recyclable classes
  - `6` other-garbage classes
  - `3` hazardous-garbage classes
- Best overlap:
  - broad partial overlap across `organic`, `paper`, `plastic`, `metal`, `glass`, `battery`, and `trash`

### Baidu-RC

- Source: https://www.mdpi.com/2071-1050/14/5/3099
- Class structure:
  - `21` recyclable-garbage classes
  - `16,847` images
- Best overlap:
  - strong recyclable-side overlap for `paper`, `plastic`, `metal`, `glass`

### Baidu Garbage Classification Dataset

- Source:
  - summarized in review snippet: https://pmc.ncbi.nlm.nih.gov/articles/PMC12115937/
- Publicly surfaced structure:
  - `17,690` images
  - `158` detailed sub-categories
  - multilingual labeling noted in the review summary
- Best overlap:
  - broad fine-grained recyclable / household-waste supervision
- Note:
  - this appears to be a distinct Baidu garbage-taxonomy family beyond the coarser `Baidu-RC` / `Baidu-214` references

### Baidu-214

- Source: https://www.mdpi.com/2071-1050/14/5/3099
- Class structure:
  - `214` classes
  - `58,063` images
  - `106` recyclable classes
  - `53` food-waste classes
  - `36` other-garbage classes
  - `19` hazardous-garbage classes
- Best overlap:
  - very broad fine-grained supervision for `organic`, `paper`, `plastic`, `metal`, `glass`, `battery`, and `trash`
- Note:
  - this is one of the largest publicly described fine-grained garbage datasets surfaced in the search

### BR-124

- Source: https://www.mdpi.com/2071-1050/14/5/3099
- Class structure:
  - merged recyclable dataset built from `Huawei-40`, `Baidu-214`, and `Baidu-RC`
  - `55,513` images in `124` classes
- Best overlap:
  - broad recyclable fine-grained supervision source for `paper`, `plastic`, `metal`, `glass`

### waste_pictures

- Sources:
  - https://www.kaggle.com/datasets/wangziang/waste-pictures
  - secondary citation snippet: https://www.e3s-conferences.org/articles/e3sconf/pdf/2024/85/e3sconf_rieem2024_04008.pdf
- Scope:
  - around `17,872` to `24,000` images depending on the cited version
  - `34` classes of waste
- Publicly surfaced class examples:
  - `bandaid`
  - `battery`
  - `bowlsanddishes`
  - plus many more fine-grained waste classes
- Best overlap:
  - direct: `battery`
  - likely close overlap with `glass`, `metal`, `paper`, `plastic`, `trash` through its broader 34-class structure
- Note:
  - this appears repeatedly in later waste-classification papers as a broader fine-grained source

### Waste-mart Dataset

- Source:
  - class table exposed in: https://journal.gnest.org/sites/default/files/Submissions/gnest_04501/gnest_04501_draft.pdf
- Classes and counts:
  - `Glass`: `501`
  - `Cardboard`: `403`
  - `Metal`: `410`
  - `Plastic`: `482`
  - `Paper`: `592`
  - `Trash`: `137`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`
- Note:
  - small, but distinct and explicitly sourced from local waste marts rather than web scraping

### Garbage Collection Dataset

- Sources:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC11889143/
  - corroborating survey summary: https://www.journalofbabylon.com/index.php/JUBPAS/article/download/6161/4561/10759
- Classes and counts:
  - `Battery`: `945`
  - `Biological`: `985`
  - `Brown Glass`: `607`
  - `Cardboards`: `891`
  - `Clothes`: `5325`
  - `Green Glass`: `629`
  - `Metal`: `769`
  - `Paper`: `1050`
  - `Plastic`: `865`
  - `Shoes`: `1977`
  - `Trash`: `697`
  - `White Glass`: `775`
- Total images:
  - `15,150`
- Best overlap:
  - direct: `battery`, `clothes`, `metal`, `paper`, `plastic`, `shoes`, `trash`
  - close: `Biological -> organic`, glass color variants -> `glass`, `Cardboards -> paper`

### GIGO Dataset

- Sources:
  - review summary: https://pmc.ncbi.nlm.nih.gov/articles/PMC12115937/
  - thesis snippet: https://norma.ncirl.ie/7361/1/rashisunildabhane.pdf
- Publicly surfaced size:
  - `25,000` images in the review summary
  - `27,603` images in the thesis snippet
- Scope:
  - common household and commercial waste
  - binary `organic` vs `recyclable` setup in the surfaced thesis summary
- Best overlap:
  - direct: `organic`
  - weak partial: `recyclable` mixes `paper`, `plastic`, `glass`, and `metal`
- Note:
  - the public snippets disagree on the exact image count, but they consistently describe it as one of the larger binary waste datasets

### Custom Bangladeshi E-Waste Image Dataset for Object Detection and Recognition

- Source: https://data.mendeley.com/datasets/77383kmdnw
- Classes:
  - `Battery Waste`
  - `Glass Waste`
  - `Keyboard`
  - `Light Bulb`
  - `Medical Waste`
  - `Metal Waste`
  - `Mobile`
  - `Mouse`
  - `Organic Waste`
  - `Paper Waste`
  - `PCB`
  - `Plastic Waste`
- Size:
  - `2,157` images
- Best overlap:
  - direct: `battery`, `glass`, `metal`, `organic`, `paper`, `plastic`
  - close: `keyboard`, `mobile`, `mouse`, `PCB` -> `ewaste`
- Note:
  - this is distinct from the earlier Bangladeshi e-waste dataset entry and has broader cross-material coverage

### Indoor Waste Image Dataset (Mendeley public-space version)

- Source: https://data.mendeley.com/datasets/dy7smxr93r
- Publicly surfaced size:
  - `4,000` color images
- Classes:
  - `paper`
  - `plastic`
  - `cap`
  - `shell`
- Best overlap:
  - direct: `paper`, `plastic`
  - close: `cap` may overlap `metal` or `plastic` depending on item type, `shell` likely maps to `trash`
- Note:
  - collected for indoor public-place waste detection rather than household tabletop classification

### NWNU-TRASH

- Sources:
  - https://www.sciencedirect.com/science/article/abs/pii/S0956053X21004815
  - corroborating review snippets: https://www.scribd.com/document/894844151/A-Systematic-Review-of-AI-Based-Techniques-for-Aut
- Scope:
  - waste-image dataset created to address the limitations of earlier small, clean-background waste sets
- Publicly surfaced size:
  - approximately `18,911` images in one review snippet
- Best overlap:
  - broad household-waste classification benchmark overlap
- Note:
  - public snippets in this pass did not expose the full class list cleanly, so I am not inventing it here

### classify-waste

- Sources:
  - https://mostwiedzy.pl/pl/publication/download/1/deep-learning-based-waste-detection-in-natural-and-urban-environments_64668.pdf
  - project page: https://agnieszka.ai/project/detect-waste-dataset/
- Classes and publicly surfaced image counts:
  - `background`: `1,017`
  - `bio`: `161`
  - `glass`: `2,304`
  - `metals and plastic`: `9,661`
  - `paper`: `1,605`
  - `non-recyclable`: `2,802`
  - `other`: `520`
  - `unknown`: `3,258`
- Total images:
  - `21,328` by the class-count table above
- Best overlap:
  - direct: `glass`, `paper`
  - close: `bio -> organic`, `metals and plastic` partially useful for `metal` and `plastic`, `non-recyclable -> trash`
- Note:
  - built by cropping and normalizing images from several public waste-detection datasets plus additional search images

### detect-waste

- Source: https://agnieszka.ai/project/detect-waste-dataset/
- Scope:
  - merged litter-detection benchmark built from several public detection datasets
- Publicly surfaced composition:
  - `Extended TACO`
  - `Wade-AI`
  - `UAVVaste`
  - `TrashCan`
  - `Trash-ICRA19`
  - `Drinking-Waste`
  - `MJU-Waste`
- Publicly surfaced scale in a thesis summary:
  - over `28,000` images and around `40,000` objects
- Best overlap:
  - strong scene-level waste and litter detection benchmark
- Note:
  - this is more useful for detection / scene generalization than for plain folder classification

### Wade-AI

- Sources:
  - https://agnieszka.ai/project/detect-waste-dataset/
  - https://github.com/letsdoitworld/wade-ai
- Publicly surfaced structure:
  - `1` class
  - about `1,400` images
- Domain:
  - waste in the wild from Google Street View
- Best overlap:
  - useful as an outdoor `trash/litter` detection auxiliary source

### UAVVaste

- Sources:
  - https://paperswithcode.com/dataset/uavvaste
  - https://www.kaggle.com/datasets/federicominutoli/uavvaste
- Publicly surfaced structure:
  - `772` images
  - `3,716` annotations
  - single waste-detection class
- Domain:
  - aerial / drone waste detection
- Best overlap:
  - useful for scene-level litter / trash detection

### Trash-ICRA19

- Sources:
  - https://agnieszka.ai/project/detect-waste-dataset/
  - review snippet in https://github.com/AgaMiko/waste-datasets-review
- Publicly surfaced structure:
  - `7` classes
  - `5,700` images
- Domain:
  - underwater selected trash categories
- Best overlap:
  - auxiliary litter / debris detection source

### OpenLitterMap

- Sources:
  - https://opengeospatialdata.springeropen.com/articles/10.1186/s40965-018-0050-y
  - dataset review summary: https://github.com/AgaMiko/waste-datasets-review
- Publicly surfaced structure:
  - over `100,000` images in total
  - `11` main categories
  - `187` subcategories
- Publicly surfaced examples:
  - `Dogshit`
  - `Unidentified Plastic`
  - `Large/Random Dump`
  - `Metal Object`
  - `Plastic Bags`
  - `Books`
  - `Magazines`
  - `Paper`
  - `Washing-up Bottle`
- Best overlap:
  - very broad litter / trash / plastic / paper / metal open-data source
- Note:
  - one of the largest publicly described crowdsourced litter datasets

### Refined Household Garbage Dataset

- Source:
  - https://arxiv.org/abs/2510.21833
- Publicly surfaced note:
  - `43` corrected mislabels relative to the earlier household-garbage benchmark version used in that study
- Best overlap:
  - same practical overlap as the underlying household-garbage family
- Note:
  - not a distinct raw dataset lineage, but a useful corrected benchmark variant worth being aware of

### VN-trash

- Sources:
  - https://www.mdpi.com/2412-3811/7/4/47
  - https://doaj.org/article/8ae02704e25d4c66967c4ae42ead8fcf
- Classes:
  - `organic`
  - `inorganic`
  - `medical`
- Publicly surfaced size:
  - `5,904` images in the commonly cited original version
  - later review snippets mention a larger `13,000+` image variant/family
- Best overlap:
  - direct: `organic`
  - weak partial: `inorganic` mixes `glass`, `metal`, `paper`, `plastic`, and `trash`
- Note:
  - this is one of the standard Southeast Asian trash benchmarks and remains useful despite its coarse taxonomy

### TriCascade WasteImage

- Sources:
  - main paper snippet: https://e-space.mmu.ac.uk/638258/1/1-s2.0-S0950705125000760-main.pdf
  - additional citation snippet: https://www.scribd.com/document/948781789/1-s2-0-S0950705125000760-main
- Structure:
  - `35,264` images
  - three-stage hierarchical taxonomy
  - stage 1: `2` classes
  - stage 2: `9` classes
  - stage 3: `36` classes
- Publicly surfaced stage-2 classes:
  - `green waste`
  - `recyclable waste`
  - `glass`
  - `metal`
  - `polymer`
  - `leather and fabric`
  - `medical waste`
  - `e-waste`
  - `hazardous waste`
- Publicly surfaced fine-grained examples from the 36-class stage:
  - `Foods`
  - `Animal Dead Body`
  - `Cardboard`
  - `Newspaper`
  - `Paper Cups`
  - `Paper`
  - `Brown Glass`
  - `Bowls and Dishes (Porcelain)`
  - `Green Glass`
  - `White Glass`
  - `Beverage Cans`
  - `Construction Scrap`
  - `Metal Containers`
  - `Plastic Bag`
  - `Plastic Bottle`
  - `Plastic Container`
  - `Plastic Cup`
  - `Tetra Pak`
  - `Cloths (Synthetic Fabric)`
  - `Shoes`
  - `Gloves`
  - `Masks`
  - `Bandage`
- Best overlap:
  - direct: `ewaste`, `glass`, `metal`
  - close: `green waste -> organic`, `Cardboard/Newspaper/Paper Cups/Paper/Tetra Pak -> paper`, `polymer` and plastic subclasses -> plastic, `Cloths` -> clothes, `Shoes` -> shoes, `hazardous waste` partly overlaps `battery` and `trash`
- Note:
  - one of the strongest many-class hierarchical waste benchmarks surfaced so far

### Recycleye WasteNet

- Source: https://recycleye.com/wastenet/
- Scope:
  - proprietary industrial waste dataset
  - publicly claimed to exceed `3 million` training images
  - later public marketing material mentions `1 billion+` analyzed images across production usage
- Best overlap:
  - broad industrial waste-stream coverage
- Note:
  - not a public download dataset, so it is not directly usable like the open sources above
  - still worth tracking because it is one of the largest waste-image resources publicly described

### WaRP / WaRP-C / WaRP-D

- Source:
  - class examples surfaced in the paper snippet: https://grafft.github.io/assets/pdf/eaai2024.pdf
- Publicly surfaced class examples:
  - `bottle`
  - `cans`
  - `cardboard`
  - `canister`
  - `detergent`
- Best overlap:
  - close: `cans -> metal`, `cardboard -> paper`, bottles/canisters/detergent containers -> likely `plastic` or `glass` depending on subtype
- Note:
  - useful as a waste-recycling-plant dataset family rather than a household tabletop set

### Construction and Demolition Waste Object Detection Dataset (CODD)

- Source: https://data.mendeley.com/datasets/wds85kt64j/3
- Publicly surfaced size:
  - `3,129` images
  - `16,545` annotated samples
- Classes:
  - `bricks`
  - `concrete`
  - `tiles`
  - `wood`
  - `pipes`
  - `plastics`
  - `general waste`
  - `foaming insulation`
  - `stones`
  - `plaster boards`
- Best overlap:
  - direct: `plastic`
  - close: `general waste -> trash`
- Note:
  - construction-demolition domain, but still a distinct real-world waste detection benchmark

### MJU-Waste

- Source:
  - summarized in: https://mostwiedzy.pl/pl/publication/download/1/deep-learning-based-waste-detection-in-natural-and-urban-environments_64668.pdf
- Reported variants:
  - one-class `rubbish` version for litter detection
  - `16` classes for material version
  - `22` classes for instance version
- Best overlap:
  - broad litter / material recognition overlap
- Note:
  - more useful for detection/segmentation and material discrimination than clean household classification

### Recycling Waste and Litter

- Source: https://www.kaggle.com/datasets/humansintheloop/recycling-dataset
- Structure:
  - bounding boxes with main class `litter`
  - three mandatory attributes: `material`, `object`, and `brand`
- Best overlap:
  - indirectly useful for `plastic`, `metal`, `paper`, `glass`, and `trash` via its attribute system
- Note:
  - not a plain multiclass folder dataset, but a useful litter detection / material-tagging resource

### EcoDetect-YOLO surveillance waste dataset

- Source: https://www.mdpi.com/1424-8220/24/14/4666
- Scope:
  - surveillance camera domestic-waste exposure detection dataset
- Publicly surfaced size:
  - `3,995` images
- Publicly surfaced targets:
  - `9` detection target types
  - search snippet explicitly mentions `paper trash` and `plastic trash`
- Best overlap:
  - useful for real-world `trash` / public-environment scene understanding
- Note:
  - detection-oriented and scene-heavy rather than clean single-object classification

### BePLi Dataset v2

- Source: https://www.sciencedirect.com/science/article/pii/S235234092500592X
- Scope:
  - beach plastic litter instance-segmentation dataset
- Publicly surfaced size:
  - `3,722` original images
- Classes:
  - `pet_bottle`
  - `other_bottle`
  - `plastic_bag`
  - `box_shaped_case`
  - `other_container`
  - `rope`
  - `other_string`
  - `fishing_net`
  - `buoy`
  - `other_fishing_gear`
  - `styrene_foam`
  - `others`
  - `fragment`
- Best overlap:
  - strong plastic-litter auxiliary source
- Note:
  - plastic-only, but unusually fine-grained and real-world

### CleanSea

- Source: https://www.dlsi.ua.es/~jgallego/datasets/cleansea/
- Scope:
  - underwater debris detection and recognition benchmark
- Publicly surfaced labels include:
  - `Plastic waste`
  - `Metal waste`
  - additional underwater debris object categories
- Best overlap:
  - direct or close: `metal`, `plastic`
- Note:
  - underwater domain, so not a direct household classifier source, but still a distinct debris dataset

### StreetView-Waste

- Sources:
  - https://streetview-waste.di.ubi.pt/
  - https://arxiv.org/abs/2511.16440
- Scope:
  - multi-task urban waste-management dataset
  - litter segmentation plus waste-container detection/tracking
- Publicly surfaced structure:
  - large-scale benchmark for `7` classes of waste containers
  - overflowing litter segmentation task
- Best overlap:
  - useful for real-world urban `trash` / litter scene understanding
- Note:
  - not a direct household-item classifier dataset, but highly relevant for scene-level waste monitoring

### AquaTrash

- Source: https://www.kaggle.com/datasets/harshpanwar/aquatrash
- Classes:
  - `glass`
  - `paper`
  - `metal`
  - `plastic`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`

## Kaggle Datasets With Strong Practical Overlap

These are Kaggle-hosted datasets or dataset families with substantial overlap with the current taxonomy. Some are original uploads, some are cleaned derivatives, and some are mirrors of public datasets.

### Real-World Waste Classification Dataset (5 Class)

- Source: https://www.kaggle.com/datasets/tuhanasinan/real-world-waste-classification-dataset-5-class
- Classes:
  - `metal`
  - `plastic`
  - `glass`
  - `paper`
  - `organic`
- Best overlap:
  - direct: `glass`, `metal`, `organic`, `paper`, `plastic`

### Garbage Dataset / Garbage Classification V2

- Sources:
  - https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
  - class list corroborated in the related paper: https://tecnoscientifica.com/journal/idwm/article/download/408/208/2771
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
- Best overlap:
  - direct: `battery`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`
  - close: `biological -> organic`, `cardboard -> paper`

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
- Best overlap:
  - direct: `glass`, `metal`, `organic`, `paper`, `plastic`
  - close: `electronics -> ewaste`, `miscellaneous -> trash`

### Multi Class Garbage Classification Dataset

- Source: https://www.kaggle.com/datasets/vishallazrus/multi-class-garbage-classification-dataset
- Classes:
  - `cardboard`
  - `compost`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `compost -> organic`, `cardboard -> paper`

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
- Best overlap:
  - direct: `battery`, `clothes`, `metal`, `paper`, `plastic`, `shoes`
  - close: `biological waste -> organic`, glass color variants -> `glass`, `general trash -> trash`, `cardboard -> paper`

### Garbage Classification (12 classes) ENHANCED

- Source: https://www.kaggle.com/datasets/huberthamelin/garbage-classification-labels-corrections
- Classes:
  - `paper`
  - `cardboard`
  - `biological`
  - `metal`
  - `plastic`
  - `green-glass`
  - `brown-glass`
  - `white-glass`
  - `clothes`
  - `shoes`
  - `batteries`
  - `trash`
- Notes:
  - corrected / enhanced derivative of the 12-class garbage-classification family
  - useful when preferring the corrected label set over the raw upstream variant
- Best overlap:
  - direct: `battery`, `clothes`, `metal`, `paper`, `plastic`, `shoes`, `trash`
  - close: `biological -> organic`, glass color variants -> `glass`, `cardboard -> paper`

### Recyclable And Household Waste Classification

- Source: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification
- Publicly surfaced size and structure:
  - `15,000` images
  - `30` distinct categories
  - `500` images per category
- Indexed categories explicitly surfaced:
  - `Organic Waste`
  - `Textile Waste`
- Notes:
  - later papers also reference a related household-waste subset with `18` categories and `7,361` images for recyclable / hazardous / kitchen / other waste tasks
- Best overlap:
  - direct: `organic`
  - close: `Textile Waste -> clothes`

### Custom Waste Classification Dataset

- Source: https://www.kaggle.com/datasets/wasifmahmood01/custom-waste-classification-dataset
- Classes explicitly surfaced in the dataset page/snippet:
  - `battery waste`
  - `glass waste`
  - `metal waste`
  - `organic waste`
  - `paper waste`
  - `plastic waste`
  - `e-waste`
- The page also states `9 classes of waste`, but only these seven were surfaced in the indexed snippet.
- Best overlap:
  - direct: `battery`, `ewaste`, `glass`, `metal`, `organic`, `paper`, `plastic`

### Waste Segregation Image Dataset

- Source: https://www.kaggle.com/datasets/aashidutt3/waste-segregation-image-dataset
- Indexed classes surfaced in search:
  - `paper`
  - `leaf`
  - `food`
  - `wood`
  - `waste`
  - `plastic bags`
  - `plastic bottles`
  - `metal cans`
- Best overlap:
  - direct: `paper`
  - close: `food` and `leaf -> organic`, `waste -> trash`, `metal cans -> metal`, plastic variants -> `plastic`

### VN Trash Classification

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
- Best overlap:
  - direct: `paper`
  - close: `aluminum cans -> metal`, `carton` and `paper cup -> paper`, `PET bottle` and `clear plastic cup -> plastic`, `other trash -> trash`

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
- Best overlap:
  - direct: `battery`, `clothes`, `glass`, `metal`, `paper`, `plastic`
  - close: `cardboard -> paper`

### Waste Classification Data

- Source: https://www.kaggle.com/datasets/techsash/waste-classification-data
- Classes:
  - `organic`
  - `recyclable`
- Best overlap:
  - direct: `organic`
  - weak partial: `recyclable` mixes `paper`, `plastic`, `glass`, `metal`

### Waste Classification Data v2

- Source: https://www.kaggle.com/datasets/sapal6/waste-classification-data-v2
- Commonly cited class set:
  - `organic`
  - `recyclable`
  - `nonrecyclable`
- Best overlap:
  - direct: `organic`
  - close: `nonrecyclable -> trash`

### Waste Classification Dataset (Mendeley clean-up)

- Source: https://data.mendeley.com/datasets/n3gtgm9jxj/2
- Classes and counts:
  - `organic`: `13,880`
  - `recyclable`: `10,825`
- Total images:
  - `24,705`
- Notes:
  - cleaned and restructured version of the original Kaggle binary waste-classification dataset
- Best overlap:
  - direct: `organic`
  - weak partial: `recyclable` mixes `paper`, `plastic`, `glass`, and `metal`

### Waste Classification Dataset by phenomsg

- Source: https://www.kaggle.com/datasets/phenomsg/waste-classification
- Classes:
  - `hazardous`
  - `non-recyclable`
  - `organic`
  - `recyclable`
- Best overlap:
  - direct: `organic`
  - close: `non-recyclable -> trash`

### Waste Classification (organic and recyclable)

- Source: https://www.kaggle.com/datasets/preetishah/waste-classificationorganic-and-recyclable
- Classes:
  - `organic`
  - `recyclable`
- Best overlap:
  - direct: `organic`

### Image Waste Classification Dataset

- Source: https://www.kaggle.com/datasets/meetpatelp27/image-waste-classification-dataset
- Classes:
  - `organic waste`
  - `recyclable waste`
- Best overlap:
  - direct: `organic`

### CNN: Waste Classification [Image] Dataset

- Source: https://www.kaggle.com/datasets/sujaykapadnis/cnn-waste-classification-image-dataset
- Classes:
  - `organic`
  - `recyclable`
- Best overlap:
  - direct: `organic`

### Waste Classfication Dataset by kaanerkez

- Source: https://www.kaggle.com/datasets/kaanerkez/waste-classfication-dataset/data
- Indexed classes surfaced:
  - `battery`
  - `cardboard`
  - `glass`
  - `keyboard`
  - `metal`
  - `microwave`
  - `mobile`
  - `mouse`
  - `organic`
  - `paper`
  - `pcb`
  - `plastic`
  - `player`
  - `printer`
  - `television`
  - `trash`
  - `washing machine`
- Best overlap:
  - direct: `battery`, `glass`, `metal`, `organic`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`, the device classes -> `ewaste`

### New Trash Classfication Dataset

- Source: https://www.kaggle.com/datasets/glhdamar/new-trash-classfication-dataset
- Classes:
  - `plastic`
  - `paper`
  - `metal`
  - `glass`
  - `organic`
  - `e-waste`
  - `textile`
  - `trash`
- Notes:
  - explicitly documented as a merged, balanced dataset built from multiple public sources
  - the page also exposes its label-mapping rules, which is unusually useful for reuse
- Best overlap:
  - direct: `ewaste`, `glass`, `metal`, `organic`, `paper`, `plastic`, `trash`
  - close: `textile -> clothes`

### Unified Waste Classification Dataset

- Source: https://www.kaggle.com/datasets/siddhantmaji/unified-waste-classification-dataset
- Classes:
  - `plastic`
  - `paper_cardboard`
  - `glass`
  - `metal`
  - `organic_waste`
  - `textiles`
  - `battery`
  - `trash`
- Notes:
  - explicitly documented as a merged, relabeled, and balanced dataset
  - source datasets listed on the page: `Garbage Classification`, `Garbage Classification V2`, `Recyclable and Household Waste Classification`
- Best overlap:
  - direct: `battery`, `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `paper_cardboard -> paper`, `organic_waste -> organic`, `textiles -> clothes`

### garbage-dataset-10-classes by sudipp

- Source: https://www.kaggle.com/datasets/sudipp/garbage-dataset-9-classes/data
- Classes:
  - `battery`
  - `biological`
  - `cardboard`
  - `clothes`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `shoes`
  - `trash`
- Notes:
  - explicitly documented as a merged dataset built from `Garbage Classification`, `Garbage Dataset`, and `Garbage Classification (12 classes)`
- Best overlap:
  - direct: `battery`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`
  - close: `biological -> organic`, `cardboard -> paper`

### Garbage Images Dataset (2000/class)

- Source: https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification
- Classes:
  - `plastic`
  - `metal`
  - `glass`
  - `cardboard`
  - `paper`
  - `trash`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`

### Garbage Image Dataset by farzadnekouei

- Source: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
- Classes:
  - `cardboard`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`

### Labeled Trash Dataset (Pascal VOC, COCO and Yolo)

- Source: https://www.kaggle.com/datasets/pandapowr/labeled-trash-dataset-pascal-voc-coco-and-yolo
- Classes:
  - `Bottle`
  - `Bottle Cap`
  - `Can`
  - `Cardboard`
  - `Cigarette Box`
  - `Cigarette Butt`
  - `Cup`
  - `Plastic`
- Best overlap:
  - direct: `plastic`
  - close: `can -> metal`, `cardboard -> paper`, `bottle` and `bottle cap` may split between `glass`, `plastic`, and `metal` depending on item material

### Fotini10k

- Source: https://arxiv.org/abs/2104.00868
- Classes:
  - `plastic bottles`
  - `aluminum cans`
  - `paper and cardboard`
- Best overlap:
  - close: `plastic bottles -> plastic`, `aluminum cans -> metal`, `paper and cardboard -> paper`
- Note:
  - narrow but clean for recyclable-object pretraining

### ReCoDeWaste

- Source: https://github.com/prasadvineetv/ReCoDeWaste
- Classes surfaced in the repo/page:
  - `cardboard`
  - `hard plastic`
  - `metal`
  - plus additional construction and demolition material classes
- Best overlap:
  - direct: `metal`
  - close: `cardboard -> paper`, `hard plastic -> plastic`
- Note:
  - construction/demolition domain rather than household-waste domain, but still a real material-sorting dataset

### Garbage Classification (12 classes) by mostafaabla

- Source: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- Classes:
  - `paper`
  - `cardboard`
  - `biological`
  - `metal`
  - `plastic`
  - `green-glass`
  - `brown-glass`
  - `white-glass`
  - `clothes`
  - `shoes`
  - `batteries`
  - `trash`
- Best overlap:
  - direct: `battery`, `clothes`, `metal`, `paper`, `plastic`, `shoes`, `trash`
  - close: `biological -> organic`, glass color variants -> `glass`, `cardboard -> paper`

### Household Waste Dataset index on Papers With Code

- Source: https://paperswithcode.com/dataset/household-waste-dataset
- Exposed classes:
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
- Best overlap:
  - direct: `battery`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`
  - close: `biological -> organic`, `cardboard -> paper`
- Note:
  - this is useful as an index/corroboration page; the underlying data lineage still points back to the broader garbage dataset family

### Drinking Waste Classification

- Source: https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification
- Classes:
  - `aluminium cans`
  - `glass bottles`
  - `PET bottles`
  - `HDPE milk bottles`
- Best overlap:
  - direct: `glass`
  - close: `aluminium cans -> metal`, bottle variants -> `plastic`

### EcoDetect Recyclable Waste Detection Dataset

- Source: https://www.kaggle.com/datasets/ahsan71/ecodetect-recyclable-waste-detection-dataset
- Classes:
  - `plastic`
  - `paper`
  - `aluminum`
- Best overlap:
  - direct: `paper`, `plastic`
  - close: `aluminum -> metal`

## Narrower Specialty Datasets

These are narrower, but still useful if a specific class needs extra support.

### Metal Scrap Dataset

- Source: https://huggingface.co/datasets/iDharshan/metal_scrap_dataset
- Scope:
  - synthetic / generated-style metal-scrap imagery
- Best overlap:
  - direct: `metal`
- Caution:
  - lower trust than real-world waste datasets because of likely synthetic bias

### Roboflow E-Waste Dataset

- Sources:
  - https://universe.roboflow.com/electronic-waste-detection
  - https://universe.roboflow.com/jeff-t1fqn/e-waste-dataset-r0ojc-981jy
- Scope:
  - large annotated e-waste detection dataset
  - one surfaced version reports `19613` images and `77` classes
- Exposed classes include:
  - `Battery`
  - `Laptop`
  - `Microwave`
  - `Smartphone`
  - `Printer`
  - `PCB`
  - `Computer-Keyboard`
  - `Computer-Mouse`
  - `Flat-Panel-TV`
  - `CRT-TV`
  - `Washing-Machine`
  - many other device-specific e-waste labels
- Best overlap:
  - direct: `battery`
  - close: almost all remaining classes -> `ewaste`
- Note:
  - this is a strong specialty source for `ewaste`, but it is object-detection-oriented and device-fine-grained

### Roboflow100-VL waste-material subset

- Source: https://media.roboflow.com/rf100vl/rf100vl.pdf
- Exposed waste-material classes in the waste subset:
  - `Aggregate`
  - `Cardboard`
  - `Hard Plastic`
  - `Metal`
  - `Soft Plastic`
  - `Timber`
- Best overlap:
  - direct: `metal`
  - close: `cardboard -> paper`, both plastic variants -> `plastic`
- Note:
  - useful mainly as a material-sorting auxiliary source, not as a full household-waste taxonomy

### Plastic Waste Classification

- Source: https://www.kaggle.com/datasets/yadavmohit04/plastic-waste-classification-dataset
- Scope:
  - plastic-only
- Best overlap:
  - direct: `plastic`

### Images of mixed-type recyclable waste

- Source: https://zenodo.org/records/15575903
- Scope:
  - RGB and HSI images of mixed recyclable materials on a conveyor belt
- Publicly surfaced material examples:
  - `PET`
  - `HDPE`
  - `aluminum`
- Best overlap:
  - direct: `plastic`
  - close: `aluminum -> metal`
- Note:
  - useful as a conveyor / industrial sorting auxiliary dataset

### SWIR hyperspectral plastics detection in the environment

- Source: https://data.mendeley.com/datasets/y8cvcs8tt5
- Scope:
  - SWIR hyperspectral cubes for plastic-litter detection in environmental scenes
- Publicly surfaced size:
  - `7` hyperspectral cubes
- Best overlap:
  - direct: `plastic`
- Note:
  - niche, but distinct and useful if hyperspectral plastic detection ever matters

### E-Waste Classification Dataset

- Source: https://www.kaggle.com/datasets/akshat103/e-waste-classification-dataset
- Scope:
  - e-waste-focused
- Best overlap:
  - direct: `ewaste`

### E-Waste Vision Dataset

- Sources:
  - https://paperswithcode.com/paper/ewastenet-a-two-stream-data-efficient-image
  - https://github.com/NifulIslam/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification
- Scope:
  - `8` e-waste device classes
- Best overlap:
  - direct or close: broad `ewaste` support
- Note:
  - useful as another independent e-waste source beyond the Roboflow-style detection sets

### pLitterStreet

- Source: https://paperswithcode.com/paper/plitterstreet-street-level-plastic-litter
- Scope:
  - street-level plastic litter detection and mapping
  - more than `13,000` annotated images according to the paper summary
- Best overlap:
  - direct: plastic-heavy litter source
  - close: can help with `trash` / roadside clutter scenes even though the focus is plastic litter

### Biomedical Waste Dataset

- Source: https://www.kaggle.com/datasets/mario78/medical-waste-dataset
- Scope:
  - medical-waste-only
- Best overlap:
  - none direct to current final taxonomy, but useful as hard negatives against `trash` or `ewaste` depending on future scope

## Additional Non-Duplicate Sweep: 2026-04-02

These entries were added in a fresh pass after the earlier survey. They are not duplicates of the sources already listed above, though some are derivative or mixed-source datasets and are flagged accordingly.

### Recycled Dataset (Portland State)

- Source: https://web.cecs.pdx.edu/~singh/rcyc-web/dataset.html
- Publicly surfaced size:
  - `11,500` images
  - `5` classes
  - `2,300` images per class
- Classes:
  - `boxes`
  - `glass_bottles`
  - `soda cans`
  - `crushed soda cans`
  - `water bottles`
- Best overlap:
  - direct-ish: `glass`, `metal`, `plastic`
  - close: `boxes -> paper/cardboard`
- Note:
  - lower resolution (`32x32`) and tightly controlled capture setup, so it is more useful as a narrow recyclable-material source than as a realistic household-waste benchmark

### rootstrap-org/waste-classifier

- Source: https://huggingface.co/datasets/rootstrap-org/waste-classifier
- Publicly surfaced size:
  - downloaded size about `1.15 GB`
  - viewer reports `3,261` rows
- Classes:
  - `cardboard`
  - `compost`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`, `compost -> organic`
- Note:
  - explicitly described as `TrashNet + internet images`; still worth listing because it is a materially expanded derivative rather than a pure mirror

### (256x256) Litter Sort Annotated Wastes

- Source: https://huggingface.co/datasets/mnemoraorg/256x256-litter-sort-annotated-wastes
- Publicly surfaced size:
  - `14,268` images
- Classes:
  - `Plastic`
  - `Metal`
  - `Glass`
  - `Cardboard`
  - `Paper`
  - `Trash`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`
- Note:
  - the dataset card claims duplicates were removed and the set was quality-filtered and balanced; that makes it a potentially cleaner six-material source than many older TrashNet-style packs

### WasteBench

- Source: https://huggingface.co/datasets/aliman8/WasteBench-Dataset
- Publicly surfaced size:
  - `952` images
- Categories:
  - `Rigid Plastic`
  - `Soft Plastic`
  - `Cardboard`
  - `Metal`
  - `Glass`
  - `Paper`
  - `Trash`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `trash`
  - close: `rigid plastic + soft plastic -> plastic`, `cardboard -> paper`
- Note:
  - this is an evaluation-oriented VLM benchmark rather than a classic train set, but it is still relevant for hard-scene validation and cluttered waste-image testing

### SpectralWaste

- Sources:
  - Zenodo: https://zenodo.org/records/10880544
  - paper summary: https://arxiv.org/abs/2403.18033
- Publicly surfaced size:
  - `23.1 GB` segmentation archive
- Classes:
  - `film`
  - `basket`
  - `video tape`
  - `filament`
  - `trash bag`
  - `cardboard`
- Best overlap:
  - close: `cardboard -> paper`, `trash bag -> trash`
- Note:
  - this is an industrial multimodal waste-sorting dataset rather than a household taxonomy match, but it is one of the more distinctive recent open datasets and could be useful for cluttered sorting robustness

### dmedhi/garbage-image-classification-detection

- Source: https://huggingface.co/datasets/dmedhi/garbage-image-classification-detection
- Publicly surfaced classes:
  - `Cardboard`
  - `Glass`
  - `Metal`
  - `Paper`
  - `Plastic`
  - `Trash`
- Publicly surfaced annotation type:
  - bounding boxes
  - segmentation masks
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`, `trash`
  - close: `cardboard -> paper`
- Note:
  - this one is useful specifically because it exposes detection and segmentation supervision, not just folder-level classification labels

### Baseline Marine Debris Data (CHNMS)

- Source: https://zenodo.org/records/11268518
- Publicly surfaced structure:
  - `40` marine debris item-type categories
  - `6` material categories
  - `7` source-activity categories
- Material categories:
  - `plastic`
  - `glass`
  - `metal`
  - `cloth`
  - `paper and wood`
  - `mixed`
- Best overlap:
  - direct-ish: `glass`, `metal`, `plastic`
  - close: `cloth -> clothes`, `paper and wood` partially useful for `paper`, `mixed -> trash`
- Note:
  - this is a marine-debris assessment dataset rather than a household garbage benchmark, but it is genuinely distinct and can help with messy real-world litter material distribution

### recycle_net

- Source: https://github.com/SebastianCharmot/recycle_net
- Publicly surfaced size:
  - `1,969` annotated images
- Classes:
  - `glass`
  - `paper`
  - `plastic`
  - `metal`
- Best overlap:
  - direct: `glass`, `metal`, `paper`, `plastic`
- Note:
  - valuable mainly because it is an object-detection-oriented recyclable dataset with manual bounding boxes rather than pure folder classification

### RecycleTree model family data references

- Source: https://huggingface.co/pyesonekyaw/recycletree_paper
- Publicly surfaced paper-model data:
  - `9,646` images across `12` classes
- Exposed paper-model classes:
  - `Beverage Carton`
  - `Cardboard`
  - `Chopsticks`
  - `Disposables`
  - `Paper Bag`
  - `Paper Packaging`
  - `Paper Product`
  - `Paper Receipt`
  - `Paper Roll`
  - `Paper Sheet`
  - `Tissue Box`
  - `Tissue Paper`
- Publicly surfaced broader family:
  - materials classification model
  - plastic classification model
  - metal classification model
  - glass classification model
  - others classification model
- Best overlap:
  - direct-ish: strong `paper` specialization
  - close: `Beverage Carton`, `Cardboard`, `Paper Packaging`, `Tissue*` all map well to `paper`
- Note:
  - this is especially useful if you later decide to strengthen paper/cardboard subtyping or create specialist heads per top-level material

### Underwater Plastic Pollution Detection

- Source: https://www.kaggle.com/datasets/arnavs19/underwater-plastic-pollution-detection
- Publicly surfaced size:
  - train `3,628`
  - valid `1,001`
  - test `501`
  - total `5,130` images
- Classes:
  - `Mask`
  - `can`
  - `cellphone`
  - `electronics`
  - `gbottle`
  - `glove`
  - `metal`
  - `misc`
  - `net`
  - `pbag`
  - `pbottle`
  - `plastic`
  - `rod`
  - `sunglasses`
  - `tire`
- Best overlap:
  - direct-ish: `metal`, `plastic`
  - close: `electronics -> ewaste`, `can -> metal`, `gbottle -> glass`, `misc -> trash`, `glove` can act as clutter/trash hard negative
- Note:
  - domain is underwater debris, so this is not a direct household match, but it is another genuinely distinct many-class litter source

### PlasticInWater

- Source: https://huggingface.co/datasets/OceanCV/PlasticInWater
- Publicly surfaced size:
  - `4,511` images
  - `15` classes
- Classes:
  - `Black Plastic Cap`
  - `Blue Nitrile Glove`
  - `Blue Plastic Cap`
  - `Brown Multilayer Plastic`
  - `Green Plastic Cap`
  - `Orange Plastic Cap`
  - `Plastic Bottle`
  - `Purple Insulation Foam`
  - `Purple Multilayer Plastic Bag`
  - `Red-Orange BOPP Bag`
  - `Red Cap`
  - `Red Netting`
  - `Red Plastic Straw`
  - `Yellow Foam`
  - `Yellow Rope`
- Best overlap:
  - direct: `plastic`
  - close: gloves/rope/netting/foam help as trashy clutter subclasses if you later break `trash` down further
- Note:
  - niche and plastic-only, but distinct and reasonably fine-grained

## Meta-Resources

These are not directly training datasets, but they are useful index pages for finding more waste datasets and understanding how dataset families relate to each other.

- Waste datasets review:
  - https://agamiko.github.io/waste-datasets-review/
- TACO repository:
  - https://github.com/pedropro/TACO

## Practical Non-Duplicate Shortlist For This Repo

If the goal is to add external data without drowning in mirrors and near-duplicates, these are the highest-value unique sources to prioritize first:

1. `RealWaste`
2. `TACO`
3. `TrashNet`
4. `MMCDWaste`
5. `TrashBox`
6. `Indoor Waste Image Dataset`
7. `Garbage Classification V2`
8. `Garbage_dataset_PlusYaml`
9. `Real-World Waste Classification Dataset (5 class)`
10. `BDWaste`
11. `Litter Assessment Dataset`
12. `Custom Bangladeshi E-Waste Dataset`
13. `New Trash Classfication Dataset`
14. `Unified Waste Classification Dataset`
15. `Roboflow E-Waste Dataset`

Why this shortlist:

- it covers nearly every current local class except that `trash` remains the hardest class to source cleanly
- it avoids overcounting obvious TrashNet mirrors
- it includes stronger support for `battery`, `clothes`, `ewaste`, `shoes`, and `trash`, which are harder to source than `paper`, `plastic`, `glass`, and `metal`

## Deduplication Note

Kaggle contains many mirrors, repackagings, and lightly modified reuploads of the same core families, especially:

- `TrashNet`
- `RealWaste`
- `Waste Classification Data`
- general `garbage-classification` six-class packs

This file favors unique datasets or meaningfully expanded derivatives instead of counting every mirror as a separate source.

## Fresh Sweep Additions

On the latest broader web sweep, the most useful non-duplicate additions beyond the earlier list were:

- `New Trash Classfication Dataset`
- `Unified Waste Classification Dataset`
- `garbage-dataset-10-classes`
- `Garbage Classification (12 classes)` by `mostafaabla`
- `Garbage Classification (12 classes) ENHANCED`
- `Roboflow E-Waste Dataset`
- `HGI-30`
- `GlobalWasteData`
- `Custom Waste Classification Dataset`
- `Fotini10k`
- `ReCoDeWaste`
- `Recycled Dataset (Portland State)`
- `rootstrap-org/waste-classifier`
- `(256x256) Litter Sort Annotated Wastes`
- `WasteBench`
- `SpectralWaste`
- `dmedhi/garbage-image-classification-detection`
- `Baseline Marine Debris Data (CHNMS)`
- `recycle_net`
- `RecycleTree model family data references`
- `Underwater Plastic Pollution Detection`
- `PlasticInWater`

These are the ones most worth checking next if the goal is to expand the current project’s class support without drowning in mirrors.
