**Overview**
The repo now has an Ollama-first end-to-end curation path for material-class dataset building.

Default text model:
- `deepseek-r1:8b`

Default vision model:
- `qwen2.5vl:3b`

Default target classes when `--class-spec` is omitted:
- `organic`
- `metal`
- `paper`

These defaults are defined in [scripts/ollama_pipeline_defaults.py](/home/anurag-basistha/Projects/ESD/scripts/ollama_pipeline_defaults.py).

**Flow**
1. User provides target classes plus optional seed objects.
2. DeepSeek expands each class into common real-world object subclasses.
3. Multi-source download runs for each prefixed query:
   - direct Bing scraping
   - Google via `icrawler`
   - Bing via `icrawler`
   - Baidu via `icrawler`
4. Images smaller than `224x224` are deleted immediately.
5. Exact dedupe and perceptual dedupe (`sha256` + `imagehash phash`) run before model judging.
6. A cheap CLIP prefilter rejects images where the target is not dominant or the frame is dominated by humans / clutter / multiple salient objects.
7. The single default vision model runs after prefilter.
8. Each image goes through a two-stage decision:
   - stage 1: target superclass classification
   - stage 2: train-worthiness scoring for pristine, learnable, real-world data
9. The model writes `accepted/`, `rejected/`, and `uncertain/` on the fly.
10. Accepted samples are integrated into `Dataset_Final` by default with metadata sync.
11. A deterministic train / val / test split manifest is written using the exact repo split logic.
12. Training can optionally be triggered by a configured command.

**Artifact Layout**
- `raw/`
- `filtered/`
- `accepted/`
- `rejected/`
- `uncertain/`
- `integrated/`
- `manifests/`
- `logs/`

**State And Resume**
- The pipeline uses a SQLite manifest at:
  - `manifests/pipeline.sqlite`
- Resume happens per stage:
  - class discovery reuses `manifests/class_discovery/discovered_classes.json`
  - per-item download jobs resume from `download_jobs`
  - prefilter resumes from rows without `prefilter_decision`
  - VLM judging resumes from rows without `final_decision`
  - integration resumes from rows with `final_decision='accepted'` and pending integration

**Per-Image Ledger**
Each image row tracks:
- source engine
- original URL
- query
- download timestamp
- sha256
- phash
- resolution
- exact dedupe result
- perceptual dedupe result
- prefilter decision
- class-stage decision
- train-worthiness decision
- final decision
- integration status

**Health Logs**
- domain health:
  - `logs/domain_health.json`
- model health:
  - `logs/model_health.json`
- download jobs:
  - `manifests/download_jobs.json`
- provenance summary:
  - `manifests/provenance_summary.json`

**Threshold Calibration**
- `scripts/calibrate_ollama_thresholds.py`
- Runs the current vision judge against pristine dataset folders such as:
  - `organic`
  - `metal`
  - `paper`
  - `plastic`
- Reports positive-class acceptance and plastic false-accept rates.

Default download ceilings per listed item:
- direct Bing: `1250`
- Google `icrawler`: `1250`
- Bing `icrawler`: `1250`
- Baidu `icrawler`: `1250`

Effective total requested ceiling per item:
- `5000`

**Scripts**
- [scripts/setup_ollama_models.py](/home/anurag-basistha/Projects/ESD/scripts/setup_ollama_models.py)
- [scripts/run_ollama_end_to_end_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/run_ollama_end_to_end_pipeline.py)
- [scripts/calibrate_ollama_thresholds.py](/home/anurag-basistha/Projects/ESD/scripts/calibrate_ollama_thresholds.py)

**Example**
```bash
.venv/bin/python scripts/setup_ollama_models.py

.venv/bin/python scripts/run_ollama_end_to_end_pipeline.py \
  --class-spec '{"paper":{"seed_objects":["envelopes","books"]},"metal":{"seed_objects":["cans","keys"]}}' \
  --output-root review_downloads/ollama_end_to_end_pipeline \
  --integrate-accepted
```
