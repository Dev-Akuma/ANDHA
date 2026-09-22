# WLASL100 Sign Language Recognition Project

## 1. Project Purpose

This repository is a prototype pipeline for sign-language recognition using the WLASL100 video dataset. The intended research workflow is:

1. Download and verify WLASL100 videos.
2. Extract fixed-length MediaPipe Holistic landmark sequences.
3. Generate controlled synthetic landmark augmentations.
4. Train a sequence classifier using hands, hand plus face, or hand plus face plus pose inputs.
5. Compare 12 experimental conditions and report accuracy, macro-F1, and inference latency.

The recommended execution model is hybrid:

- **Laptop:** write, debug, and validate the pipeline using approximately five videos.
- **Google Colab T4 GPU:** perform full landmark extraction when needed, train models, generate the experiment matrix, and save the final results.

This prevents long training runs from stressing the laptop and keeps raw-video handling separate from the lightweight landmark data used for training.

## 2. Repository Contents

| File or directory | Purpose | Current state |
| --- | --- | --- |
| `notebook.ipynb` | Guided notebook containing setup, extraction functions, augmentation, data loading, training, metrics, experiment matrix, and research notes | Complete scaffold; expensive actions are disabled by default |
| `extract_landmarks.py` | Command-line MediaPipe extractor | Usable for local smoke tests and full runs; includes MediaPipe Tasks fallback |
| `augmentations.py` | Reusable NumPy transformations | Complete prototype module |
| `model.py` | Modality-agnostic PyTorch BiGRU classifier | Complete baseline model |
| `simulate_research.py` | Controlled end-to-end rehearsal using five extracted sequences | Executed successfully |
| `requirements.txt` | Python dependencies | Includes NumPy, OpenCV, MediaPipe, PyTorch, scikit-learn, pandas, and tqdm |
| `README.md` | Short setup guide | Basic setup and Colab handoff |
| `AGENT_WORKLOG.md` | Change history and handoff notes | Updated during implementation |
| `WSLA100/wlasl100-new` | Downloaded raw WLASL100 package | Present locally |
| `WSLA100/processed_landmarks` | Compressed extracted landmarks | Contains the five-video hands-only rehearsal output |
| `simulation_results.csv` | Output from the controlled rehearsal | Present; not a publishable result |

The `.venv` directory is a local environment and should not be uploaded to Colab or committed to source control.

## 3. Dataset Status

The downloaded package is located at:

```text
WSLA100/wlasl100-new/WLASL_100/
```

Its current split inventory is:

| Split | Videos observed |
| --- | ---: |
| `train` | 1,001 |
| `val` | 242 |
| `test` | 200 |
| **Total** | **1,443** |

The package is organized as:

```text
WLASL_100/
  train/<gloss>/<video_id>.mp4
  val/<gloss>/<video_id>.mp4
  test/<gloss>/<video_id>.mp4
```

This is already compatible with the extractor. The original planning estimate of approximately 2,000 videos should be replaced with the verified count of 1,443 for this downloaded package. The dataset still represents the WLASL100 class subset, but exact counts should be measured from the files rather than assumed.

The notebook's KaggleHub cell downloads `thtrnphc/wlasl100-new`, detects either the returned directory or a `versions/1` child, and copies the contents into `WSLA100/wlasl100-new`.

## 4. Landmark Representation

The extractor samples each video to 30 frames. Missing landmark groups are represented by zero coordinates so every saved sequence has a stable shape.

The current feature widths are:

| Modality | Landmark groups | Values per frame |
| --- | --- | ---: |
| `hands` | 21 left-hand + 21 right-hand landmarks, each with xyz | 126 |
| `hand_face` | Hands + 468 face landmarks, each with xyz | 1,530 |
| `hand_face_pose` | Hands + face + 33 pose landmarks, each with xyz | 1,629 |

The planned phrase “84 or 126 hand values” depends on whether coordinates are represented as xy or xyz. This implementation uses xyz, so hands-only input is 126 values per frame.

Output layout:

```text
WSLA100/processed_landmarks/
  hands/<split>/<gloss>/<video_id>.npz
  hand_face/<split>/<gloss>/<video_id>.npz
  hand_face_pose/<split>/<gloss>/<video_id>.npz
```

Each archive contains:

- `keypoints`: a float32 array shaped `(30, feature_width)`
- `label`: the gloss directory name
- `video_id`: the source video stem
- `split`: `train`, `val`, or `test`

Compressed `.npz` files are the current implementation choice. The research plan mentions `.npy`, but `.npz` is preferable here because it stores the array and metadata together and remains easy to load with NumPy.

## 5. What Has Been Implemented

### 5.1 Download and verification

The notebook can download the Kaggle dataset locally. The dataset verification cell checks the expected path, lists available directories when the path is wrong, and counts MP4 files.

### 5.2 MediaPipe extraction

`extract_landmarks.py` supports:

```powershell
python extract_landmarks.py --limit 5 --modality hands
python extract_landmarks.py --limit 5 --modality hand_face
python extract_landmarks.py --limit 5 --modality hand_face_pose
```

Important implementation details:

- Samples 30 frames per video.
- Resizes frames to a fixed `640 x 480` size before inference. This avoids MediaPipe temporal-graph failures when source videos change resolution between frames.
- Uses the legacy `mp.solutions.holistic` API when available.
- Falls back to MediaPipe Tasks `HolisticLandmarker` for newer MediaPipe installations such as the local `1.0.1` environment.
- Downloads and caches the Tasks model under `WSLA100/processed_landmarks/holistic_landmarker.task` when required.
- Pads short reads with the last valid frame so output remains fixed length.
- Preserves train, validation, and test directories in the output.
- Skips output archives that already exist, allowing interrupted jobs to resume.

### 5.3 Augmentation

`augmentations.py` and the notebook implement:

- Rotation between -15 and +15 degrees.
- Scaling between 0.8 and 1.2.
- Gaussian coordinate jitter with default standard deviation 0.01.
- Temporal stretch/compression between 0.8 and 1.2 while retaining 30 frames.
- Augmentation-count helpers for 25%, 50%, and 100% additional training samples.

Augmentation is applied only to training data. Validation and test samples must remain unmodified.

### 5.4 Model

`model.py` contains `BiGRUClassifier`:

```text
Input:       (batch, 30, feature_width)
BiGRU:       configurable hidden size, two directions
Normalization: LayerNorm
Regularization: Dropout
Output:      class logits
```

The input width is supplied at construction time, so the same class supports all three modalities.

### 5.5 Training and metrics

The notebook includes:

- PyTorch `Dataset` and `DataLoader` support for `.npz` archives.
- Lazy augmentation to avoid duplicating all augmented arrays in RAM.
- Cross-entropy loss.
- Adam optimizer.
- Accuracy.
- Macro-F1.
- Average evaluation latency in milliseconds per sample.
- Automatic CUDA selection when a GPU is available.

### 5.6 Controlled rehearsal

The five-video rehearsal completed successfully after fixing three environment-specific issues:

1. MediaPipe `1.0.1` does not expose the older `mp.solutions` namespace.
2. The Tasks API returned plain landmark lists instead of `.landmark` containers.
3. Some source videos changed dimensions or returned one fewer readable frame.

The rehearsal generated five hands-only archives, trained a compact five-class BiGRU fixture for two epochs, and wrote `simulation_results.csv`.

Observed rehearsal output:

| Metric | Value |
| --- | ---: |
| Validation accuracy | 0.400 |
| Validation macro-F1 | 0.280 |
| Test accuracy | 0.400 |
| Test macro-F1 | 0.280 |
| Test latency | approximately 2.11 ms/sample |
| Device | CPU |

These values are **pipeline checks only**, not research findings. The simulation deliberately uses repeated perturbed templates across splits, so it is not a valid evaluation protocol.

## 6. Current Gaps and Risks

1. **Full landmark extraction is not complete.** Only five hands-only videos have been processed. Full `hands`, `hand_face`, and `hand_face_pose` extraction remains to be run.
2. **The embedded notebook extractor is behind the standalone extractor.** The notebook still contains the original `mp.solutions.holistic` implementation and does not yet include the Tasks fallback, fixed-size resizing, or short-read padding. Use `extract_landmarks.py` for the next smoke tests until those cells are synchronized.
3. **No real baseline experiment has been run.** The 0% augmentation, hands-only experiment on the complete training split is still pending.
4. **No real 12-run matrix has been run.** The matrix code is present but disabled and requires all three modality datasets.
5. **No checkpointing or early stopping is implemented.** Long Colab runs should add model checkpoints, resume support, and best-validation tracking before full experiments.
6. **No formal video-integrity report exists.** The current checks count files but do not record unreadable videos, duration distributions, corrupt files, or class balance in a report.
7. **No paper assets exist yet.** The research notes are in the notebook, but the abstract, related work, equations, result tables, and IEEE/Overleaf project still need to be created.
8. **The stated 10x to 15x GPU speedup is a planning assumption.** It should be measured for this implementation and hardware rather than presented as a guaranteed result.

## 7. Exact Next Steps

### Step 1: Synchronize notebook extraction code

Copy the tested logic from `extract_landmarks.py` into the notebook extraction cells, or make the notebook call the script directly. The notebook path must support MediaPipe 1.x before it is used as the main runner.

### Step 2: Run all three local smoke tests

Run five videos for each modality:

```powershell
python extract_landmarks.py --limit 5 --modality hands
python extract_landmarks.py --limit 5 --modality hand_face
python extract_landmarks.py --limit 5 --modality hand_face_pose
```

For each modality, verify one archive:

```powershell
python -c "import numpy as np; from pathlib import Path; p=next(Path('WSLA100/processed_landmarks').glob('*/**/*.npz')); d=np.load(p); print(p, d['keypoints'].shape)"
```

Expected feature widths are 126, 1530, and 1629 respectively.

### Step 3: Produce a full extraction inventory

Run full extraction for each modality after the smoke tests pass. Record:

- Number of successful archives per split.
- Number and paths of failures.
- Total output size.
- Processing time per modality.
- Percentage of videos with missing hands, face, or pose landmarks.

This is the point to move to Colab if the laptop becomes too slow or thermally constrained.

### Step 4: Run the real baseline first

Run hands-only with 0% augmentation before the 11 other conditions. Save:

- Training configuration.
- Random seed.
- Epoch history.
- Best validation checkpoint.
- Test accuracy.
- Test macro-F1.
- Test latency.
- Device and software versions.

This becomes Run 1 and the first defensible result for the paper.

### Step 5: Add experiment durability

Before the matrix, add:

- Checkpoint saving after every epoch.
- Resume-from-checkpoint support.
- A results file written after every completed run.
- A run identifier and timestamp.
- GPU memory and runtime logging.
- A fixed seed policy.

This prevents losing all results if a Colab session disconnects during Run 8 or later.

### Step 6: Execute the 12-run matrix on Colab

The matrix is:

| Modality | Augmentation levels |
| --- | --- |
| Hands | 0%, 25%, 50%, 100% |
| Hands + face | 0%, 25%, 50%, 100% |
| Hands + face + pose | 0%, 25%, 50%, 100% |

Upload the code and processed landmark archives, select a T4 runtime, set `RUN_EXPERIMENTS = True`, and save `experiment_results.csv` after every run.

### Step 7: Write the research draft

Use the measured data to write:

- Section 3: preprocessing, landmark representation, augmentation equations, BiGRU architecture, training configuration, and metrics.
- Section 4: baseline and matrix results, class distribution, runtime, and latency.
- Limitations: small pilot status, MediaPipe missing landmarks, modality dimensionality, and hardware dependence.

## 8. Team Responsibilities

### Data and Pipeline Lead

- Maintain dataset paths and integrity reports.
- Keep `extract_landmarks.py` and notebook extraction code synchronized.
- Run full extraction for all three modalities.
- Deliver the processed landmark dataset and extraction statistics.

### Augmentation and Mathematics Lead

- Validate rotation, scale, jitter, and temporal equations.
- Confirm augmentation counts and training-only application.
- Document the transformations in the methodology section.

### Modeling and Training Lead

- Add checkpointing and early stopping.
- Validate the loader and BiGRU on the real baseline data.
- Run the 12 matrix conditions on Colab.
- Maintain metrics and runtime logs.

### Paper and Writing Lead

- Create the IEEE/Overleaf project.
- Write abstract, introduction, and related work.
- Convert measured results into tables and figures.
- Keep claims tied to recorded experiments rather than planning assumptions.

## 9. Definition of Done for Phase 1

Phase 1 is complete when:

- All 1,443 videos have an integrity status.
- All three modality datasets have been extracted or documented with failures.
- The hands-only 0% augmentation baseline has a saved checkpoint and metrics.
- All 12 runs have persisted result rows.
- Accuracy, macro-F1, latency, device, seed, and runtime are recorded.
- The notebook and standalone scripts produce the same preprocessing behavior.
- The research draft contains measured methodology and baseline results.