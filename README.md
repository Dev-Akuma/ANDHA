# WLASL100 Prototype Pipeline

The downloaded videos are stored in `WSLA100/wlasl100-new/WLASL_100`, split into `train`, `val`, and `test` folders.

## Local setup

From the project folder, activate the virtual environment and install the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the first smoke test on five videos:

```powershell
python extract_landmarks.py --limit 5 --modality hands
```

The compressed outputs will be written to:

```text
WSLA100/processed_landmarks/hands/<split>/<gloss>/<video_id>.npz
```

Each file contains `keypoints` with shape `(30, features)`, plus `label`, `video_id`, and `split`. Use `--modality hand_face` or `--modality hand_face_pose` for the other planned inputs. Remove `--limit` only after the five-video test succeeds.

## Colab handoff

Upload this project code and `WSLA100/processed_landmarks` to Drive after local extraction. Run the extractor in Colab for the full dataset if the raw videos are available there; run training from the compressed landmark files rather than loading all raw videos into memory.