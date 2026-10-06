# Music Genre Classification

Classifies a music clip into one of 10 genres using the GTZAN dataset. Audio is converted into log-mel spectrograms and handcrafted features, then classified with classical ML models and a convolutional neural network (CNN). A Streamlit app lets you upload a song and see the predicted genre.

**Course:** UE24CS352A Machine Learning, Mini-Project
**Presented by:** Architha S (PES2UG24AM024) Arjun D Kumar (PES2UG24AM026)

---

## 1. Problem Statement

Given an audio clip, predict its genre out of: `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`.

Automatic genre labeling supports search, recommendation and library management on music platforms, where labeling by hand does not scale.

## 2. Dataset

[GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification): 1000 audio tracks of 30 s each (100 per genre), mono, 22050 Hz, `.wav`.

- One file, `jazz.00054.wav`, is corrupt in the Kaggle copy. The code skips unreadable files automatically.
- GTZAN has known flaws (a few duplicates and mislabeled tracks, Sturm 2013), so treat the accuracy numbers as indicative.

## 3. Approach

```
GTZAN .wav files
   -> split by TRACK (70% train / 10% val / 20% test, per genre, seed 42)
   -> cut each track into ten 3 s segments
   -> features per segment:
        (a) 88 handcrafted features (MFCC, chroma, spectral contrast, centroid,
            bandwidth, rolloff, ZCR, RMS; mean and std of each)
        (b) log-mel spectrogram, 128 mels x 130 frames, scaled to [0, 1]
   -> models: KNN, Logistic Regression, Random Forest, SVM (on a)   |   CNN (on b)
   -> track-level prediction = average of the segment probabilities
   -> evaluation on held-out tracks + Streamlit demo
```

Key design choices:

- **Split by track, not by segment.** Segments of one song look almost identical, so mixing them across train and test would leak information and inflate accuracy.
- **Log-mel spectrograms.** The mel scale mimics human pitch perception, and the spectrogram turns audio into an image that a CNN can read.
- **Fair model selection.** Classical models are tuned with 5-fold `GroupKFold` (grouped by track) on train+val. The best model is chosen by CV score, never by test score.
- **Track-level voting.** One 3 s segment can mislead, so all segments of a track vote.

**CNN architecture:** 4 blocks of Conv(3x3) - BatchNorm - ReLU - MaxPool - Dropout(0.25) with 32/64/128/256 filters, then Global Average Pooling, Dense(128, L2) - Dropout(0.4) - Softmax. 423,466 parameters. Trained with Adam (1e-3), SpecAugment (random frequency and time masks), EarlyStopping and ReduceLROnPlateau.

## 4. Results

Evaluated on held-out test tracks (about 200). "Track" means the song-level prediction after averaging its segments.

| Model | CV acc (train+val) | Segment acc | Track acc | Track macro-F1 |
|---|---|---|---|---|
| KNN (PCA + distance weighting) | 0.672 | 0.682 | 0.790 | 0.785 |
| Logistic Regression | 0.694 | 0.707 | 0.815 | 0.810 |
| Random Forest | 0.714 | 0.734 | 0.825 | 0.821 |
| **SVM (RBF)** | **0.755** | **0.777** | **0.855** | **0.853** |
| CNN (log-mel) | n/a | 0.763 | 0.825 | 0.819 |

Confusion matrices are saved as `cm_*.png`, and CNN training curves as `cnn_curves.png`.

**Reading the results:** the SVM is slightly ahead of the CNN, but with about 200 test tracks the difference (roughly 6 tracks) is within statistical noise, so the two are effectively tied. The hand-tuned classical features are strong on a dataset this small; the CNN's advantage would grow with more data or pretraining.

## 5. Project Structure

```
.
├── common.py              # shared config, audio features, evaluation helpers
├── extract_features.py    # step 1: split by track, extract features + mel-spectrograms
├── train_classical.py     # step 2: KNN / LogReg / RandomForest / SVM with grouped CV
├── train_cnn.py           # step 3: CNN on log-mel spectrograms
├── app.py                 # step 4: Streamlit demo
├── requirements.txt
├── best_classical.joblib  # trained best classical model (SVM-RBF)
├── cnn_best.keras         # trained CNN
├── classical_results.json, cnn_results.json
├── cm_*.png, cnn_curves.png
└── demo_songs/            # sample test clips, named <track>__predicted-<genre>.wav
```

## 6. Setup

Requires Python 3.10+.

```bash
git clone https://github.com/prithzzz/Music-Genre-Classification
cd Music-Genre-Classification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For `.mp3` uploads in the demo, install `ffmpeg` (for example `sudo apt install ffmpeg`). `.wav` works without it.

## 7. Run

### Option A: just run the demo (trained models are included)

```bash
streamlit run app.py
```

Open the local URL, pick a model (CNN or classical), and upload a song. Try the files in `demo_songs/`, which are held-out test clips.

### Option B: reproduce everything from scratch

1. **Download GTZAN** from Kaggle and place the genre folders so you have `data/genres_original/<genre>/*.wav`. Using the Kaggle API:

   ```bash
   pip install kagglehub
   python -c "import kagglehub; print(kagglehub.dataset_download('andradaolteanu/gtzan-dataset-music-genre-classification'))"
   ```

   Copy the printed folder's `genres_original` into `data/`.

2. **Run the pipeline in order:**

   ```bash
   python extract_features.py   # writes data/features.csv and data/mel.npy   (~10 min)
   python train_classical.py    # writes best_classical.joblib, classical_results.json, cm_*.png
   python train_cnn.py          # writes cnn_best.keras, cnn_results.json, cnn_curves.png (GPU recommended)
   streamlit run app.py
   ```

Everything is seeded (`SEED = 42` in `common.py`), so the split is reproducible.

### Google Colab

Use a GPU runtime (T4). Clone the repo, install requirements, download the dataset as above, and run the three scripts with `!python <script>.py`.

## 8. Limitations

- GTZAN is small and old (30 s clips from around 2000), so accuracy on modern songs will be lower.
- Genre labels are subjective. The models mostly confuse overlapping genres, such as rock/country/blues and jazz/classical.
- Test set size is about 200 tracks, so results carry roughly +/-3% uncertainty.
- Each 3 s segment is normalized by its own maximum, which discards absolute loudness.

## 9. Future Work

Fine-tune a pretrained audio or image model, add overlapping segments, evaluate with repeated splits or cross-validation, and test on a larger dataset such as FMA.

## 10. References

- Tzanetakis & Cook, *Musical genre classification of audio signals*, IEEE Trans. Speech and Audio Processing, 2002.
- Sturm, *The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use*, 2013.
- Huang, Pugh & Serafini, *Music Genre Classification*, Stanford CS229 project report, 2018 (reference for the approach).
- McFee et al., *librosa: Audio and Music Signal Analysis in Python*; scikit-learn; TensorFlow/Keras; Streamlit.