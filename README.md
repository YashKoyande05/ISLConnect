# 🤟 ISLConnect — Indian Sign Language Recognition

**Phase 1: Character-level Indian Sign Language recognition using Computer Vision**

ISLConnect is a Python-based computer vision web application that recognizes Indian Sign Language hand signs and converts recognized characters into text.

The Phase 1 system focuses on **A–Z and 0–9 character recognition**. It uses MediaPipe to detect 21 hand landmarks, converts them into a normalized 63-dimensional feature vector, and classifies the vector using a lightweight neural-network classifier.

> **Important:** Phase 1 is character/fingerspelling recognition. It is not yet continuous natural-language ISL translation. Phase 2 is designed around temporal landmark sequences for word and sentence recognition.

---

## ✨ Features

- A–Z character recognition
- 0–9 number recognition
- MediaPipe hand landmark detection
- 21 landmarks × 3 coordinates = 63 features
- Wrist-relative landmark normalization
- MLP neural-network classifier
- Confidence score
- Image upload mode
- Browser camera mode
- Prediction stabilization
- Character-to-text construction
- Delete, space and clear controls
- Responsive dark web interface
- Modular architecture prepared for Phase 2

---

## 🧠 Architecture

```text
Image / Browser Camera
        ↓
      OpenCV
        ↓
    MediaPipe
        ↓
  21 Hand Landmarks
        ↓
 Landmark Normalization
        ↓
  63-D Feature Vector
        ↓
 MLP Classifier
        ↓
 A-Z / 0-9
        ↓
 Prediction Stabilization
        ↓
      Text
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Flask | Web backend/API |
| OpenCV | Image processing |
| MediaPipe | Hand landmark detection |
| scikit-learn | MLP classifier |
| NumPy | Numerical processing |
| Pandas | Dataset processing |
| HTML/CSS | Frontend |
| JavaScript | Browser interaction |

---

# 🚀 Setup

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ISLConnect
```

## 2. Create a virtual environment

Python 3.11 is recommended.

Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# 📂 Dataset

Place the Phase 1 image dataset in:

```text
dataset/raw/
```

The folder structure must be:

```text
dataset/raw/
├── 0/
├── 1/
├── 2/
├── ...
├── 9/
├── A/
├── B/
├── ...
└── Z/
```

Each folder should contain images of that sign.

The folder name becomes the class label.

### Dataset note

Public ISL datasets differ in their class sets. Some commonly used datasets contain A–Z and 1–9 but do not contain 0. If your selected dataset has no `0` class, the model will contain the classes actually present in the dataset. A separate 0-class dataset can be added later.

Always verify dataset licensing and attribution requirements before redistributing dataset files.

---

# 🔎 Test MediaPipe before training

Put any clear hand image at:

```text
dataset/raw/test.jpg
```

Run:

```powershell
python -m training.test_mediapipe
```

Expected:

```text
Hand detected!
Landmarks: 21
Feature vector length: 63
```

A window should display the detected hand landmarks.

---

# 🧮 Extract landmarks

After the dataset has been placed in the correct folders:

```powershell
python -m training.extract_landmarks
```

This creates:

```text
dataset/processed/landmarks.csv
```

The CSV contains:

```text
f0 ... f62,label
```

That means:

```text
63 landmark features + class label
```

---

# 🤖 Train the Phase 1 model

Run:

```powershell
python -m training.train
```

The training script:

1. Loads landmark features
2. Encodes class labels
3. Performs a stratified train/test split
4. Trains an MLP classifier
5. Prints accuracy
6. Prints a classification report
7. Prints a confusion matrix
8. Saves the trained model

Generated files:

```text
model/
├── isl_model.joblib
└── labels.json
```

---

# ▶️ Start the web application

After training:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 🖼️ No webcam?

No problem.

Use:

```text
UPLOAD → Choose image → Recognize sign
```

The browser camera mode is optional.

---

# 📊 Model

The current Phase 1 classifier uses:

```text
Input:       63 features

Dense:       128 neurons
Activation:  ReLU

Dense:       64 neurons
Activation:  ReLU

Output:      number of dataset classes
Activation:  probability distribution
```

The model uses scikit-learn's `MLPClassifier`.

---

# 🧩 Why MediaPipe landmarks?

Raw image classification requires the model to learn both:

1. Where the hand is
2. What the hand shape means

MediaPipe already solves the hand localization problem.

We therefore represent the hand as:

```text
21 landmarks × X,Y,Z
= 63 features
```

The landmarks are translated relative to the wrist and scaled to reduce sensitivity to image position and hand size.

---

# 🗺️ Roadmap

## Phase 1 — Character Recognition

- [x] Project architecture
- [x] MediaPipe hand detection
- [x] 21-landmark extraction
- [x] Landmark normalization
- [x] Training pipeline
- [x] MLP classifier
- [x] Flask API
- [x] Upload interface
- [x] Browser camera interface
- [x] Confidence filtering
- [x] Text construction
- [ ] Expand/verify A–Z + 0–9 dataset
- [ ] Improve validation accuracy

## Phase 2 — Word Recognition

The Phase 1 architecture intentionally prepares the system for temporal recognition.

```text
Video
 ↓
MediaPipe
 ↓
Landmark sequence
 ↓
LSTM / GRU / Transformer
 ↓
Word
```

Planned improvements:

- Two-hand recognition
- MediaPipe Holistic
- Temporal landmark sequences
- Word-level ISL datasets
- LSTM/GRU baseline
- Transformer-based experiments
- Common ISL vocabulary

## Phase 3 — Sentence Translation

```text
Continuous Signs
      ↓
Temporal Recognition
      ↓
ISL Gloss / Sign Sequence
      ↓
Language Processing
      ↓
English Sentence
```

Potential features:

- Sentence generation
- Context-aware prediction
- Grammar transformation
- Text-to-speech
- User vocabulary/custom signs

---

# ⚠️ Limitations

The Phase 1 system recognizes individual character/number signs.

It should not be described as a complete Indian Sign Language translator.

Natural ISL communication contains temporal movement, two-handed signs, facial/body cues and linguistic structure. These will be addressed progressively in Phase 2 and Phase 3.

---

# 📚 References

- MediaPipe documentation
- OpenCV documentation
- scikit-learn documentation
- Indian Sign Language Research and Training Centre (ISLRTC)
- Public Indian Sign Language datasets
- Research literature on Indian Sign Language recognition

---

## 👨‍💻 Project

**ISLConnect**

Indian Sign Language Recognition using Computer Vision

Built with:

`Python` `Flask` `OpenCV` `MediaPipe` `scikit-learn` `JavaScript`
