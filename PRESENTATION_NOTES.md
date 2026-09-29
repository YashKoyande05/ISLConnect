# Presentation Notes

## 30-second explanation

ISLENS is a computer-vision-based Indian Sign Language recognition system. In Phase 1, it recognizes individual alphabet and number signs and converts them into text. OpenCV handles image input, MediaPipe detects 21 hand landmarks, and the normalized landmark coordinates form a 63-feature vector for a lightweight neural-network classifier.

## Why MediaPipe?

It gives us structured hand geometry instead of requiring the classifier to learn hand localization from raw pixels.

## Why Phase 2?

Character recognition is not the same as continuous ISL translation. Phase 2 will preserve the landmark representation but process a sequence of landmarks over time using LSTM/GRU or Transformer models.

## If asked about the webcam

The web application supports both browser camera input and image upload. The upload mode makes the demo independent of camera hardware.

## If asked about accuracy

Only report the accuracy printed by `training.train` on the actual dataset used. Do not claim an accuracy before training and validation.
