# KlockiAI Mobile

> Mobile transfer learning in your browser — no server, no install.

Train a custom image classifier using your phone camera and MobileNetV3-Small. Works entirely client-side; all processing happens on your device.

## How it works

1. **Training** — capture photos with your camera for each class, download the base model (~3 MB, one-time), then train a dense classifier on top of the extracted features
2. **Inference** — load a saved model and classify live camera frames in real time with configurable FPS and confidence threshold

## Features

- Runs entirely in the browser via [TensorFlow.js](https://www.tensorflow.org/js)
- MobileNetV3-Small feature extractor (~3 MB, cached after first download)
- 2× data augmentation (brightness jitter + horizontal flip) via Web Worker
- Up to 6 custom classes; configurable epochs, learning rate, and batch size
- Save models to browser IndexedDB or export as a portable `.json` bundle
- PWA-ready — add to home screen on iOS and Android
- Dark mode support

## Requirements

Camera requires **HTTPS** or `localhost`. Works on Chrome, Safari, and Firefox on mobile and desktop.

## Usage

Open `index.html` via a local server or visit the deployed GitHub Pages URL.
