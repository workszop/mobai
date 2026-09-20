# KlockiAI Mobile

Touch-first version of [klocki](https://github.com/workszop/klocki): train an image classifier on your phone, in the browser, with transfer learning (MobileNetV3-Small features + a small dense head in TensorFlow.js). Live at https://mobi.workszop.org.

## How it works

1. **Data** – start the camera, hold a class button to record frames (square center crop, 224 px). Each frame is embedded immediately with MobileNet, so only a 1024-float feature vector and a 64 px thumbnail are kept. The dataset is autosaved to IndexedDB and restored on the next visit.
2. **Train** – 20% of each class is held out as a test set. The classifier trains on the rest and reports test accuracy plus a confusion matrix. Training takes about a second because features are precomputed.
3. **Save** – the classifier is auto-saved in the browser under a name. Export as a small model file, or as a full bundle that includes the base model for fully offline use on another device.
4. **Inference** – load a saved model (or a file), start the camera, see live predictions with a confidence threshold.

## Files

- `index.html` – the whole app (Polish primary, English secondary, `T = { pl, en }`).
- `sw.js` – service worker caching the app shell and pinned CDN libraries for offline use. The base model is cached in IndexedDB by the app itself.
- `site.webmanifest` + icons – PWA install packaging (white Lucide camera on black).

## Development

Serve over http (camera needs a secure context or localhost):

```
python3 -m http.server 8080
```

Automated checks can read state from `window.KLOCKI.state` and the `data-*` attributes on `<body>` (`data-screen`, `data-base-status`, `data-train-phase`), class cards (`data-count`, `data-pending`) and step cards (`data-state`). Chrome's `--use-fake-device-for-media-stream --use-fake-ui-for-media-stream` flags give a synthetic camera for headless runs.
