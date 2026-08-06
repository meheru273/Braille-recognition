# `inference/` — run a trained model on new images

- `infer.py` (planned): load fine-tuned RF-DETR (`RFDETRMedium(pretrain_weights=<ckpt>)`),
  `model.predict(img, threshold=...)` → `supervision.Detections` → draw boxes / emit COCO.
- Reuse the `annotate/crop_page.py` + `enhance.py` front-end so inference matches training preprocessing.
- Later: `model.export()` → ONNX for a lighter deployment, and wire the fine-tuned weights into the
  HF Gradio Space via `huggingface_hub.hf_hub_download` (Phase 6) — the Space itself stays as-is for now.
