# Interview drill

The answers are short on purpose. The full reasoning for each is in `NOTES.md` under the number in brackets.

## 60-second pitch

> I built a glaucoma screening model for fundus photos. I trained EfficientNet-B0 on ORIGA, 650 images from Singapore, and I tested it externally on 257 images from a New Delhi eye hospital that it never saw. Glaucoma shows at the optic disc, so I localise the disc with a classical blob detector and crop it. An ablation showed that crop matters most on the external data: AUC 0.863 vs 0.775 for the full image. Internal test AUC is 0.832, in line with published ORIGA results. The most interesting finding is that the ranking transferred to the new hospital but the calibration didn't: the threshold tuned in Singapore gives only 25% specificity in Delhi. So in deployment you'd re-calibrate per site. It is served through a Flask app that never stores images, shows Grad-CAM, and displays only metrics computed by the evaluation script.

## Draw this on the whiteboard

```
raw fundus (3072×2048)
  └─ crop black border, pad square ............................ [5]
      └─ DoG blob detector → optic-disc centre → 0.40·D crop ... [6]
          └─ resize 256 (cache) → 224, ImageNet normalise ...... [9][10]
              └─ EfficientNet-B0 (ImageNet-pretrained, fine-tuned) [13]
                  └─ 1 logit → Platt scaling → P(glaucoma) ...... [14][19]
                      └─ ≥ threshold (85% val sensitivity)? → refer [18]
                      └─ Grad-CAM on last conv block ............ [25]
```

## Rapid-fire questions

| Question | Answer in one or two lines |
|---|---|
| Why crop the optic disc? | Glaucoma signs (cup-to-disc ratio, rim thinning) sit at the disc. At 224 px a full fundus leaves it about 25 px wide. The ablation confirmed it: external AUC 0.863 vs 0.775. [6][7] |
| Why EfficientNet-B0? | Best ImageNet accuracy per FLOP in its class: 5.3 M params vs 11.7 M for ResNet18, and it fits CPU-only training. [13] |
| Why pretrained? | 455 training images is far too few to learn features from scratch, and low-level ImageNet filters transfer. [13] |
| Why AUC and not accuracy? | 74% of images are normal, so predicting "normal" always scores 74% accuracy. AUC is independent of threshold and prevalence. [17] |
| How did you pick the threshold? | On validation, the highest threshold reaching 85% sensitivity. In screening a miss costs much more than a false alarm. [18] |
| What is calibration and why Platt over temperature scaling? | It makes probabilities mean what they say. `pos_weight` shifts the logits, and only Platt's bias term removes that shift. AUC is unchanged. [19] |
| How did you handle class imbalance? | `pos_weight ≈ 2.86` in BCE; the resulting probability bias is fixed afterwards by calibration. [22] |
| How do you know it generalises? | Through an external test on a different hospital, country and camera (VEIRC, New Delhi). [3] |
| What is data leakage and how did you avoid it? | Evaluation data influencing training or choices. All choices were made on validation, augmentation happens on the fly, and the external set comes from another site. [4][24] |
| What does Grad-CAM show, and what are its limits? | It shows *where* the evidence for the decision was, at 7×7 resolution. It does not show *what* the model measured. [25] |
| A bug you found? | timm initialises a 1-output head as U(−1, 1), which gave an initial loss of 4.4 instead of about 0.7. Re-initialising the head fixed it. [14] |
| Why is `weights_only=True` important? | `.pth` files are pickles, and loading an untrusted pickle can execute code. [16] |
| Why not `debug=True` in Flask? | The Werkzeug debugger lets anyone who triggers an error run Python on the server. [28] |
| Train/serve skew? | One `GlaucomaPredictor` code path is shared by the app and the evaluation. It handles BGR→RGB and stores preprocessing settings in the checkpoint. [8] |
| How confident are you in the AUC? | Bootstrap 95% CIs are reported. With 25 to 32 positives they are wide, roughly ±0.1. [20] |
| Biggest limitations? | Small data, image-level splits, a single seed, and a heuristic disc crop. Also, glaucoma diagnosis needs IOP, visual fields and OCT, so this can only triage. [31] |
| What would you do next? | More data (REFUGE, G1020, AIROGS), disc/cup segmentation for an explicit CDR, seed ensembles, a quality/OOD gate, and ONNX export. [31] |
| Why no ROP? | There was no credible public ROP dataset available. It needs a different camera (RetCam) and population, so it would be a separate model. [32] |

## If asked about the original repo

Keep it factual and neutral. For example: "I started from a public demo, checked its claims against the code, and found the metrics were placeholders and the architecture in the README didn't match. So I rebuilt it with a reproducible pipeline and an honest evaluation." See [1].

## Numbers to know cold

| | Internal test (ORIGA) | External (VEIRC) |
|---|---|---|
| AUC [95% CI] | 0.832 [0.75–0.90] | 0.863 [0.81–0.91] |
| Sensitivity / specificity at the validation threshold | 92% / 60% | 100% / 25% |
| Full-fundus AUC (ablation) | 0.800 | 0.775 |
| ECE before → after Platt | 0.25 → 0.06 | 0.49 → 0.20 |

Training set: 455 images (118 glaucoma). Validation: 97 (25). Test: 98 (25). External: 257 (32), plus 107 suspects at median P = 0.35. Best epoch: 6. Learning rate 1e-4, chosen over 3e-4 on validation. Training takes about 40 s per epoch on 2 CPU cores. [34]
