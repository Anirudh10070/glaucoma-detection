# NOTES: one note per design decision

Every `[^N]` in the code and README points to section **N** below. Each note says **what** was done, **why**, and **what an interviewer is likely to ask**, with a short answer you can give.

---

## [1] What was wrong with the original repo

The original `Web-Nethralaya_V1` (MIT, © 2025 Takshak) was a working Flask demo, but its README did not match the code:

| README claim | Reality in the code |
|---|---|
| ResNet50 | `torchvision.models.resnet18` (the 44.7 MB checkpoint has no `conv3` / `layer4.2`, so it is a ResNet18) |
| Glaucoma **and ROP** | 2-class glaucoma vs normal only. "ROP" appears nowhere in the code |
| Training notebook `G_L.ipynb` | Not in the repo, so the model cannot be reproduced |
| Accuracy 97.28%, F1 0.95, AUC 0.997 | A hard-coded Python dict commented `# example`, shown as "Model Performance" on every result |
| "No data storage, images deleted" | Uploads were saved to `static/uploads/` and never deleted, and about 20 of them were committed to git |
| "Suitable for clinical environments" | `app.run(debug=True)`, no upload validation, and same-name uploads overwrote each other |

There were also smaller bugs: Grad-CAM colours were swapped (BGR/RGB), it used the deprecated `register_backward_hook`, and the 0.5/0.5 normalisation did not match an ImageNet backbone.

**Interview angle.** *"Why rebuild instead of fix?"* Nothing in it could be reproduced (no data pipeline, no training code, no evaluation), and the numbers were invented. The only parts kept are the idea (upload a fundus image, get a prediction and a Grad-CAM map) and the name.

---

## [2] Training data: ORIGA

- **What:** ORIGA-light. 650 colour fundus photos (168 glaucoma, 482 normal) from the Singapore Eye Research Institute, taken 2004–2007 (Zhang et al., EMBC 2010, doi:10.1109/IEMBS.2010.5626137). The images are 3072×2048.
- **How it is obtained:** official access is "on request", so `prepare_data.py` clones a public GitHub mirror (`Quera-fr/Glaucoma-Dataset`). We checked that it is ORIGA: 650 images, exactly 168/482, and every image is 3072×2048. **No image is committed to this repo**, because neither dataset allows redistribution.
- **Why ORIGA:** it is the most widely used public glaucoma benchmark, it has full-field (not pre-cropped) images, so the preprocessing pipeline is exercised, and published results exist to compare against.
- **Benchmark context:** Chen et al. (EMBC 2015) reported **AUC 0.831** on ORIGA with a 6-layer CNN, and a later contextual-CNN variant from the same group reported 0.838. ORIGA is known to be hard: cases are often early or borderline, and the labels come from clinical diagnosis, not from the image alone.

**Interview angle.** *"Your AUC is lower than the 0.99 people post on Kaggle. Why?"* Many of those results come from easy datasets, leakage (augmented copies of one image in both train and test), or reporting on the validation set they tuned on. On ORIGA, published CNNs sit in the low-to-mid 0.8s.

---

## [3] External test set: VEIRC (New Delhi)

- **What:** the glaucoma dataset released by Prof. M. K. Dutta's group (Venu Eye Institute & Research Centre, New Delhi), from `ProfMKD/Glaucoma-dataset`. It has 32 glaucoma (vertical CDR ≥ 0.7), 225 normal (CDR ≤ 0.4) and 107 suspect (CDR 0.4–0.7) images. It is research-use only and must be cited (Singh et al. 2016, CMPB 124:108–120; Issac et al. 2015, CMPB 122:229–244).
- **Why:** an internal test split comes from the *same* hospital, camera and population as training, so it overestimates real-world performance. VEIRC differs in country, ethnicity, camera, field of view, zoom and label definition (CDR thresholds vs ORIGA's clinical grading). That is a genuine **domain shift**. It is used **only** for final reporting, never for training, tuning or model selection.
- **Suspects:** there is no binary ground truth for them, so they are excluded from AUC/sensitivity. `evaluate.py` reports their median probability and referral rate. A sensible model puts them between the normals and the glaucomas.
- **Caveat:** only 32 positives, so the external AUC has a wide confidence interval (see [20]).

**Interview angle.** *"How do you know the model generalises?"* Through external validation on data from another site. The drop from internal to external performance is the most informative number in this project.

---

## [4] Train / validation / test split

- **What:** ORIGA is split 70/15/15 (455 / 97 / 98) with `train_test_split(..., stratify=label, random_state=42)`, so every split keeps the ~26% glaucoma rate. The mirror's own Train/Validation folders are ignored.
- **Why three sets:** **train** fits weights. **Validation** chooses the epoch (early stopping), the calibration, the threshold and the view (disc vs full). **Test** is touched once, at the end. Using test for any decision leaks information and inflates results.
- **Limitation:** splits should be by **patient**, because both eyes of one person are correlated. The ORIGA mirror has no patient IDs, so the split is by image. That is one more reason the external set [3] matters.

**Interview angle.** *"What is data leakage, and how did you prevent it?"* Information from evaluation data influencing training or choices. Here: all choices are made on validation, augmentation is applied on the fly so augmented copies cannot land in test, and the external set comes from a different hospital.

---

## [5] Fundus border crop

- **What:** pixels with max(R,G,B) ≤ 20 are treated as black camera border. A morphological opening removes specks, the bounding box of the largest region is cropped, and the result is zero-padded to a square.
- **Why:** ORIGA images are 3:2 with large black side borders. Resizing them straight to a square distorts the circular retina into an ellipse and wastes pixels. Cameras differ in border size, so this also normalises across datasets.
- **Guard:** if the detected region is under 20% of the image (for example a very dark photo), the crop is skipped rather than cutting to garbage.

---

## [6] Optic-disc localisation and crop

- **Clinical reason:** glaucoma is diagnosed from the optic nerve head: the cup-to-disc ratio (CDR), rim thinning (ISNT rule) and vessel bending at the disc edge. At 224 px, a whole-fundus image leaves the disc only about 25–30 px wide, and the cup detail is lost. A disc crop spends all 224 px on the region that matters.
- **Method:** a classical **Difference-of-Gaussians** blob detector on the 0.5·R + 0.5·G channel: `blur(σ = size/50) − blur(σ = size/10)`, arg-max inside the eroded field of view. The small blur matches the disc's size; subtracting the large blur removes broad illumination. The first version (arg-max of a single blur) locked onto **flash glare on the field-of-view rim** in several images. DoG fixed those on visual inspection of about 250 images. A unit test (`test_locate_disc_ignores_rim_glare`) reproduces that failure mode.
- **Crop size:** a square of side 0.40 × fundus diameter, roughly 2.5–4 disc diameters, so the peripapillary area (where RNFL defects show) is included. The window is clamped inside the image rather than padded.
- **Known failures:** roughly 2% of crops still miss the disc (heavy glare, or bright exudates in VEIRC). Next step: a small learned disc detector, or a U-Net segmenting disc and cup, which would also give an explicit CDR.

**Interview angle.** *"Why not let the CNN find the disc itself?"* With 455 training images, the inductive bias of a domain-informed crop is worth more than the flexibility. The ablation in [7] tests that claim instead of assuming it.

---

## [7] Two views and the ablation

- **What:** every image is cached twice, as the `full` fundus and as the `disc` crop, both 256 px. The same model and recipe are trained on each, and they are compared on validation, internal test and external test.
- **Why:** this turns "disc cropping helps" from an assumption into a measured result. The deployed view is chosen on **validation AUC**, not test [24].
- **Result:** the disc crop won on every split (AUC): validation **0.818 vs 0.796**, internal test **0.832 vs 0.800**, external **0.863 vs 0.775**. The internal gap is within the confidence intervals. The external gap is the convincing one, since the intervals barely overlap ([0.81–0.91] vs [0.70–0.84]). The likely reason: a crop sized relative to the fundus diameter removes most of what differs between cameras (field of view, periphery, vignetting), while the full-fundus model partly learned ORIGA-specific context. More in [34].

---

## [8] Train/serve parity (one code path)

- **What:** the web app and `evaluate.py` both call `nethralaya.inference.GlaucomaPredictor`, which runs the same `preprocess()` used to build the training cache (256 px), then the same eval transform (resize to 224, ImageNet normalisation). `evaluate.py` deliberately reads the *raw* images, not the cache, so it tests exactly what the app runs.
- **Why:** **training/serving skew** is one of the most common production ML bugs. Examples are a different resize order, different colour order, or a different crop at inference. The checkpoint stores `view`, `img_size`, calibration and threshold, so the deployed artefact is self-describing.
- **BGR vs RGB:** OpenCV decodes images as **BGR**, while PIL/torchvision and the ImageNet statistics assume **RGB**. Every `cv2.imread`/`imdecode` is followed by `cvtColor(..., BGR2RGB)`. The original repo's Grad-CAM wrote RGB arrays with `cv2.imwrite` and blended a BGR colour map with an RGB image, so its colours were swapped.

---

## [9] Resolution: cache 256, train 224

- 224×224 is EfficientNet-B0's native resolution, so the pretrained filters see objects at the scale they were trained on. Each image is cached at 256 so `RandomResizedCrop` has margin to crop from.
- Evaluation resizes the whole 256 image to 224 with **no centre crop**. A centre crop could cut off part of an off-centre disc or the fundus edge.
- **Trade-off:** higher resolution (384+) helps fine disc detail but costs about 3× compute per image. We measured 0.11 s/img at 224 vs 0.32 s/img at 384 per training step on the 2-core CPU used here. The disc crop gets the detail benefit without the cost.

---

## [10] ImageNet normalisation

Inputs are normalised with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), the statistics the backbone was pretrained with. The first layer's filters and the BatchNorm running statistics expect inputs in that range. The original repo used mean = std = 0.5, which feeds the network a shifted distribution.

---

## [11] Data augmentation

| Augmentation | Why it is valid for fundus images |
|---|---|
| RandomResizedCrop (scale 0.75–1, near-square) | small zoom and position changes, like slightly different framing |
| Horizontal flip | a left-eye image mirrored looks like a right eye, so this is anatomically real |
| Rotation ±15° | head or camera tilt |
| Colour jitter (brightness/contrast 0.2, saturation 0.1, hue 0.02) | flash intensity, pigmentation and camera colour differences |

- **Deliberately not used:** vertical flip. It swaps superior and inferior, and the inferior/superior rim asymmetry (ISNT rule) is a real glaucoma sign. Strong hue shifts are also avoided, because pallor of the disc is diagnostic.
- Augmentation is applied **on the fly** each epoch, never by saving augmented copies, which would risk leaking copies across splits (see the original repo's `001_aug1.png`).

---

## [12] In-memory dataset

The 455 training images at 256² × 3 bytes take about 90 MB, so they are decoded once into RAM. On a 2-core CPU, PNG decoding every epoch would otherwise compete with the forward and backward pass. On a large dataset you would instead use `num_workers > 0` and possibly a format like WebDataset or LMDB.

---

## [13] Backbone: EfficientNet-B0 (ImageNet-pretrained)

- **What:** `timm` `efficientnet_b0` with the `efficientnet_b0_ra` ImageNet weights, downloaded from the timm author's GitHub release. It has 5.3 M parameters with the ImageNet head (4.0 M with our 1-logit head) and about 0.39 GFLOPs at 224, compared with 11.7 M and about 1.8 GFLOPs for ResNet18.
- **Why EfficientNet:** Tan & Le (ICML 2019) scale depth, width and resolution together ("compound scaling") from a NAS-designed base (B0). It is built from **MBConv** blocks (inverted residuals with depthwise convolutions) plus **squeeze-and-excitation** channel attention and SiLU activations. It gets better ImageNet accuracy per FLOP than ResNets, which matters on CPU-only hardware and would matter for a phone or edge deployment.
- **Why pretrained:** 455 images is far too few to learn visual features from scratch. Low-level ImageNet filters (edges, blobs, textures) transfer well even to medical images, so we **fine-tune all layers** with a small learning rate.
- **Interview angle:** *"Why not a ViT?"* Vision transformers need much more data or strong pretraining to beat CNNs, and they are heavier on CPU. At this data scale a small CNN is the right baseline.

---

## [14] Single-logit head, and an initialisation bug that was caught

- **What:** `num_classes = 1` with `BCEWithLogitsLoss`, so p = sigmoid(logit). A 2-way softmax is mathematically equivalent but adds a redundant parameter. One logit also makes calibration [19] and thresholding [18] simpler.
- **Bug found during development:** timm initialises EfficientNet's classifier as U(−1/√fan_out, +1/√fan_out), and with **fan_out = 1** that is U(−1, 1) over 1280 features. Initial logits were in the tens, and the first epoch's loss was **4.4** instead of about 0.7–1.0. The fix re-initialises the head with N(0, 0.01) weights and zero bias, so every initial prediction is about 0.5. After the fix, epoch-1 loss was 1.0 instead of 4.4, and epoch-1 validation AUC was 0.71 instead of 0.67.
- **Interview angle:** *"What is the expected initial loss of a binary classifier?"* With p = 0.5 it is −ln 0.5 ≈ 0.69 for unweighted BCE. Checking that number is a standard sanity check. This bug was found exactly that way.

---

## [15] Regularisation: dropout + stochastic depth + weight decay

- `drop_rate = 0.3`: dropout on the 1280-d pooled features before the classifier.
- `drop_path_rate = 0.1`: **stochastic depth** (Huang et al., 2016). During training, whole residual branches of MBConv blocks are randomly skipped, which acts like training an ensemble of shallower networks.
- AdamW weight decay 0.01 [23].
- Early stopping on validation AUC [24].

With 455 images, overfitting is the main risk: training loss keeps falling while validation AUC plateaus. `reports/history_<view>.csv` and `reports/learning_curves.png` show this.

---

## [16] Safe checkpoint loading

`torch.load(..., weights_only=True)`. A `.pth` file is a **pickle**, and unpickling can execute arbitrary code. The original repo's `torch.load("model.pth")` ran on a file downloaded from the internet. With `weights_only=True`, only tensors and primitive containers are allowed. PyTorch made this the default in 2.6; it is written explicitly here so it is visible.

---

## [17] Metrics: why AUC, and sensitivity at 85% specificity

- **AUC-ROC:** the probability that a random glaucoma image scores higher than a random normal image. It does not depend on a threshold or on class prevalence, which makes it the standard headline metric for screening models.
- **Accuracy is misleading here:** with 74% normals, predicting "normal" for everything scores 74%.
- **Sens@Spec 0.85** is the metric used by the RIM-ONE DL benchmark, reported so results are comparable with that literature.
- **Also reported:** sensitivity, specificity and a confusion matrix at the *chosen* threshold [18], plus ECE [19].

---

## [18] Operating threshold: target 85% sensitivity

- **What:** after calibration, the threshold is set on **validation** as the highest one that still reaches ≥ 85% sensitivity. Highest means the fewest false positives among the thresholds that meet the target.
- **Why not 0.5:** the threshold is a clinical decision, not a statistical one. In **screening**, a missed glaucoma (false negative) means irreversible vision loss, while a false positive costs one specialist visit. So we fix sensitivity and accept the resulting specificity.
- **Caveat:** validation has only 25 positives, so sensitivity moves in 4% steps and the chosen threshold is noisy. Test or external sensitivity can land well below 85% (see the results), which is exactly why the README reports them separately. A real deployment would set the threshold on a larger, local calibration set.

---

## [19] Calibration: Platt scaling, measured by ECE

- **Problem:** the probability should mean something: of images given 70%, about 70% should be glaucoma. Neural nets are usually **over-confident** (Guo et al., ICML 2017). Here there is an extra, deliberate bias: `pos_weight ≈ 2.9` [22] pushes every logit up by about ln 2.9, so raw probabilities over-call glaucoma.
- **Fix:** **Platt scaling**, p = σ(a·z + b), with *a* and *b* fitted on validation by minimising log-loss. Temperature scaling (b = 0) cannot remove the bias, which is why both parameters are used. Since a > 0 the ranking is unchanged, so **AUC is identical before and after**.
- **ECE** (expected calibration error) puts predictions into 10 confidence bins and averages |mean predicted − observed rate| weighted by bin size. `metrics.json` reports ECE before (`ece_uncalibrated`) and after (`ece`).
- **Caveat:** calibration is population-specific. A model calibrated on ORIGA is not necessarily calibrated on VEIRC, which has a different prevalence and camera. External ECE shows how much it shifts.

---

## [20] Bootstrap 95% confidence intervals

The test set is resampled with replacement 2000 times, AUC is recomputed each time, and the 2.5th and 97.5th percentiles are reported. With 25 internal or 32 external positives, the uncertainty is large (an AUC interval roughly ±0.08–0.1 wide). **A single AUC from a 98-image test set should never be quoted without an interval.**

**Interview angle:** *"Is model A better than model B?"* Only if the difference holds up beyond the noise. A paired test such as DeLong's, or a paired bootstrap on the same images, is the right tool. Overlapping intervals from two different evaluations are only a rough guide.

---

## [21] Reproducibility

Fixed seeds (Python, NumPy, torch, DataLoader generator), a fixed split seed, a pinned `requirements.txt`, and the full training history saved to `reports/history_<view>.csv`. Bit-exact reproduction across machines is still not guaranteed, because thread scheduling and BLAS kernels differ. The honest way to report robustness is to repeat with several seeds and report the spread. That was not done here because of CPU time, and it is listed in [31].

---

## [22] Class imbalance: `pos_weight`

There are 118 glaucoma and 337 normal training images. `BCEWithLogitsLoss(pos_weight = 337/118 ≈ 2.86)` weights each positive example's loss by about 2.86, so the two classes contribute equally to the gradient.

| Option | Trade-off |
|---|---|
| **Loss weighting (used)** | simple and needs no data duplication; it biases the probabilities, which [19] corrects |
| Oversampling positives | same effect, but repeats the same 118 images more often, so more overfitting risk |
| Focal loss | down-weights easy examples; more useful under extreme imbalance (e.g. 1:100) |
| Do nothing | AUC is threshold-free and fairly robust, but training may under-fit the minority class |

---

## [23] Optimiser and schedule: AdamW, warm-up, cosine

- **AdamW** (Loshchilov & Hutter) decouples weight decay from the adaptive gradient update. In plain Adam, L2 regularisation gets rescaled by the adaptive learning rate and stops acting like true weight decay.
- **lr 1e-4, batch 16:** the first run used 3e-4. Validation loss rose from epoch 4 while training loss collapsed, which is classic overfitting. A single re-run at 1e-4 (disc view, same seed) reached a best validation AUC of **0.818 vs 0.807**, with lower validation loss (0.84 vs 1.10 at the selected epoch). Both runs are in `reports/recipe_selection.json`. That AUC gap is well **within noise** for 25 validation positives. The choice rests mainly on the healthier loss curve, and only one alternative was tried, to avoid over-tuning on a small validation set. The test and external sets were not looked at for this decision.
- **One epoch of linear warm-up:** at step 0 the new head is random and Adam's second-moment estimates are poor, so large early updates can wreck the pretrained features.
- **Cosine decay to 0** (as in SGDR, without restarts): smooth annealing that ends training at a low learning rate.

---

## [24] Model selection and early stopping

- After each epoch, validation AUC is computed and the best weights are kept. Training stops if AUC does not improve for 7 epochs, with a maximum of 25.
- Selection uses **AUC, not loss**, because AUC is the metric we care about, and validation loss with `pos_weight` is a poorly scaled proxy.
- The **view** (disc vs full) that gets deployed is also chosen by validation AUC. Test and external numbers never influence any choice, so they remain unbiased estimates.

---

## [25] Grad-CAM

- **What:** Selvaraju et al. (ICCV 2017). Take the last convolutional feature maps A (here `bn2` output: 1280 × 7 × 7) and the gradient of the score with respect to them. Average the gradient over space to get one weight per channel, form a weighted sum of the maps, apply ReLU, upsample and normalise.
- **Which score:** for a "refer" prediction, the glaucoma logit. For "no referral", **−logit**, so the map always explains the decision that was actually shown.
- **Implementation:** a forward hook captures A and registers a **tensor** hook (`out.register_hook`) for its gradient. `Module.register_backward_hook`, which the original repo used, is deprecated and gives wrong gradients for modules that contain several operations.
- **Limits, worth saying out loud:** 7×7 resolution is coarse, and Grad-CAM shows *where* the evidence was, not *what* it was. A map centred on the disc is reassuring but does not prove the model measured the cup-to-disc ratio. Grad-CAM is a sanity check, not an explanation a clinician can rely on.

---

## [26] Privacy: uploads never touch the disk

- **What:** `request.files[...].read()` → `np.frombuffer` → `cv2.imdecode`. The images for the result page are sent back as base64 `data:` URLs. Nothing is written to disk, and `tests/test_app.py` checks that no file in the project changes after a request.
- **Why:** fundus images are health data. The original repo claimed this behaviour, but in fact saved every upload to `static/uploads/` and committed them to git. Under India's **DPDP Act 2023**, or HIPAA and GDPR elsewhere, persisting health images without consent and retention rules is a real compliance problem.
- **Still missing for production:** HTTPS, authentication, audit logging, and making sure no reverse proxy or log captures request bodies.

---

## [27] Metrics come from the evaluation report, never hard-coded

The footer table in the app is rendered from `reports/metrics.json`, which is written by `scripts/evaluate.py` for the exact checkpoint being served. A test asserts that the test AUC shown on the page equals the one in the report. This fixes the most serious problem in the original repo, where invented numbers were shown next to every medical prediction.

---

## [28] Web hardening basics

- `debug=False`: Flask's debug mode enables the Werkzeug **interactive debugger**, which lets anyone who can trigger an error run Python on the server. It must never be reachable in production.
- `MAX_CONTENT_LENGTH = 15 MB`: Flask rejects bigger requests with HTTP 413, so nobody can exhaust memory with a huge upload.
- Extension allow-list, plus checking that the bytes actually decode as an image.
- Binds to `127.0.0.1` by default. For deployment you would use a WSGI server (gunicorn or waitress) behind a reverse proxy, not `app.run`.

---

## [29] JSON API

`POST /api/predict` returns `{probability, referable, threshold, model_view}`. The UI is for demos; an API is how a screening system, a hospital PACS integration or a batch job would actually call the model. Grad-CAM is skipped there to keep latency low (it needs a backward pass).

---

## [30] Tests

- `tests/test_core.py` (no data needed) covers: the border crop is square and tight; the disc locator finds a synthetic disc **despite rim glare**; the disc crop stays inside the image when the disc is at the edge; the metric helpers; Platt scaling recovers known parameters; and ECE = 0 for a perfectly calibrated input.
- `tests/test_app.py` (needs `weights/`) covers: an end-to-end upload returns a result; **nothing is written to disk**; the JSON API works; bad uploads get HTTP 400; and the page metrics match the report.

Run them with `pytest -q`.

---

## [31] Limitations and next steps (say these before you are asked)

1. **Small data:** 455 training images, with 25 test and 32 external positives, so the confidence intervals are wide.
2. **Image-level split:** no patient IDs, so fellow-eye leakage is possible inside ORIGA [4].
3. **One seed:** variance across seeds was not measured [21].
4. **Heuristic disc crop**, which fails on about 2% of images [6].
5. **Glaucoma is not a fundus-only diagnosis:** clinicians combine IOP, visual fields and OCT (RNFL thickness). A fundus model can triage for referral; it cannot diagnose.
6. **No out-of-distribution check:** a non-fundus image still gets a probability. A next step is an image-quality and "is this a fundus" gate before the classifier.
7. **Next steps:** more training data (REFUGE, G1020, AIROGS / EyePACS-light), disc and cup segmentation for an explicit vertical CDR, seed ensembles, test-time augmentation, threshold re-calibration on local data, and ONNX or TFLite export for edge devices.

---

## [32] Why ROP was dropped

Retinopathy of prematurity affects premature infants. It is imaged with wide-field contact cameras (e.g. RetCam) and graded by zone, stage and "plus disease" (ICROP). It is a different camera, population and task from adult glaucoma, so it cannot share this model. The only public ROP data reachable while building this was a 91-image Kaggle sample with no patient information or grading provenance, which is not enough for a credible model. Claiming ROP support without data is what the original repo did, so this project states the scope instead.

---

## [33] Attribution and licences

- The original idea and name come from `Takshak-CS/Web-Nethralaya_V1` (MIT). This repo is a from-scratch rewrite; none of the original code is reused.
- **ORIGA:** cite Zhang et al. 2010. Access is on request from SERI, and the images are not redistributed here.
- **VEIRC:** research use only, with mandatory citation (see [3]). The images are not redistributed here.
- For the same reason, `reports/gradcam_examples.png` (Grad-CAM on ORIGA test crops) is generated locally by `evaluate.py` but git-ignored.
- **Pretrained weights:** timm (Apache-2.0), `efficientnet_b0_ra`.
- **This repo's code:** MIT (see `LICENSE`). The licence covers the code and the trained weights, not the datasets.

---

## [34] Reading the results (the part to rehearse)

All numbers are from `reports/metrics.json` (deployed disc model unless stated), and re-running `evaluate.py` reproduced them exactly.

| | Internal test (ORIGA) | External (VEIRC) |
|---|---|---|
| Glaucoma / normal | 25 / 73 | 32 / 225 |
| AUC [95% CI] | **0.832** [0.75–0.90] | **0.863** [0.81–0.91] |
| Sensitivity / specificity at the validation threshold (0.187) | 92% / 60% (TP 23, FN 2, TN 44, FP 29) | 100% / **25%** (TP 32, FN 0, TN 56, FP 169) |
| Sensitivity at 85% specificity | 0.44 | 0.62 |
| ECE before → after Platt | 0.250 → 0.064 | 0.492 → 0.196 |
| Median P(glaucoma): normal / glaucoma | 0.13 / 0.43 | 0.26 / 0.52 |

1. **Ranking transfers; the threshold does not.** External AUC (0.863) holds up. It is not below internal, and the gap is within noise. A likely reason is that VEIRC's "glaucoma" class requires CDR ≥ 0.7 while its normals are CDR ≤ 0.4, so borderline eyes sit in the excluded suspect group and the binary task is easier. But every external probability shifted **up**: the median for normals went from 0.13 to 0.26. So the ORIGA threshold flags 75% of Indian normals. This is the classic deployment lesson. Discrimination can generalise while **calibration and operating points are site-specific**. The fix is to re-fit Platt scaling and the threshold on a small labelled local sample before go-live, and to monitor the referral rate after.
2. **Calibration works in-distribution:** Platt scaling cut internal-test ECE from 0.25 to 0.06, and was fitted on validation only. It helps less out of distribution (0.49 → 0.20), which is consistent with point 1.
3. **High-sensitivity operating point:** at the validation threshold, internal sensitivity is 92% at 60% specificity. That is reasonable for "refer for a specialist look", not for diagnosis. Sensitivity at 85% specificity is only 0.44 internally, so the ROC curve is weak at the high-specificity end. If the interviewer asks "could this rule glaucoma in?", the answer is no.
4. **Suspects fall in between:** VEIRC suspect eyes (CDR 0.4–0.7) have median probability 0.35, between external normals (0.26) and glaucomas (0.52). The model's score tracks the cup-to-disc spectrum even though it never saw CDR labels.
5. **Train/serve parity check:** validation AUC computed inside the training loop from the 256-px cache (0.8183) equals validation AUC from `evaluate.py`, which runs raw images through the app's inference path (0.818). Preprocessing is identical in both paths [8].
6. **Against the literature:** Chen et al. (2015) reported AUC 0.831 on ORIGA, but trained on 99 images and tested on the remaining 551, a different protocol. Our 0.832 [0.75–0.90] is *in the same range*, not "better". Say it that way.
7. **Overfitting is visible:** training loss falls steadily while validation loss bottoms out around epochs 6–7 (`reports/learning_curves.png`). Early stopping picked epoch 6 for both views.
