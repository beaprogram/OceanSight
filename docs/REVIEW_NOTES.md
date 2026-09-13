# Visual review of the completed run

These are observations from the generated contact sheets, not causal diagnoses.

## Unsupervised audit

The highest-ranked outliers include several frames from `vid_000250`, showing a log-like object and a large dark background, and frames with prominent ROV equipment. That is consistent with this audit detecting unusual **appearance**, not automatically discovering incorrect labels. Multiple frames from one video can dominate a review queue; a useful extension is to deduplicate the review queue by video before sending it to an annotator.

## Detection failures

- `vid_000203_frame0000009.jpg`: predicted boxes cover smaller parts of a larger labeled debris region. The fixed-IoU matching counts two false positives and two missed boxes. Object granularity and localization deserve review.
- `vid_000205_frame0000041.jpg`: the model highlights bright small regions while missing the large, low-contrast labeled region. This image has three false positives and one miss in the saved report.
- `vid_000320_frame0000056.jpg`: no predictions at confidence .25, despite two labeled boxes in a cluttered scene.
- `vid_000431_frame0000032.jpg`: two predictions in an image with no target debris labels. A manual source-annotation review would be needed to distinguish false alarms from missing target labels.

Do not relabel these test images and report a new score as though it were untouched evaluation. Use training/validation images for model changes and reserve a fresh final test for the next phase.

## Failed optimization

The initial broad static INT8 QDQ conversion ran successfully but achieved zero test AP. Its raw outputs also differ substantially from the FP32 baseline. This is a useful example of why conversion success and file compression are insufficient acceptance criteria. FP32 ONNX is the working deployment path. The exact cause of the INT8 failure has not been established.

## Verification correction

An initial backend evaluation used the validator's default rectangular preprocessing for PyTorch and fixed square preprocessing for ONNX. Before reporting final results, the script was corrected to `rect=False` for every test backend and rerun. The committed final metrics use identical square inputs and show matching FP32 PyTorch/ONNX AP. This correction changes evaluation methodology, not model selection or hyperparameters.
