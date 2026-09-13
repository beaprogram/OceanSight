# Data and model card

**Intended use:** local educational and portfolio demonstration of candidate marine-debris detection for human review.

**Not intended:** robotic control, navigation, autonomous object removal, exhaustive pollution measurement, or counts of unique tracked debris.

**Data lineage:** JAMSTEC J-EDI → TrashCan researchers → third-party Hugging Face CSV/image mirror → this project's video-disjoint capped subset. Source revision and metadata/image hashes are in the committed reports. Original segmentation masks are not used. The third-party conversion remains unverified against the original archive.

**Labels:** every source `trash_*` category becomes binary `marine_debris`; all other source objects are ignored for detection labels. Source annotations are retained in the manifest. Boxes are clipped to image dimensions; degenerate/nonfinite boxes fail preparation.

**Selection:** two compact pretrained YOLO detectors fine-tuned with matching settings. Model selection uses validation mAP50–95. The held-out test set is used for final backend comparison and post-hoc failure inspection. There is no multi-seed confidence interval or official-benchmark claim.

**Quality review:** standardized handcrafted color/quality features, KMeans, Isolation Forest, and PCA. These are exploratory training-only appearance analyses. Outliers are not automatically removed and no semantic cluster labels are inferred.

**Deployment:** fixed square ONNX input, CPUExecutionProvider, batch one. The app uses explicit preprocessing/NMS. FP32 is the deployment default. Experimental INT8 status is recorded even if conversion fails. Calibration uses 64 training images; no validation/test calibration.

**Potential bias/domain shift:** mostly deep underwater imagery and a small subset of source videos; unknown location overlap; limited shallow-water/harbor coverage; missing completely unannotated negative frames; unknown mirror conversion errors. No demographic/person-related model use is intended.

**Known practical limits:** low resolution can obscure small debris; model confidence is uncalibrated; counts depend on threshold; no tracking, segmentation, material classes, depth or object size estimates. Actual failures are saved in reports rather than replaced by hypothetical examples.

**Reproducibility:** fixed seed, pinned mirror, package version lock, source manifest, model hashes and training logs. Hardware/software variation and GPU operations may prevent bit-for-bit retraining equality.

**Licensing:** project source uses AGPL-3.0. Dataset/annotation/weight rights are separate. Verify original owner terms before redistribution or commercial use. No dataset images are included in Git.
