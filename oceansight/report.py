"""Render a readable results summary from measured JSON, never placeholder values."""
import json
from .common import ROOT, REPORTS

def run():
    def read(name):
        p=REPORTS/name
        return json.loads(p.read_text()) if p.exists() else None
    dataset=read('dataset.json');comparison=read('comparison.json');selection=read('selection.json')
    metrics=read('test_metrics.json');benchmark=read('benchmark.json');quant=read('quantization.json');failures=read('failure_cases.json')
    lines=['# Measured results','', 'Generated from local report files. Metrics are proportions, not percentages. Missing steps are explicitly marked.','']
    if dataset:
        lines+=['## Dataset','', '| Split | Images | Videos | Debris boxes | Negative images |','|---|---:|---:|---:|---:|']
        for s,r in dataset['counts'].items():lines.append(f"| {s} | {r['images']} | {r['videos']} | {r['boxes']} | {r['negative_images']} |")
        lines+=['','Custom video-disjoint subset; not the official TrashCan benchmark.','']
    lines+=['## Validation model comparison','','| Model | Epochs | mAP50 | mAP50–95 | Checkpoint MB |','|---|---:|---:|---:|---:|']
    for r in comparison or []:
        v=r['validation'];lines.append(f"| {r['model']} | {r['epochs']} | {v['metrics/mAP50(B)']:.4f} | {v['metrics/mAP50-95(B)']:.4f} | {r['size_mb']:.2f} |")
    lines+=['',f"Selected: **{selection['selected']}**, using validation mAP50–95." if selection else 'Selection: not run.','']
    lines+=['## Held-out test','','| Backend | mAP50 | mAP50–95 | File MB |','|---|---:|---:|---:|']
    for r in metrics or []:
        v=r['test'];lines.append(f"| {r['backend']} | {v['metrics/mAP50(B)']:.4f} | {v['metrics/mAP50-95(B)']:.4f} | {r['size_mb']:.2f} |")
    if not metrics:lines+=['','Test evaluation: not run.']
    lines+=['','## CPU forward benchmark','']
    if benchmark:
        lines += ['| Backend | Median ms | P95 ms | Repeated samples |','|---|---:|---:|---:|']
        for r in benchmark['results']:lines.append(f"| {r['backend']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['samples']} |")
        lines+=['',benchmark['scope'],f"Batch 1, {benchmark['threads']} CPU threads, {benchmark['imgsz']}×{benchmark['imgsz']}; 5 warmup calls, 30 images × 3 repetitions. Sequential backend timing on an M4 MacBook Air; no confidence intervals or power measurements.",'',
                'Checkpoint and ONNX file sizes use different serialization/precision conventions. Compare INT8 size with FP32 ONNX for the quantization compression ratio; the PyTorch checkpoint is not an FP32 tensor-storage baseline.','']
    else:lines+=['Not run.','']
    lines+=['## Quantization','',f"Status: {quant['status']}" if quant else 'Not run.','']
    if failures:
        t=failures['totals'];lines+=['## Demo-pipeline error analysis','',f"At confidence 0.25 and IoU 0.5: **{t['tp']} true positives, {t['fp']} false positives, {t['fn']} missed boxes**. Precision {t['precision']:.3f}; recall {t['recall']:.3f}.",'',
          'These fixed-threshold counts come from the custom ONNX demo preprocessing/NMS. They differ in definition from AP and from the validator’s best-F1 operating-point precision/recall. See the locally generated failure contact sheet for the highest-error examples.','']
    lines+=['## Interpretation and next work','',
       'This is a short, single-seed subset experiment. It demonstrates the full workflow but does not establish reliable open-water performance. The comparison is between two related YOLO architectures. FP32 ONNX remains the app default; quantization is an experimental result, not an assumed improvement.','',
       '**Not run:** full original dataset verification/training, RT-DETR/Faster R-CNN, multi-seed uncertainty, semantic embedding analysis, physical edge-device profiling, calibrated confidence, tracking, material classification and production deployment.','']
    (ROOT/'docs'/'RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__':run()
