Roboflow yawn external comparison
Images: 393; saved MediaPipe mouth ROI: 344; ROI failures: 49
All face-mode models use the full original face, with no MediaPipe restriction.
All mouth-mode models use EXACTLY the same cached mouth ROI from 12_external_eval.
Team ResNet18 mouth results reuse original prediction CSV, not a new inference.
All thresholds are operating points on the already-used development set.
Primary architecture comparison is threshold=0.50 within the same input mode;
face-vs-mouth comparisons measure input pipeline changes, NOT only architecture.
All unknowns count as incorrect. Timings exclude I/O and MediaPipe ROI extraction.
Do not claim these results are performance on an untouched independent final test set.
