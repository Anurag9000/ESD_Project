# Training Readiness Summary

Total images: 56167

Final classes:
- organic: 9007 (16.04%)
- paper: 13444 (23.94%)
- other: 29901 (53.24%)
- metal: 3815 (6.79%)

Applied changes:
- moved all images from `plastic` into `other`
- moved metallic items from `other` into `metal` using filename prefixes `metal*` and `alum*`
- moved additional model-predicted metallic items from `other` into `metal`
- regenerated every file with normalized filenames
- corrected extensions based on actual image content
- stripped EXIF and GPS metadata
