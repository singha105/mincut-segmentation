# Learned regional term vs scribble histograms

Oxford-IIIT Pet official test split, 3,662 images on a 128 px grid. Mean per-image
foreground IoU and pixel accuracy, with the dataset's border band excluded.

| Method | N-link weight (tuned on validation) | Foreground IoU | Pixel accuracy |
| --- | ---: | ---: | ---: |
| Graph cut, scribble histograms (existing) | 200 | 0.644 | 0.848 |
| U-Net alone, ignores scribbles | n/a | 0.895 | 0.963 |
| Graph cut, U-Net regional term | 20 | 0.902 | 0.966 |

The U-Net has 1,942,577 parameters and was trained on 3,312 images
from the trainval split (best validation epoch: 23). Scribbles are simulated from the
ground truth with a per-image seed, so every method sees the same ones. They are
strokes through three random foreground points and four random background points.
