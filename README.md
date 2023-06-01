# Curvature-Invariant Method
## Requirements

* tqdm >= 4.52.0
* numpy >= 1.19.2
* scipy >= 1.6.3
* open3d >= 0.13.0
* torchvision >= 0.7.0
* scikit-learn >= 1.0

## Datasets and Models

#### Example Usage

##### Generate adversarial examples by attacking PointNet:

```
python main.py --attack_method curvature --surrogate_model pointnet_cls --target_model pointnet_cls
```
