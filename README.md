# Curvature-Invariant Method
## Requirements

* tqdm >= 4.52.0
* numpy >= 1.19.2
* scipy >= 1.6.3
* open3d >= 0.13.0
* torchvision >= 0.7.0
* scikit-learn >= 1.0

## Datasets and Models

You can download the dataset from [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).

Please download the [pretrained models](https://drive.google.com/file/d/1L25i0l6L_b1Vw504WQR8-Z0oh2FJA0G9/view?usp=sharing) and put them under "./checkpoint"


#### Example Usage

##### Generate adversarial examples by attacking PointNet:

```
python main.py --attack_method curvature --surrogate_model pointnet_cls --target_model pointnet_cls
```
