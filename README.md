<div align="center">

# Fiber HGNN

</div>

<div align="center">
  <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a>
  <a href='https://github.com/chengwang96/Fiber_HGNN'><img src="https://img.shields.io/badge/GitHub-FiberHGNN-9E95B7%3Flogo%3Dgithub"></a> &nbsp; 
  <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-FiberHGNN-blue'></a> &nbsp; 
  <br>
</div>


## Introduction
This is an official PyTorch implementation of Heterogeneous Graph Neural Network for Fiber Tract Segmentation. We also provide implementations of others fiber segmentation methods (FiberGeoMap, TractCloud) and point cloud analysis methods (PointNet).


## 1. Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```
<details>
<summary> For Linux Kernel 6.0 or above (e.g. Ubuntu 24)
</summary>
 
Solution from [Sam Cheung](https://github.com/deemoe404).

Please run the following command before installing Chamfer Distance:
```
sudo apt install gcc-10 g++-10

su
cd /usr/local/src
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.tar.xz
tar -xf linux-5.4.tar.xz && cd linux-5.4
make headers_install INSTALL_HDR_PATH=/usr/local/linux-headers-5.4

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CFLAGS="-I/usr/local/linux-headers-5.4/include"
export CPPFLAGS="-I/usr/local/linux-headers-5.4/include"
```

In `extensions/chamfer_dist/setup.py`, in the `extra_compile_args` field, pass the correct header path to nvcc by adding the following line as the second element of `ext_modules`:
```
extra_compile_args={"nvcc": ['--system-include=/usr/local/linux-headers-5.4/include']}
```

</details>

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```


## 2. Datasets
We use HCP105 in this work. HCP105 is publicly available and can be accessed from [Zenodo](https://doi.org/10.5281/zenodo.1088277). After downloading the dataset, you can use our pre-processing tool to remove duplicate streamlines, downsample the point cloud using the FPS algorithm, and generate multi-category labels.

```bash
python trk_loader.py --clean_data_dir --check_duplicate_num --check_duplicate_name --update_trk_files --data_clean --data_preprocess --add_cr_label
```

You can adjust the input parameters to execute the specified preprocessing tasks.


## 3. Fiber HGNN training
You can quickly start experiments using bash scripts from the ./scripts folder, including the training of Fiber HGNN and comparative methods.

```bash
bash ./scripts/train_hgnn.sh # train Fiber HGNN
bash ./scripts/train_geomap.sh # train FiberGeoMap
bash ./scripts/train_tractcloud.sh # train TractCloud
bash ./scripts/train_pointnet.sh # train PointNet
```

The test.sh can be used for testing various models.

```bash
bash ./scripts/test.sh
```

## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)