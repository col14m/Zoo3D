# Preparing ScanNet Data for Zoo3D Model
## 1. Ground-Truth Point Clouds

To prepare ScanNet data, we follow the procedure described in [OneFormer3D](https://github.com/filaPro/oneformer3d/tree/main/data/scannet).

### Step 1. Get Preprocessed ScanNet Detection Data

* Follow the [official procedure](https://github.com/filaPro/oneformer3d/tree/main/data/scannet),
  **or** simply download the preprocessed version of ScanNet from [this link](https://huggingface.co/datasets/maksimko123/UniDet3D/blob/main/scannet.tar.gz).
* Unzip it in the **current directory**.
* Preprocess 2D images following the [ImVoxelNet procedure](https://github.com/filaPro/imvoxelnet/tree/master/data/scannet) to extract fixed number of frames per scene (we use 300) or simply download the preprocessed version of ScanNet from [this link](https://huggingface.co/datasets/maksimko123/scannet/tree/main).
* Follow preprocessing from [OneFormer3D](https://github.com/filaPro/oneformer3d/tree/main/data/scannet) to get ground truth pkl files **or** simply download it from [this link](https://huggingface.co/datasets/Andre7416/rec_scannet/tree/main).

---

## 2. Posed and Unposed Point Clouds

To generate posed and unposed point clouds, we use preprocessed 2D images from previous step.

### Step 1. Reconstruction

* Clone [reconstruction folder](https://github.com/col14m/TUN3D/tree/main/reconstruction) from **TUN3D** to this folder.
* Replace `reconstruct.py` file in cloned folder to our modified [reconstruct script](./reconstruct.py)
* Follow the instructions provided in [TUN3D](https://github.com/col14m/TUN3D/tree/main/reconstruction) **or** simply download the preprocessed version ([posed](https://huggingface.co/datasets/Andre7416/rec_scannet/blob/main/rec_posed_images.zip), [unposed](https://huggingface.co/datasets/Andre7416/rec_scannet/blob/main/rec_unposed_images.zip)).
---
After preprocessing datasets, the following data structure should be obtained:
```
../data/
    scannet/
        posed_images/
            xxxxx.jpg
            xxxxx.png
            xxxxx.txt
            intrinsic.txt
        rec_posed_images/
            xxxxx.jpg
            xxxxx.png
            xxxxx.txt
            intrinsic.txt
        rec_unposed_images/
            xxxxx.jpg
            xxxxx.png
            xxxxx.txt
            intrinsic.txt
        points/
            scene0011_00.bin
            ...
        points_posed/
            scene0011_00.bin
                ...
        points_unposed/
            scene0011_00.bin
            ...
        ...
```
