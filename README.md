# 🦁 Zoo3D: Zero-Shot 3D Object Detection at Scene Level 🐼

## 🐾 News
🦁 We released Zoo3D₀!

⏳🐢 Code will be released soon! Stay tuned

This repository contains an implementation of Zoo3D, a zero-shot indoor 3D object detection method introduced in our paper:

> **Zoo3D: Zero-Shot 3D Object Detection at Scene Level**<br>
> [Andrey Lemeshko](https://github.com/Andre7416),
> [Bulat Gabdullin](https://github.com/bulatko),
> [Nikita Drozdov](https://github.com/anac0der),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ),
> [Danila Rukhovich](https://github.com/filaPro),
> [Maksim Kolodiazhnyi](https://github.com/col14m)
> <br>
> https://arxiv.org/abs/2511.20253
## Data Preparation

Follow steps from [`data/scannet`](data/scannet) to prepare data.

## Running Zoo3D₀
1. Follow the **Further installation** steps from [MaskClustering](https://github.com/PKU-EPIC/MaskClustering) to install all required dependencies
2. Move our `mask_predict.py` script into cropformer
```bash
cp mask_predict.py Zoo3D_0/third_party/detectron2/projects/CropFormer/demo_cropformer
```
3. Clone [sam2](https://github.com/facebookresearch/sam2) repository into `third_party` folder and install it
```bash
cd Zoo3D_0/third_party
git clone git@github.com:facebookresearch/sam2.git
cd sam2
pip install -e .
```
4. Locate the corresponding configuration file in the `configs` folder
5. Update the following variables:
   - `cropformer_path` in the configuration file
   - `CUDA_LIST` in `run.py`
6. Run the following command:

```bash
cd Zoo3D_0
python run.py --config config_name
```
Result will be stored at `data/prediction/config_name`

## Metrics

| Dataset | Task | mAP@25 | mAP@50 | 
|---------|------|--------|--------|
| Scannet200 | [point clouds](Zoo3D_0/configs/scannet200.json) | 23.6 | 16.0 |
| Scannet200 | [posed images](Zoo3D_0/configs/scannet200_posed.json) | 15.7 | 7.5 |
| Scannet200 | [unposed images](Zoo3D_0/configs/scannet200_unposed.json) | 8.7 | 3.1 |


## Predictions Example

<p align="center">
  <img src="https://github.com/user-attachments/assets/7552e073-a52b-40de-ab5c-f00ba89fa9f8" alt="Zoo3D₀ predictions"/>
</p>

## Citation 
If you find this work useful for your research, please cite our paper:
```
@article{lemeshko2025zoo3d,
  title={Zoo3D: Zero-Shot 3D Object Detection at Scene Level},
  author={Lemeshko, Andrey and Gabdullin, Bulat and Drozdov, Nikita and Konushin, Anton and Rukhovich Danila and Kolodiazhnyi, Maksim},
  journal={arXiv preprint arXiv:2511.20253},
  year={2025}
}
```