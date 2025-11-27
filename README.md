<div align="center">
<h1>Metropolis-Hastings Sampling for 3D Gaussian Reconstruction</h1>
<div align="center">

<div align="center">
<img src="assets/main.gif" width="600">
</div>

### NeurIPS 2025

<a href='https://hjhyunjinkim.github.io/'><strong>Hyunjin Kim</strong></a><sup>1</sup> · 
<a href='https://www.haebeom.com'><strong>Haebeom Jung</strong></a><sup>2</sup> · 
<a href='https://jaesik.info/'><strong>Jaesik Park</strong></a><sup>2</sup>

<sup>1</sup>UC San Diego · <sup>2</sup>Seoul National University

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2506.12945"><img src="https://img.shields.io/badge/arxiv-2506.12945-b31b1b"></a>
  <a href="https://hjhyunjinkim.github.io/MH-3DGS/"><img src="https://img.shields.io/badge/Project%20Page-MH3DGS-blue"></a> 
</p>
</div>


## News
- **[11/26/2025]** Code is now available!
- **[09/25/2025]** MH-3DGS is accepted to <b>NeurIPS 2025</b> 🎉


## Setup

```shell
git clone https://github.com/hjhyunjinkim/MH-3DGS.git --recursive
cd MH-3DGS
conda config --set channel_priority flexible
conda env create --file environment.yml
conda activate mh_3dgs
```

## Datasets
We use the same datasets used by [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), the datasets can be found at:

- [Mip-NeRF360](https://jonbarron.info/mipnerf360/)
- [Tanks&Temples](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)


## Running

To train, use the following command. We withhold a test set for evaluation by using the ```--eval``` flag.

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval
```

For evaluation, use the following commands.
```shell
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

You may also use the ```full_eval.py``` script for full evaluation of the datasets:
```shell
python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder> --output_path <output folder>
```

## Acknowledgements
Our work is based on the open-sourced official implementations of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc).


## Citation
If you find our work useful, please consider citing:
```
@inproceedings{
  kim2025metropolishastings,
  title={Metropolis-Hastings Sampling for 3D Gaussian Reconstruction},
  author={Hyunjin Kim and Haebeom Jung and Jaesik Park},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
