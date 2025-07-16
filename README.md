
# Super-Resolution with GMFN and Swin-Enhanced Variants

This repository contains the implementation of three deep learning models for ×4 image super-resolution using the DIV2K dataset. The models are based on the Gated Multi-scale Feedback Network (GMFN), with architectural and loss function enhancements.

## 📁 Files

- `GMFN.ipynb`: Baseline implementation of the Gated Multi-scale Feedback Network.
- `GMFN_Swin.ipynb`: GMFN enhanced with Swin Transformer blocks for improved global context modeling.
- `GMFN_Swin_with_prec_loss.ipynb`: Swin-enhanced GMFN trained with perceptual loss (VGG19 feature space) in addition to L1.

## 📊 Dataset

All models are trained and evaluated on the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/), using ×4 bicubic downsampling for the LR inputs.

- Training set: 800 images
- Validation set: 100 images
- Downsampling: Bicubic, scale ×4

## ⚙️ Requirements

The notebooks were developed and tested using:

- Python 3.8+
- PyTorch or TensorFlow (depending on notebook)
- NumPy, OpenCV, Matplotlib
- tqdm, torchvision
- [For perceptual loss]: pretrained VGG19 (from torchvision.models or tf.keras.applications)

You may need to adapt paths and environment settings depending on your setup.

## 🧪 Results Summary

| Model                      | PSNR (Y) | SSIM (Y) | FID    |
|---------------------------|----------|----------|--------|
| GMFN                      | 27.57 dB | 0.7980   | 29.25  |
| GMFN_Swin                 | 27.63 dB | 0.8004   | 27.87  |
| GMFN_Swin + Perceptual    | 26.63 dB | 0.7627   | 31.56  |

*Note:* The perceptual-loss model shows sharper visual details despite slightly
