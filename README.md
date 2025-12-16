This repository provides the **official PyTorch implementation** of our paper:

> **ZeroPur: Succinct Training-Free Adversarial Purification**

**ZeroPur** is a highly efficient *training-free* adversarial purification defense.  
Unlike existing approaches, it **does not rely on external generative models or auxiliary networks**, significantly reducing computational overhead while maintaining strong defense performance.

---

## Method Overview

We are motivated by the observation that **adversarial examples tend to lie outside the natural image manifold**, and that purification can be viewed as projecting them back onto this manifold.

ZeroPur achieves this goal through a **simple yet effective two-stage process**:

### 1. Guided Shift
A shifted embedding of the adversarial example is obtained under the guidance of its **blurred counterpart**, capturing coarse manifold directions.

- Implemented as `coarse_shifting` in `zeropur.py`

### 2. Adaptive Projection
A directional vector constructed from the shifted embedding provides momentum to iteratively project the image back onto the natural manifold, while enforcing a **perceptual similarity constraint (LPIPS)**.

- Implemented as `fine_alignment` in `zeropur.py`

---

## Datasets

ZeroPur is evaluated on the following **official benchmark datasets**:

- CIFAR-10  
- CIFAR-100  
- ImageNet-1K  

---


## Checkpoints
We offer checkpoints for testing in the ```zero_pur/checkpoint``` folder.


## Evaluation
### Test ZeroPur against AutoAttack attacks on CIFAR-10
cd zero_pur/ \
python eval.py 


```
## Dependencies
python 3.9.22, PyTorch = 1.13.1, cudatoolkit = 11.8.0, torchvision = 0.14.1, tqdm, scikit-learn, numpy, opencv-python
```
