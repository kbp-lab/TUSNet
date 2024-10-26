# TUSNet: A deep-learning model for one-shot transcranial ultrasound simulation and phase aberration correction

[Kasra Naftchi-Ardebili](https://bioengineering.stanford.edu/people/kasra-naftchi-ardebili)†, [Karanpartap Singh](https://karanps.com)†, Reza Pourabolghasem, Gerald R. Popelka, [Kim Butts Pauly](https://kbplab.stanford.edu).

† denotes equal contribution.

**Stanford University, Schools of Engineering and Medicine**

<hr>

![Figure introducing the architecture and training procedure for TUSNet. Displays encoder/multi-decoder setup and evaluation pipeline.](figures/intro.png)

We are excited to present TUSNet, the first end-to-end deep learning model capable of computing transcranial ultrasound pressure fields and phase aberration corrections with high accuracy and efficiency. Outperforming traditional methods like k-Wave by over 1200X, TUSNet achieves 98.3% accuracy in peak pressure estimation with a computation time of only 21 milliseconds, providing a fast and accurate solution for non-invasive ultrasound-based therapies.

## Abstract

Transcranial ultrasound (TUS) has emerged as a promising tool in clinical and research settings due to its potential to modulate neuronal activity, open the blood-brain barrier, facilitate targeted drug delivery via nanoparticles, and perform thermal ablation, all non-invasively. By delivering focused ultrasound waves to precise regions anywhere in the brain, TUS enables targeted energy deposition and is being explored in over fifty clinical trials as a treatment for conditions such as opioid addiction, Alzheimer’s disease, dementia, epilepsy, and glioblastoma. However, effective TUS treatment requires careful ultrasound parameter design and precise computation of the focal spot’s location and pressure, as skull heterogeneity increases the risk of off-target sonication or insufficient energy delivery to neural tissue. In clinical settings, this phase aberration correction must be computed within seconds. To achieve this, commercial devices often rely on faster methods, such as ray tracing, to predict the focus location and pressure. While computationally efficient, these methods may not always provide the high level of accuracy needed for optimal TUS delivery. We present TUSNet, the first end-to-end deep learning approach to solve for both the pressure field and phase aberration corrections without being bound to the inherent trade-off between accuracy and efficiency. TUSNet computes the 2D transcranial ultrasound pressure field and phase corrections within 21 milliseconds (over 1200X faster than k-Wave, a MATLAB-based acoustic simulation package), achieving 98.3% accuracy in estimating the peak pressure magnitude at the focal spot with a mean positioning error of only 0.18 mm compared to a ground truth from k-Wave.

## Getting Started

_Coming soon!_

## Citation

_Citation will be available on October 29th!_

