# Layer-wise Update Frequency for Continual Diffusion Under Nested Learning Paradigm

**👥 Authors**
* Amit Sason
* Eldar Mamedov
* Or Shraga
* E-mails: {amitsaso, eldarmam, shragao}@post.bgu.ac.il 

<br>

**📝 About The Project**
This project addresses the challenge of "Catastrophic Forgetting" in generative diffusion models when trained on sequential tasks. Our objective is to demonstrate a frequency-based continual learning strategy in Denoising Diffusion Probabilistic Models (DDPMs) that outperforms existing regularization methods (like EWC) in retaining prior knowledge while successfully acquiring new tasks.

Taking inspiration from the human brain's multi-time-scale update mechanisms (Nested Learning - NL), we propose a training strategy where different layers of a U-Net architecture are updated at varying frequencies. Specifically, shallow layers are updated frequently to capture fine-grained visual details, while deeper layers integrate information over longer, slower cycles to preserve abstract semantic representations.

We evaluated our approach using a class-conditional DDPM on a partitioned MNIST dataset:
* **Task A:** Digits 0-5
* **Task B:** Digits 6-9

<br>

**📊 Key Results**
Our empirical results show that a frequency schedule of `[1, 4, 16]` (Shallow, Middle, Deep) provided the optimal performance, significantly reducing forgetting and maintaining higher image quality for previous tasks compared to standard baselines.

| Method | Task A Accuracy (Retention) | Overall FID Score |
| :--- | :---: | :---: |
| Baseline (Sequential) | 53.6% | - |
| EWC Regularization | 71.6% | - |
| **Ours (Frequency 1_4_16)** | **93.0%** | **20.794** |

*Note: The frequency-based model achieved significant retention of Task A and reduced visual artifacts, with a minor trade-off in the final digits of Task B.*

<br>

**📂 Repository Structure**
* `train_A_than_B_base.py` - Script for training the baseline sequential diffusion model.
* `train_A_than_B_freq_1_4_16.py` - Script for training the model using our proposed Nested-Learning-inspired frequency updates (1_4_16).
* `train_A_than_B_EWC.py` - Script for training the model using Elastic Weight Consolidation (EWC).
* `eval.py` / `inference_evaluatin.ipynb` - Scripts and notebooks for evaluating FID scores and accuracy.
* `predict.py` - Inference script to generate new samples.
* `run_diffusion_modules.ipynb` - Interactive notebook for running the diffusion modules.

<br>

**🚀 How to Run**

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/orshraga/Continual-Diffusion-NL.git
cd Continual-Diffusion-NL
```

**2. Train the baseline model:**
```bash
python train_A_than_B_base.py
```

**3. Train the model using the frequency-based approach:**
```bash
python train_A_than_B_freq_1_4_16.py
```

**4. Evaluate the models:**
Use the `inference_evaluatin.ipynb` notebook to generate images and calculate FID and Accuracy scores.

<br>

**⚙️ How to Change Layer Update Frequencies**
The core of this project lies in altering the update frequencies of different parts of the U-Net architecture. 

The architecture is partitioned into three logical depth groups: `Shallow`, `Middle`, and `Deep`. You can change the update schedule `S = [f_shallow, f_middle, f_deep]` in the training scripts (e.g., `train_A_than_B_freq_1_4_16.py`).

For example, to set the optimal frequency `[1, 4, 16]`:
* **Shallow level:** Updated at every optimization step (`k=1`).
* **Middle level:** Updated every 4 batches (`k=4`).
* **Deep level:** Updated every 16 batches (`k=16`).

Look for the `update_frequency` parameter or the optimizer step conditions in the training loop to experiment with other schedules like `[1, 2, 8]`, `[1, 4, 8]`, or `[2, 16, 32]`.

<br>

**📚 Relevant Papers & Acknowledgments**
1. Behrouz, A., et al. (2025). *Nested Learning: The Illusion of Deep Learning Architectures*. Google Research.
2. Ho, J., et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
3. Wang, Z., et al. (2025). *Avoid Catastrophic Forgetting with Rank-1 Fisher from Diffusion Models*. Georgia Institute of Technology.

*This project was developed as part of "Generative Models" academic coursework and research at Ben-Gurion University.*
