# Layer-wise Update Frequency for Continual Diffusion Under Nested Learning Paradigm

## 👥 Authors
* Amit Sason
* Eldar Mamedov
* Or Shraga
*E-mails: {amitsaso,eldarmam,shragao}@post.bgu.ac.il 


## 📝 About The Project
[cite_start]This project addresses the challenge of "Catastrophic Forgetting" in generative diffusion models when trained on sequential tasks[cite: 2163]. [cite_start]Our objective is to demonstrate a frequency-based continual learning strategy in Denoising Diffusion Probabilistic Models (DDPMs) that outperforms existing regularization methods (like EWC) in retaining prior knowledge while successfully acquiring new tasks[cite: 2164].

[cite_start]Taking inspiration from the human brain's multi-time-scale update mechanisms (Nested Learning - NL), we propose a training strategy where different layers of a U-Net architecture are updated at varying frequencies[cite: 2167]. [cite_start]Specifically, shallow layers are updated frequently to capture fine-grained visual details, while deeper layers integrate information over longer, slower cycles to preserve abstract semantic representations[cite: 2168, 2231, 2232].

[cite_start]We evaluated our approach using a class-conditional DDPM on a partitioned MNIST dataset[cite: 2169]:
* [cite_start]**Task A:** Digits 0-5 [cite: 2213]
* [cite_start]**Task B:** Digits 6-9 [cite: 2214]

[cite_start]Our empirical results show that a frequency schedule of `[1, 4, 16]` (Shallow, Middle, Deep) provided the optimal performance, significantly reducing forgetting and maintaining higher image quality for previous tasks[cite: 2287, 2172].

## 📂 Repository Structure
* [cite_start]`train_A_than_B_base.py` - Script for training the baseline sequential diffusion model.
* [cite_start]`train_A_than_B_freq_1_4_16.py` - Script for training the model using our proposed Nested-Learning-inspired frequency updates (1_4_16).
* [cite_start]`eval.py` / `inference_evaluatin.ipynb` - Scripts and notebooks for evaluating FID scores and accuracy[cite: 2531, 2549].
* [cite_start]`predict.py` - Inference script to generate new samples.
* [cite_start]`run_diffusion_modules.ipynb` - Interactive notebook for running the diffusion modules[cite: 2831].


## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
Train the baseline model:Bashpython train_A_than_B_base.py
Train the model using the frequency-based approach:Bashpython train_A_than_B_freq_1_4_16.py
Evaluate the models:Use the inference_evaluatin.ipynb notebook to generate images and calculate FID and Accuracy scores.⚙️ How to Change Layer Update FrequenciesThe core of this project lies in altering the update frequencies of different parts of the U-Net architecture.The architecture is partitioned into three logical depth groups: Shallow, Middle, and Deep.
You can change the update schedule S = [f_shallow, f_middle, f_deep] in the training scripts (e.g., train_A_than_B_freq_1_4_16.py).For example, to set the optimal frequency [1, 4, 16]:Shallow level: Updated at every optimization step (k=1).Middle level: Updated every 4 batches (k=4).Deep level: Updated every 16 batches (k=16).Look for the update_frequency parameter or the optimizer step conditions in the training loop to experiment with other schedules like [1, 2, 8], [1, 4, 8], or [2, 16, 32].

📚 Relevant Papers
Behrouz, A., et al. (2025). Nested Learning: The Illusion of Deep Learning Architectures. Google Research.

Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.

Wang, Z., et al. (2025). Avoid Catastrophic Forgetting with Rank-1 Fisher from Diffusion Models. Georgia Institute of Technology.














## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name