import os, urllib.request, tarfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from scipy import linalg

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

# Task split (כמו שהגדרת עכשיו)
A_DIGITS = {0, 1, 2, 3, 4, 5}
B_DIGITS = {6, 7, 8, 9}

# -----------------------------
# UTILS
# -----------------------------
def to_group(d):
    return 0 if int(d) in A_DIGITS else 1

# -----------------------------
# MNIST JUDGE (xuhdev/max-pytorch-mnist)
# -----------------------------
class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 6, 5)
        self.fc1   = nn.Linear(96, 32)
        self.fc2   = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 96)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x
    
    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 96)
        x = F.relu(self.fc1(x)) 
        return x


def load_mnist_judge(
    device=None,
    model_dir="./pretrained_mnist",
    model_url="https://github.com/xuhdev/max-pytorch-mnist/raw/master/pretrained-model/mnist-classifier.tar.gz",
    pt_name="mnist-classifier.pt",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tar_path = model_dir / "mnist-classifier.tar.gz"
    pt_path  = model_dir / pt_name

    if not tar_path.exists() and not pt_path.exists():
        print(f"Downloading MNIST judge weights from: {model_url}")
        urllib.request.urlretrieve(model_url, tar_path)
        print(f"Saved: {tar_path}")

    if not pt_path.exists():
        print(f"Extracting: {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(model_dir)
        if not pt_path.exists():
            names = tarfile.open(tar_path, "r:gz").getnames()
            raise FileNotFoundError(f"Did not find {pt_path}. Tar contents: {names}")

    model = MyConvNet().to(device)
    state_dict = torch.load(pt_path, map_location=device, weights_only=True)

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device

# -----------------------------
# LOAD GENERATED DATA
# -----------------------------
def load_generated_samples(samples_dir):
    """
    Loads samples from multiple .npz files (one per digit) in the directory.
    Expected format: digit_{label}_{count}.npz (e.g., digit_9_1000.npz)
    """
    samples_dir = Path(samples_dir)
    all_images = []
    all_labels = []
    
    print(f"Loading .npz files from {samples_dir}...")

    # Iterate over all 10 digits to find their corresponding files
    for digit in range(10):
        # We look for the specific file pattern generated previously
        # Assuming filename is: digit_{digit}_1000.npz
        filename = f"digit_{digit}_1000.npz"
        file_path = samples_dir / filename

        if file_path.exists():
            # 1. Load data
            data = np.load(file_path)
            imgs_np = data['samples']  # Shape (N, 1, 28, 28)
            
            # 2. Convert to Tensor
            imgs_tensor = torch.from_numpy(imgs_np).float()
            
            # 3. Create Labels (N,) filled with the current digit
            lbls_tensor = torch.full((imgs_tensor.shape[0],), digit, dtype=torch.long)

            all_images.append(imgs_tensor)
            all_labels.append(lbls_tensor)
        else:
            print(f"Warning: File {filename} not found. Skipping digit {digit}.")

    if not all_images:
        raise FileNotFoundError(f"No valid 'digit_X_1000.npz' files found in {samples_dir}")

    # 4. Concatenate all batches into single tensors
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # 5. Ensure images are in [0, 1] range (required for eval.py logic)
    # If your data is already [0,1], this does nothing. If it's unbounded, it clips it.
    images = torch.clamp(images, 0.0, 1.0)

    print(f"Successfully loaded {len(labels)} images total.")
    return images, labels

# -----------------------------
# LOAD REAL MNIST (TEST SPLIT)
# -----------------------------
def load_real_mnist(split="AB"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))   # -> [-1,1]
    ])

    mnist = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=tf
    )

    if split == "AB":
        indices = list(range(len(mnist)))
    elif split == "A":
        indices = [i for i, (_, y) in enumerate(mnist) if y in A_DIGITS]
    elif split == "B":
        indices = [i for i, (_, y) in enumerate(mnist) if y in B_DIGITS]
    else:
        raise ValueError("split must be A / B / AB")

    return DataLoader(
        Subset(mnist, indices),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

# -----------------------------
# CONFUSION + ACCURACY
# -----------------------------
@torch.no_grad()
def compute_confusion(classifier, images_01, true_labels):
    """
    images_01: [N,1,28,28] in [0,1]
    classifier expects [-1,1]
    """
    images_01 = images_01.to(DEVICE)
    true_np = true_labels.cpu().numpy()

    x = (images_01 - 0.5) / 0.5  # -> [-1,1]

    logits = classifier(x)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

    cm10 = confusion_matrix(true_np, preds, labels=list(range(10)))
    accuracy = float(np.mean(preds == true_np))

    true_grp = np.array([to_group(x) for x in true_np])
    pred_grp = np.array([to_group(x) for x in preds])
    cm2 = confusion_matrix(true_grp, pred_grp, labels=[0, 1])

    forgetting = float(cm2[0, 1] / (cm2[0].sum() + 1e-8))  # A -> B rate

    return {"accuracy": accuracy, "cm10": cm10, "cm2": cm2, "forgetting": forgetting}

# -----------------------------
# FEATURE EXTRACTION (FID)
# -----------------------------
@torch.no_grad()
def extract_features_real(classifier, dataloader_normed):
    """
    dataloader yields x in [-1,1] already
    Returns [N, 32] features (penultimate layer)
    """
    feats = []
    for x, _ in dataloader_normed:
        x = x.to(DEVICE)
        features = classifier.forward_features(x) 
        feats.append(features.cpu().numpy())
    return np.concatenate(feats, axis=0)

@torch.no_grad()
def extract_features_gen(classifier, images_01):
    """
    Returns [N, 32] features (penultimate layer)
    """
    feats = []
    for i in range(0, len(images_01), BATCH_SIZE):
        batch = images_01[i:i+BATCH_SIZE].to(DEVICE)
        batch = (batch - 0.5) / 0.5
        features = classifier.forward_features(batch)
        feats.append(features.cpu().numpy())
    return np.concatenate(feats, axis=0)

def compute_fid(real_feats, gen_feats):
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_g = np.cov(gen_feats, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))

# -----------------------------
# MAIN EVAL (A / B / AB)
# -----------------------------
def evaluate(samples_dir_current, samples_dir_prev_A):
    classifier, _ = load_mnist_judge(device=DEVICE, model_dir="./pretrained_mnist")

    gen_images, gen_labels = load_generated_samples(samples_dir_current)
    gen_labels_np = gen_labels.cpu().numpy()

    mask_A = np.isin(gen_labels_np, list(A_DIGITS))
    mask_B = np.isin(gen_labels_np, list(B_DIGITS))

    genA_images = gen_images[mask_A]
    genA_labels = gen_labels[mask_A]
    genB_images = gen_images[mask_B]
    genB_labels = gen_labels[mask_B]

    # Confusions
    conf_AB = compute_confusion(classifier, gen_images, gen_labels)
    conf_A  = compute_confusion(classifier, genA_images, genA_labels) if len(genA_images) else None
    conf_B  = compute_confusion(classifier, genB_images, genB_labels) if len(genB_images) else None

    # Real loaders/features
    realA = load_real_mnist("A")
    realB = load_real_mnist("B")
    realAB = load_real_mnist("AB")

    real_feats_A  = extract_features_real(classifier, realA)
    real_feats_B  = extract_features_real(classifier, realB)
    #real_feats_AB = extract_features_real(classifier, realAB) # אופציונלי

    # Gen features
    
    gen_feats_A  = extract_features_gen(classifier, genA_images) if len(genA_images) else None
    gen_feats_B  = extract_features_gen(classifier, genB_images) if len(genB_images) else None
    #gen_feats_AB = extract_features_gen(classifier, gen_images) # אופציונלי

    fid_A_current = compute_fid(real_feats_A,  gen_feats_A)  if gen_feats_A is not None else None
    fid_B_current = compute_fid(real_feats_B,  gen_feats_B)  if gen_feats_B is not None else None
    
    ## Global FID (נשאיר רק להשוואה, אבל זה לא המדד הקובע)
    #fid_global_AB = compute_fid(real_feats_AB, gen_feats_AB)

    # ---------------------------------------------------------
    # התיקון הגדול: חישוב AFID (Average FID)
    # ---------------------------------------------------------
    afid_current = None
    
    # אם יש לנו תוצאות גם ל-A וגם ל-B, הממוצע הוא (A+B)/2
    if fid_A_current is not None and fid_B_current is not None:
        afid_current = (fid_A_current + fid_B_current) / 2.0
    
    # מקרי קצה: אם יש רק משימה אחת (למשל רק התחלנו לאמן)
    elif fid_A_current is not None:
        afid_current = fid_A_current
    elif fid_B_current is not None:
        afid_current = fid_B_current
    # ---------------------------------------------------------

    # חישוב Forgetting (מול מודל ישן)
    print(f"Loading previous task samples from: {samples_dir_prev_A}")
    gen_images_prev, gen_labels_prev = load_generated_samples(samples_dir_prev_A)
    gen_labels_prev_np = gen_labels_prev.cpu().numpy()
    
    mask_A_prev = np.isin(gen_labels_prev_np, list(A_DIGITS))
    
    genA_images_prev = gen_images_prev[mask_A_prev]
    genA_labels_prev = gen_labels_prev[mask_A_prev]
    fid_A_prev = None
    fid_forgetting_A = None
    acc_A_prev = None
    acc_forgetting_A = None

    if len(genA_images_prev) > 0:
        
        conf_A_prev = compute_confusion(classifier, genA_images_prev, genA_labels_prev)
        acc_A_prev = conf_A_prev["accuracy"]

        # 4. חישוב FID ישן (משווים את A הישן מול A האמיתי שכבר טענו למעלה)
        gen_feats_A_prev = extract_features_gen(classifier, genA_images_prev)
        fid_A_prev = compute_fid(real_feats_A, gen_feats_A_prev)
        if fid_A_current is not None:
            fid_forgetting_A = fid_A_current - fid_A_prev
        
        # ACC Forgetting: ככל שהמספר חיובי יותר, המצב גרוע יותר (ה-ACC ירד)
        # חישוב: ACC הישן (גבוה) פחות ACC הנוכחי (נמוך)
        if conf_A is not None:
             acc_forgetting_A = acc_A_prev - conf_A["accuracy"]


    # ==========================================
    # חלק ה: החזרת התוצאות
    # ==========================================
    return {
        "accuracy": {
            "A":  None if conf_A is None else conf_A["accuracy"],
            "B":  None if conf_B is None else conf_B["accuracy"],
            "AB": conf_AB["accuracy"],
        },

        
        "AFID": afid_current,          # <--- המדד החדש והחשוב (הממוצע)
        "fid_forgetting_A": fid_forgetting_A,
        "acc_forgetting_A": acc_forgetting_A,
        "fid_A_prev": fid_A_prev,              # ה-FID המקורי (לפני האימון על B)
        "acc_A_prev": acc_A_prev,              # ה-ACC המקורי (לפני האימון על B)
        # --------------------------
        
        "FID": {
            "A":  fid_A_current,
            "B":  fid_B_current,
            #"Global_Mixing_AB": fid_global_AB, # זה ה-Global הישן
        },
        "cm10_AB": conf_AB["cm10"],
        "cm2_AB":  conf_AB["cm2"],
    }
