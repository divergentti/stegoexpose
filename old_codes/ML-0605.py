import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import save_file, load_file
import numpy as np
import torch.nn.functional as F
from scipy.stats import chisquare
import imagehash
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.model_selection import train_test_split
import pickle
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-GUI)
import matplotlib.pyplot as plt
import pandas as pd
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tkinter")
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {DEVICE}")
MAGIC_NUMBER = b"OPENSTEGO"
PASSPHRASE = "set"

MODEL_PATH = "./training/models/"
os.makedirs(MODEL_PATH, exist_ok=True)

MODEL_FILE = "stegomodel.safetensors"
CALIBRATOR_FILE= "calibrator.pkl"
ORIGINAL_DIR = "./training/originals/"
STEGO_OPENSTEGA_DIR = "./training/openstego/"
STEGO_STEGHIDE_DIR = "./training/steghide/"
STEGO_OUTGUESS_DIR = "./training/outguess/"
SECRET_MESSAGE_FILE = "./training/embeds/testfile.txt"
STEGO_METADATA_FILE = "./training/stego_metadata.csv"
VERIFY_MESSAGE_PATH = "./training/extracted/verify.txt"

debug_phash = True
debug_OpenStegoRandomLSB = True
debug_SteghideIdentifier=True
debug_OutGuessIdentifier=True
debug_ToolIdentifier=True
debug_StegoCNN=True
debug_StegoDataset=True
debug_train_model=True
debug_evaluate_model=True

def phash_image(path):
    try:
        return imagehash.phash(Image.open(path).convert("RGB"))
    except Exception as e:
        if debug_phash:
            print(f"Failed to hash {path}: {e}")
        return None


def build_clean_phash_map(clean_dir, clean_jpg_dir):
    phash_map = {}
    for dir_path in [clean_dir, clean_jpg_dir]:
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                full_path = os.path.join(dir_path, fname)
                ph = phash_image(full_path)
                if ph is not None:
                    phash_map[ph] = full_path
    return phash_map


class JavaRandom:
    def __init__(self, init_seed: int):
        self.seed = (init_seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def next(self, bits: int) -> int:
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.seed >> (48 - bits)

    def nextInt(self, n=None):
        if n is None:
            return self.next(32)
        if n <= 0:
            raise ValueError("Bound must be positive")
        if (n & -n) == n:
            return (n * self.next(31)) >> 31
        bits = self.next(31)
        val = bits % n
        while bits - val + (n - 1) < 0:
            bits = self.next(31)
            val = bits % n
        return val


class OpenStegoRandomLSB:
    def __init__(self, stego_image_path: str, password: str = None):
        self.stego_image_path = stego_image_path
        try:
            self.stego_image = Image.open(stego_image_path).convert("RGB")
            self.stego_np = np.array(self.stego_image)
        except Exception as e:
            if debug_OpenStegoRandomLSB:
                print(f"Error loading {stego_image_path}: {e}")
            self.stego_np = None
        self.magic = MAGIC_NUMBER

    def bits_to_bytes(self, bits: list) -> bytes:
        bytes_out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i + j]
            bytes_out.append(byte)
        return bytes(bytes_out)

    def extract_bits_randomly(self, stego_np, seed: int, num_bits: int, channel_mapping=None) -> list:
        if stego_np is None:
            return []
        height, width, channels = stego_np.shape
        rand = JavaRandom(seed)
        used_positions = set()
        extracted_bits = []
        channel_bits_used = 1

        while len(extracted_bits) < num_bits:
            x = rand.nextInt(width)
            y = rand.nextInt(height)
            c = rand.nextInt(channels)
            bit_pos = rand.nextInt(channel_bits_used)
            key = f"{x}_{y}_{c}_{bit_pos}"
            if key in used_positions:
                continue
            used_positions.add(key)

            mapped_c = channel_mapping[c] if channel_mapping else c
            bit = stego_np[y, x, mapped_c] & 1
            extracted_bits.append(int(bit))

        return extracted_bits

    def identify(self, seed: int = 98234782, bits_to_extract: int = 336) -> str:
        if self.stego_np is None:
            return None
        channel_mapping = [2, 1, 0]
        extracted_bits = self.extract_bits_randomly(self.stego_np, seed, bits_to_extract, channel_mapping)
        extracted_bytes = self.bits_to_bytes(extracted_bits)
        if extracted_bytes[:len(self.magic)] == self.magic:
            if debug_OpenStegoRandomLSB:
                print(f"OpenStego identified for {self.stego_image_path}")
            return "OpenStego"
        return None


class SteghideIdentifier:
    def __init__(self, image_path):
        self.image_path = image_path
        try:
            self.image = Image.open(image_path).convert("RGB")
            self.np_image = np.array(self.image)
        except Exception as e:
            if debug_SteghideIdentifier:
                print(f"Error loading {image_path}: {e}")
            self.np_image = None

    def extract_sequential_lsb(self, np_img, num_bits):
        if np_img is None:
            return []
        height, width, _ = np_img.shape
        extracted_bits = []
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    bit = np_img[y, x, c] & 1
                    extracted_bits.append(bit)
                    if len(extracted_bits) == num_bits:
                        return extracted_bits
        return extracted_bits

    def bits_to_bytes(self, bits):
        bytes_out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i + j]
            bytes_out.append(byte)
        return bytes(bytes_out)

    def identify(self, num_bits=4096):
        bits = self.extract_sequential_lsb(self.np_image, num_bits)
        if not bits:
            return None
        byte_data = self.bits_to_bytes(bits)
        patterns = [b'steghide', b'.txt\x00']
        for pattern in patterns:
            if pattern in byte_data:
                if debug_SteghideIdentifier:
                    print(f"Steghide identified for {self.image_path} with pattern {pattern}")
                return "Steghide"
        return None


class OutGuessIdentifier:
    def __init__(self, image_path):
        self.image_path = image_path
        try:
            self.image = Image.open(image_path).convert("RGB")
            self.np_image = np.array(self.image)
        except Exception as e:
            if debug_OutGuessIdentifier:
                print(f"Error loading {image_path}: {e}")
            self.np_image = None

    def extract_dct_coefficients(self):
        from scipy.fftpack import dct
        if self.np_image is None:
            return None
        img = self.np_image[:, :, 0]
        img = img.astype(float) - 128
        h, w = img.shape
        h = h - h % 8
        w = w - w % 8
        img = img[:h, :w]
        coeffs = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = img[i:i+8, j:j+8]
                coeff = dct(dct(block.T, norm='ortho').T, norm='ortho')
                coeffs.append(coeff.flatten())
        if not coeffs:
            if debug_OutGuessIdentifier:
                print(f"No DCT coefficients extracted for {self.image_path}")
            return None
        coeffs = np.concatenate(coeffs)
        return coeffs

    def identify(self):
        if not self.image_path.lower().endswith('.jpg'):
            return None
        coeffs = self.extract_dct_coefficients()
        if coeffs is None or len(coeffs) == 0:
            if debug_OutGuessIdentifier:
                print(f"No valid DCT coefficients for {self.image_path}")
            return None
        lsb = (coeffs % 2).astype(int)
        lsb = np.clip(lsb, 0, 1)
        counts = np.bincount(lsb, minlength=2)[:2]
        expected = np.array([len(lsb) / 2] * 2)
        try:
            chi2, p = chisquare(counts, expected)
            if debug_OutGuessIdentifier:
                print(f"OutGuess chi-square for {self.image_path}: p={p:.4f}")
            if p > 0.1:
                if debug_OutGuessIdentifier:
                    print(f"OutGuess identified for {self.image_path}")
                return "OutGuess"
        except Exception as e:
            if debug_OutGuessIdentifier:
                print(f"Chisquare error for {self.image_path}: {e}")
        return None


class ToolIdentifier:
    def __init__(self, image_path):
        self.image_path = image_path
        self.identifiers = [
            OpenStegoRandomLSB(image_path),
            SteghideIdentifier(image_path),
            OutGuessIdentifier(image_path),
        ]

    def identify(self):
        for identifier in self.identifiers:
            try:
                result = identifier.identify()
                if result:
                    return result
            except Exception as e:
                if debug_ToolIdentifier:
                    print(f"Identifier error for {identifier.__class__.__name__}: {e}")
        if debug_ToolIdentifier:
            print(f"No tool identified for {self.image_path}")
        return "Unknown"


class SRMFilter(nn.Module):
    def __init__(self):
        super(SRMFilter, self).__init__()
        srm_kernels = torch.tensor([
            [[[-1,  2, -2,  2, -1],
              [ 2, -6,  8, -6,  2],
              [-2,  8, -12, 8, -2],
              [ 2, -6,  8, -6,  2],
              [-1,  2, -2,  2, -1]]],  # KV kernel
            [[[ 0,  0,  0,  0,  0],
              [ 0, -1,  2, -1,  0],
              [ 0,  2, -4,  2,  0],
              [ 0, -1,  2, -1,  0],
              [ 0,  0,  0,  0,  0]]],  # Linear kernel
            [[[ 0,  0,  1,  0,  0],
              [ 0,  2, -8,  2,  0],
              [ 1, -8, 20, -8,  1],
              [ 0,  2, -8,  2,  0],
              [ 0,  0,  1,  0,  0]]],  # Square kernel
        ], dtype=torch.float32) * 2.0  # Increased amplification
        srm_kernels = srm_kernels / srm_kernels.abs().sum(dim=(2, 3), keepdim=True)
        self.conv = nn.Conv2d(3, 9, kernel_size=5, padding=2, bias=False)
        weights = torch.zeros(9, 3, 5, 5)
        for i in range(3):
            weights[i*3, i] = srm_kernels[0]
            weights[i*3+1, i] = srm_kernels[1]
            weights[i*3+2, i] = srm_kernels[2]
        self.conv.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        out = self.gamma * out + x
        return out


class StegoCNN(nn.Module):
    def __init__(self):
        super(StegoCNN, self).__init__()
        self.srm = SRMFilter()
        self.conv1 = nn.Conv2d(12, 24, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 96, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        self.attn1 = AttentionBlock(96)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.attn2 = AttentionBlock(192)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.52)
        self.fc1 = nn.Linear(192 * 16 * 16, 384)
        self.fc2 = nn.Linear(384, 1)
        self._initialize_weights()

    def forward(self, x, is_clean=None):
        x_srm = self.srm(x)
        if torch.any(torch.isnan(x_srm)) or torch.any(torch.isinf(x_srm)):
            if debug_StegoCNN:
                print("Warning: NaN or Inf in SRM output")
        if is_clean is not None:
            for i in range(len(is_clean)):
                label = "clean" if is_clean[i] else "stego"
                if debug_StegoCNN:
                    print(f"SRM Output ({label}) Mean: {x_srm[i].mean().item():.4f}, Std: {x_srm[i].std().item():.4f}")
        x = torch.cat([x, x_srm], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attn1(x)
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.attn2(x)
        x = self.pool(x)
        x = x.view(-1, 192 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):  # Increased alpha to heavily favor stego
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)  # Increased weight for stego class

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        weights = self.class_weights[targets.long()]
        F_loss = F_loss * weights
        return F_loss.mean()


class StegoDataset(Dataset):
    def __init__(self, original_dir: str, stego_dirs: dict, transform=None, split='train', train_ratio=0.7):
        self.transform = transform
        self.original_dir = original_dir
        self.stego_dirs = stego_dirs
        self.data = []
        self.split = split

        # Add clean images
        original_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.bmp', '.png', '.jpg'))]
        for fname in original_files:
            full_path = os.path.join(original_dir, fname)
            if os.path.isfile(full_path):
                self.data.append((full_path, 0.0, "clean"))

        if debug_StegoDataset:
            print(f"Loaded {len(original_files)} clean images from {original_dir}")

        # Add stego images from metadata CSV
        if not os.path.exists(STEGO_METADATA_FILE):
            raise FileNotFoundError(f"Metadata file not found: {STEGO_METADATA_FILE}")
        df = pd.read_csv(STEGO_METADATA_FILE)
        for _, row in df.iterrows():
            tool = row['tool']
            fname = row['outfile']
            if tool not in stego_dirs:
                if debug_StegoDataset:
                    print(f"Unknown tool '{tool}' in metadata")
                continue
            full_path = os.path.join(stego_dirs[tool], fname)
            if os.path.isfile(full_path):
                self.data.append((full_path, 1.0, tool))
            else:
                if debug_StegoDataset:
                    print(f"Stego file not found: {full_path}")

        if not self.data:
            if debug_StegoDataset:
                raise ValueError("No image data found for training.")

        from sklearn.model_selection import train_test_split
        labels = [label for _, label, _ in self.data]
        strat_labels = [(label, tool) for _, label, tool in self.data]
        indices = list(range(len(self.data)))
        train_indices, val_indices = train_test_split(
            list(range(len(self.data))),
            test_size=0.3,
            stratify=strat_labels,
            random_state=42
        )
        self.indices = train_indices if split == 'train' else val_indices

        if split == 'train':
            clean_indices = [i for i in self.indices if self.data[i][2] == "clean"]
            num_stego = len([i for i in self.indices if self.data[i][2] != "clean"])
            self.indices += random.sample(clean_indices, min(len(clean_indices), num_stego))  # Match stego count
            if debug_StegoDataset:
                print(f"Oversampled clean images: +{min(len(clean_indices), num_stego)}")

        if debug_StegoDataset:
            print(f"Final dataset size: {len(self.indices)} ({split})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path, label, tool = self.data[actual_idx]

        try:
            with Image.open(img_path).convert('RGB') as image:
                image_data = np.array(image)

                # Apply stronger augmentation for stego class
                if label == 1.0 and self.split == 'train':
                    # Apply more aggressive augmentation for stego
                    transform_stego = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.7),
                        transforms.RandomVerticalFlip(p=0.7),
                        transforms.RandomRotation(30),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1),
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])
                    transformed_image = transform_stego(Image.fromarray(image_data))
                else:
                    # Regular transform for clean
                    transformed_image = self.transform(Image.fromarray(image_data))

            return transformed_image, torch.tensor(label, dtype=torch.float32), tool
        except Exception as e:
            if debug_StegoDataset:
                print(f"Error loading image {img_path}: {e}")
            blank = Image.new('RGB', (256, 256), color='black')
            if self.transform:
                blank = self.transform(blank)
            return blank, torch.tensor(0.0, dtype=torch.float32), "failed"

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=100, save_path=None):
    best_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None

    for epoch in range(num_epochs):
        # Manual warmup for the first 5 epochs
        if epoch < 5:  # 5-epoch warmup
            lr = 1e-6 + (3e-5 - 1e-6) * epoch / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # After warmup, let the scheduler handle the learning rate
        elif epoch == 5:
            # Reset the learning rate to the initial value after warmup
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-5  # Match the initial lr in optimizer

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_preds, train_labels = [], []

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda') if DEVICE == 'cuda' else torch.no_grad():
                outputs = model(inputs, is_clean=targets == 0)
                loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(targets.cpu().numpy())

            if batch_idx == 0:
                if debug_train_model:
                    print(f"Epoch {epoch + 1}, Sample Probs: {probs[:4].cpu().detach().numpy().flatten()}")
                    print(f"Epoch {epoch + 1}, Sample Labels: {targets[:4].cpu().detach().numpy()}")

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        if debug_train_model:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        cm = confusion_matrix(train_labels, train_preds, labels=[0, 1])
        if debug_train_model:
            print(f"Train Confusion Matrix:\n{cm}")

        if debug_train_model and epoch % 5 == 0:
            stego_correct = sum(1 for p, t in zip(train_preds, train_labels) if t == 1 and p == t)
            print(f"Epoch {epoch + 1} | Stego Correct: {stego_correct}/{sum(1 for t in train_labels if t == 1)}")

        if (epoch + 1) % 5 == 0:  # Print validation every 5 epochs
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    with torch.amp.autocast('cuda') if DEVICE == 'cuda' else torch.no_grad():
                        outputs = model(inputs, is_clean=targets == 0)
                        loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(targets.cpu().numpy())

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if debug_train_model:
                print(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
            if debug_train_model:
                print(f"Val Confusion Matrix:\n{cm}")

            plt.figure(figsize=(6, 4))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Epoch {epoch + 1} Validation Confusion Matrix")
            plt.colorbar()
            plt.xticks([0, 1], ['Clean', 'Stego'])
            plt.yticks([0, 1], ['Clean', 'Stego'])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f"confusion_matrix_epoch_{epoch + 1}.png")
            plt.close()

            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                epochs_no_improve = 0
                state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                save_file(state_dict, save_path)
                if debug_train_model:
                    print(f"Saved best model to {save_path}")
            else:
                epochs_no_improve += 5  # Account for 5 epochs between validations

            # Early stopping check
            if epochs_no_improve >= patience:
                if debug_train_model:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Step the scheduler after each epoch (after warmup)
        if epoch >= 5:  # Start using the scheduler after the warmup phase
            scheduler.step()
            if debug_train_model:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1} Learning Rate: {current_lr:.6f}")

    return history


def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_targets, all_scores, all_tools = [], [], [], []

    with torch.no_grad():
        for inputs, targets, tools in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            preds = (scores > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_tools.extend(tools)

            if len(all_preds) <= 8:  # First few samples
                if debug_evaluate_model:
                    print(f"Eval Sample Probs: {scores[:4].cpu().numpy().flatten()}")
                    print(f"Eval Sample Labels: {targets[:4].cpu().numpy()}")

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_scores = np.array(all_scores).flatten()

    accuracy = (all_preds == all_targets).mean()
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Plot probability histogram
    clean_scores = all_scores[all_targets == 0]
    stego_scores = all_scores[all_targets == 1]
    plt.figure(figsize=(8, 6))
    plt.hist(clean_scores, bins=50, alpha=0.5, label='Clean', color='blue')
    plt.hist(stego_scores, bins=50, alpha=0.5, label='Stego', color='red')
    plt.title('Validation Probability Distribution')
    plt.xlabel('Calibrated Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('probability_histogram.png')
    plt.close()

    tool_metrics = {}
    for tool in set(all_tools):
        if tool == "failed":
            continue
        tool_indices = [i for i, t in enumerate(all_tools) if t == tool]
        if not tool_indices:
            continue

        tool_preds = all_preds[tool_indices]
        tool_targets = all_targets[tool_indices]
        tool_cm = confusion_matrix(tool_targets, tool_preds, labels=[0, 1])
        tn, fp, fn, tp = tool_cm.ravel()

        tool_metrics[tool] = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1': 0,
            'confusion_matrix': tool_cm
        }
        tool_metrics[tool]['f1'] = 2 * tool_metrics[tool]['precision'] * tool_metrics[tool]['sensitivity'] / (
            tool_metrics[tool]['precision'] + tool_metrics[tool]['sensitivity']
        ) if (tool_metrics[tool]['precision'] + tool_metrics[tool]['sensitivity']) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': cm,
        'tool_metrics': tool_metrics,
        'predictions': all_preds,
        'targets': all_targets,
        'scores': all_scores,
        'tools': all_tools
    }

    return metrics


def calibrate_model(model, val_loader):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).cpu().numpy().flatten()
            features.extend(outputs)
            labels.extend(targets.cpu().numpy().flatten())

    features = np.array(features).reshape(-1, 1)
    labels = np.array(labels)
    lr = LogisticRegression(C=0.1)
    calibrated_model = CalibratedClassifierCV(lr, method='sigmoid')
    calibrated_model.fit(features, labels)
    return calibrated_model


def predict_with_calibration(model, calibrator, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        raw_output = model(image_tensor).cpu().numpy().flatten()[0]
    calibrated_prob = calibrator.predict_proba(np.array([[raw_output]]))[0, 1]
    return raw_output, calibrated_prob


def load_model(model_path):
    model = StegoCNN().to(DEVICE)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_random_image_paths(metadata_file, stego_dirs, original_dir, num_samples=5):
    # Read the metadata CSV
    df = pd.read_csv(metadata_file)

    # Randomly sample rows
    sampled_rows = df.sample(n=num_samples, random_state=42)  # random_state for reproducibility

    # Extract paths
    image_paths = []
    sample_paths = []

    for _, row in sampled_rows.iterrows():
        # Add stego image path
        tool = row['tool']
        stego_path = os.path.join(stego_dirs.get(tool, original_dir), row['outfile'])
        if os.path.exists(stego_path):
            image_paths.append(stego_path)
            sample_paths.append(stego_path)

        # Add original image path (if not already included)
        original_path = os.path.join(original_dir, row['original'])
        if os.path.exists(original_path) and original_path not in image_paths:
            image_paths.append(original_path)
            sample_paths.append(original_path)

    # Ensure we have at least num_samples unique paths
    while len(image_paths) < num_samples:
        additional_row = df.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]
        tool = additional_row['tool']
        stego_path = os.path.join(stego_dirs.get(tool, original_dir), additional_row['outfile'])
        if os.path.exists(stego_path) and stego_path not in image_paths:
            image_paths.append(stego_path)
            sample_paths.append(stego_path)

    return image_paths[:num_samples], sample_paths[:num_samples]


def balance_dataset(self):
    # Count samples by class and tool
    clean_count = sum(1 for _, label, _ in self.data if label == 0.0)
    stego_by_tool = {}
    for _, label, tool in self.data:
        if label == 1.0:
            stego_by_tool[tool] = stego_by_tool.get(tool, 0) + 1

    print(f"Initial counts - Clean: {clean_count}, Stego: {sum(stego_by_tool.values())}")
    print(f"Stego by tool: {stego_by_tool}")

    # Balance across both classes and tools
    total_stego = sum(stego_by_tool.values())

    # Sample same number of clean as total stego images
    if clean_count > total_stego:
        clean_indices = [i for i in range(len(self.data)) if self.data[i][1] == 0.0]
        keep_clean = random.sample(clean_indices, total_stego)

        # Keep only the sampled clean indices and all stego indices
        new_data = [self.data[i] for i in range(len(self.data))
                    if self.data[i][1] == 1.0 or i in keep_clean]
        self.data = new_data

    print(f"After balancing - Clean: {sum(1 for _, label, _ in self.data if label == 0.0)}, "
          f"Stego: {sum(1 for _, label, _ in self.data if label == 1.0)}")


def create_balanced_sampler(dataset):
    targets = [int(label) for _, label, _ in dataset.data]
    class_counts = torch.bincount(torch.tensor(targets))
    weight = 1. / class_counts.float()
    sample_weights = torch.tensor([weight[t] for t in targets])
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler



def main():
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    stego_dirs = {
        "openstego": STEGO_OPENSTEGA_DIR,
        "steghide": STEGO_STEGHIDE_DIR,
        "outguess": STEGO_OUTGUESS_DIR
    }

    train_dataset = StegoDataset(
        original_dir=ORIGINAL_DIR,
        stego_dirs=stego_dirs,
        transform=transform_train,
        split='train'
    )

    val_dataset = StegoDataset(
        original_dir=ORIGINAL_DIR,
        stego_dirs=stego_dirs,
        transform=transform_val,
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=28,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=28,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = StegoCNN().to(DEVICE)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=100,
        save_path=MODEL_PATH + MODEL_FILE
    )
    model = load_model(MODEL_PATH+MODEL_FILE)
    metrics = evaluate_model(model, val_loader)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity (TPR): {metrics['sensitivity']:.4f}")
    print(f"Specificity (TNR): {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nTool-specific metrics:")
    for tool, tool_metrics in metrics['tool_metrics'].items():
        print(f"\n{tool}:")
        for metric_name, metric_value in tool_metrics.items():
            if metric_name == 'confusion_matrix':
                print(f"  {metric_name}:\n{metric_value}")
            else:
                print(f"  {metric_name}:{metric_value:.4f}")

    calibrator = calibrate_model(model, val_loader)
    with open(MODEL_PATH+CALIBRATOR_FILE, "wb") as f:
        pickle.dump(calibrator, f)

    def predict_stego(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform_val(image).to(DEVICE)
            raw_score, calibrated_prob = predict_with_calibration(model, calibrator, image_tensor)
            is_stego = calibrated_prob > 0.5
            tool_identifier = ToolIdentifier(image_path)
            stego_tool = tool_identifier.identify() if is_stego else None
            result = {
                'is_stego': bool(is_stego),
                'raw_score': float(raw_score),
                'calibrated_probability': float(calibrated_prob),
                'stego_tool': stego_tool
            }
            return result
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    # Get random paths
    num_samples = 5  # Adjust this number as needed
    image_paths, sample_paths = get_random_image_paths(STEGO_METADATA_FILE, stego_dirs, ORIGINAL_DIR, num_samples)

    print("\nVerifying test images with ToolIdentifier:")
    for path in image_paths:
        if not os.path.exists(path):
            print(f"{path}: File not found.")
            continue
        identifier = ToolIdentifier(path)
        tool = identifier.identify()
        print(f"{path}: Identified tool → {tool}")

    print("\nTesting predictions:")
    try:
        clean_phash_map = build_clean_phash_map(ORIGINAL_DIR, ORIGINAL_DIR)  # Fixed call
    except TypeError as e:
        print(f"Error building clean phash map: {e}. Continuing without phash comparison.")
        clean_phash_map = {}
    for path in image_paths:
        if not os.path.exists(path):
            print(f"{path}: File not found.")
            continue
        result = predict_stego(path)
        if result:
            print(f"{path}:")
            print(f"  Is Stego: {result['is_stego']}")
            print(f"  Calibrated Probability: {result['calibrated_probability']:.4f}")
            print(f"  Raw Score: {result['raw_score']:.4f}")
            print(f"  Stego Tool: {result['stego_tool']}")
            stego_phash = phash_image(path)
            if stego_phash:
                closest_clean = min(clean_phash_map.items(), key=lambda kv: abs(stego_phash - kv[0]))
                if abs(stego_phash - closest_clean[0]) <= 2:
                    print(f"  Closest Clean Match: {closest_clean[1]}")
                else:
                    print(f"  No clean match found")

    print("\nValidating dataset...")
    for path in sample_paths:
        if not os.path.exists(path):
            print(f"{path}: File not found")
        else:
            identifier = ToolIdentifier(path)
            tool = identifier.identify()
            print(f"{path}: Identified tool → {tool}")

    print("Training and evaluation complete. Model and calibrator saved.")
    return history, metrics


if __name__ == "__main__":
    history, metrics = main()