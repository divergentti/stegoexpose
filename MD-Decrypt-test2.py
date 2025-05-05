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
import re
from scipy.stats import chisquare
import imagehash
from sklearn.metrics import confusion_matrix
import torchvision
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import warnings
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {DEVICE}")
ORIGINAL_DIR = "ml/clean/"
ORIGINAL_JPG_DIR = "ml/clean.jpg/"
MAGIC_NUMBER = b"OPENSTEGO"


def phash_image(path):
    try:
        return imagehash.phash(Image.open(path).convert("RGB"))
    except Exception as e:
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


def compute_histogram_features(img_tensor):
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    diff = np.diff(img_np, axis=0)  # Vertical pixel differences
    hist, _ = np.histogram(diff.ravel(), bins=256, range=(-255, 255))
    hist = hist / (hist.sum() + 1e-6)  # Normalize
    return torch.tensor(hist, dtype=torch.float32).view(1, 256, 1, 1)


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
            return "OpenStego"
        return None


class SteghideIdentifier:
    def __init__(self, image_path):
        self.image_path = image_path
        try:
            self.image = Image.open(image_path).convert("RGB")
            self.np_image = np.array(self.image)
        except Exception as e:
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
        patterns = [b'\x00\x00\x00', b'\xFF\xFF\xFF', b'steghide', b'.txt\x00']
        if any(b in byte_data for b in patterns):
            return "Steghide"
        return None


class OutGuessIdentifier:
    def __init__(self, image_path):
        self.image_path = image_path
        try:
            self.image = Image.open(image_path).convert("RGB")
            self.np_image = np.array(self.image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            self.np_image = None

    def extract_dct_coefficients(self):
        from scipy.fftpack import dct
        if self.np_image is None:
            return None
        img = self.np_image[:, :, 0]
        img = img.astype(float) - 128
        h, w = img.shape
        if h < 8 or w < 8:
            print(f"Image {self.image_path} too small: {h}x{w}")
            return None
        h = h - h % 8
        w = w - w % 8
        img = img[:h, :w]
        coeffs = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = img[i:i + 8, j:j + 8]
                coeff = dct(dct(block.T, norm='ortho').T, norm='ortho')
                coeffs.append(coeff.flatten())
        if not coeffs:
            print(f"No DCT coefficients extracted for {self.image_path}")
            return None
        coeffs = np.concatenate(coeffs)
        return coeffs

    def identify(self):
        if not self.image_path.lower().endswith('.jpg'):
            return None
        coeffs = self.extract_dct_coefficients()
        if coeffs is None:
            print(f"No valid DCT coefficients for {self.image_path}")
            return None
        lsb = (coeffs % 2).astype(int)
        lsb = np.clip(lsb, 0, 1)
        counts = np.bincount(lsb, minlength=2)[:2]
        expected = np.array([len(lsb) / 2] * 2)
        if counts.shape != expected.shape:
            print(
                f"Shape mismatch in chisquare: counts={counts.shape}, expected={expected.shape}, lsb={np.unique(lsb)}")
            return None
        try:
            chi2, p = chisquare(counts, expected)
            if p > 0.05:
                return "OutGuess"
        except Exception as e:
            print(f"Chisquare error for {self.image_path}: {e}")
        return None


class ToolIdentifier:
    def __init__(self, image_path):
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
                print(f"Identifier error for {identifier.__class__.__name__}: {e}")
        return "Unknown"


class SRMFilter(nn.Module):
    def __init__(self):
        super(SRMFilter, self).__init__()
        srm_kernels = torch.tensor([
            [[[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]],  # KV kernel
            [[[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]],  # Linear kernel
            [[[0, 0, 1, 0, 0],
              [0, 2, -8, 2, 0],
              [1, -8, 20, -8, 1],
              [0, 2, -8, 2, 0],
              [0, 0, 1, 0, 0]]],  # Square kernel
        ], dtype=torch.float32)
        srm_kernels = srm_kernels / srm_kernels.abs().sum(dim=(2, 3), keepdim=True)
        self.conv = nn.Conv2d(3, 9, kernel_size=5, padding=2, bias=False)
        weights = torch.zeros(9, 3, 5, 5)
        for i in range(3):
            weights[i * 3, i] = srm_kernels[0]
            weights[i * 3 + 1, i] = srm_kernels[1]
            weights[i * 3 + 2, i] = srm_kernels[2]
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
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attn1 = AttentionBlock(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.attn2 = AttentionBlock(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)
        self._initialize_weights()

    def forward(self, x):
        x_srm = self.srm(x)
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
        x = x.view(-1, 256 * 16 * 16)
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
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Significantly higher weight for clean class (label 0)
        self.class_weights = torch.tensor([5.0, 1.0]).to(DEVICE)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        # Apply higher weight for clean class
        weights = self.class_weights[targets.long()]
        F_loss = F_loss * weights
        return F_loss.mean()


# Mixup data augmentation helper functions
def mixup_data(x, y, alpha=0.2):
    '''Mixup data augmentation'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Mixup loss calculation'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class StegoDataset(Dataset):
    def __init__(self, original_dir: str, original_jpg_dir: str, stego_dirs: dict, transform=None, split='train',
                 train_ratio=0.7):
        self.transform = transform
        self.original_dir = original_dir
        self.original_jpg_dir = original_jpg_dir
        self.stego_dirs = stego_dirs

        # Collect clean images (use only ml/clean)
        self.original_files = []
        if os.path.exists(original_dir):
            files = sorted(
                [f for f in os.listdir(original_dir) if f.lower().endswith('.bmp')],
                key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x
            )
            self.original_files.extend([os.path.join(original_dir, f) for f in files])
            print(f"{original_dir}: {len(files)} clean images found")
        else:
            print(f"Warning: Directory {original_dir} does not exist")

        if not self.original_files:
            raise ValueError("No clean images found in ml/clean")

        # Collect stego images
        self.stego_files = {}
        for tool, stego_dir in stego_dirs.items():
            ext = '.jpg' if "outguess" in tool else '.bmp'
            if not os.path.exists(stego_dir):
                raise ValueError(f"Stego directory {stego_dir} does not exist")
            stego_files = sorted(
                [f for f in os.listdir(stego_dir) if f.lower().endswith(ext)],
                key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x
            )
            self.stego_files[tool] = [os.path.join(stego_dir, f) for f in stego_files]
            print(f"{tool}: {len(stego_files)} stego images found")

        # Build dataset: include clean images with at least one stego match
        self.data = []
        clean_base_names = set()
        for orig_path in self.original_files:
            base_name = re.search(r'image(\d+)', os.path.basename(orig_path))
            if not base_name:
                print(f"Warning: Invalid clean filename {os.path.basename(orig_path)}")
                continue
            base_name = base_name.group(1)
            clean_base_names.add(base_name)
            self.data.append((orig_path, 0.0, "clean"))

        for tool, stego_files in self.stego_files.items():
            for stego_path in stego_files:
                base_name = None
                if "openstego" in tool:
                    match = re.search(r'image(\d+)\.jpg\.bmp$', os.path.basename(stego_path))
                    if match:
                        base_name = match.group(1)
                elif "steghide" in tool:
                    match = re.search(r'image(\d+)_steghide\.bmp$', os.path.basename(stego_path))
                    if match:
                        base_name = match.group(1)
                elif "outguess" in tool:
                    match = re.search(r'image(\d+)_outguess\.jpg$', os.path.basename(stego_path))
                    if match:
                        base_name = match.group(1)
                if not base_name:
                    print(f"Warning: Invalid stego filename {os.path.basename(stego_path)} in {tool}")
                    continue
                if base_name in clean_base_names:
                    self.data.append((stego_path, 1.0, tool))
                else:
                    print(f"Warning: No clean match for stego image{base_name} in {tool}")

        if not self.data:
            raise ValueError("No clean-stego pairs found")

        # Log dataset composition
        tool_counts = {"clean": 0, "openstego": 0, "steghide": 0, "outguess": 0}
        for _, label, tool in self.data:
            tool_counts[tool] += 1
        print("Dataset composition:", tool_counts)

        # Stratified train/val split
        from sklearn.model_selection import train_test_split
        labels = [label for _, label, _ in self.data]
        indices = list(range(len(self.data)))
        train_indices, val_indices = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels,
            random_state=42,
            shuffle=True  # Ensure shuffling before split
        )
        self.indices = train_indices if split == 'train' else val_indices

        # Add oversampling for clean images in training set
        if split == 'train':
            clean_indices = [i for i in self.indices if self.data[i][2] == "clean"]
            # Duplicate clean indices 3 times to balance the dataset
            extra_clean_indices = clean_indices * 3
            self.indices = self.indices + extra_clean_indices
            print(
                f"After oversampling: {len(self.indices)} samples ({len(extra_clean_indices)} additional clean samples)")

        print(f"Dataset: {len(self.data)} total, {len(self.indices)} {'train' if split == 'train' else 'val'}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path, label, tool = self.data[actual_idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32), tool
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as a fallback
            if self.transform:
                blank = torch.zeros(3, 256, 256)
            else:
                blank = Image.new('RGB', (256, 256), color='black')
                if self.transform:
                    blank = self.transform(blank)
            return blank, torch.tensor(0.0, dtype=torch.float32), "failed"


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, mixup_alpha=0.2, save_path=None):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Apply mixup augmentation
            inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)

            optimizer.zero_grad()
            outputs = model(inputs_mixed)

            # Calculate loss with mixup
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy (using original inputs)
            with torch.no_grad():
                outputs_orig = model(inputs)
                predicted = (torch.sigmoid(outputs_orig) > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)

            if batch_idx % 10 == 0:
                print(
                    f'Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # Save best model
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            # Save model using safetensors
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, save_path)
            print(f"Saved best model to {save_path}")

    return history


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_scores = []
    all_tools = []

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

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_scores = np.array(all_scores).flatten()

    # Calculate metrics
    accuracy = (all_preds == all_targets).mean()
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Tool-specific metrics
    tool_metrics = {}
    for tool in set(all_tools):
        if tool == "failed":
            continue
        tool_indices = [i for i, t in enumerate(all_tools) if t == tool]
        if not tool_indices:
            continue

        tool_preds = all_preds[tool_indices]
        tool_targets = all_targets[tool_indices]

        if len(set(tool_targets)) < 2:  # Need both classes for confusion matrix
            continue

        tool_cm = confusion_matrix(tool_targets, tool_preds)

        if tool_cm.size == 4:  # Ensure it's a 2x2 matrix
            tn, fp, fn, tp = tool_cm.ravel()
            tool_metrics[tool] = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            }
            tool_metrics[tool]['f1'] = 2 * tool_metrics[tool]['precision'] * tool_metrics[tool]['sensitivity'] / (
                    tool_metrics[tool]['precision'] + tool_metrics[tool]['sensitivity']) if (
                                                                                                    tool_metrics[tool][
                                                                                                        'precision'] +
                                                                                                    tool_metrics[tool][
                                                                                                        'sensitivity']) > 0 else 0

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
    """Calibrate model probabilities using Platt scaling"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).cpu().numpy().flatten()
            features.extend(outputs)
            labels.extend(targets.numpy().flatten())

    features = np.array(features).reshape(-1, 1)
    labels = np.array(labels)

    # Train logistic regression for calibration
    lr = LogisticRegression(C=1.0)
    calibrated_model = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
    calibrated_model.fit(features, labels)

    return calibrated_model


def predict_with_calibration(model, calibrator, image_tensor):
    """Make a calibrated prediction for a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        raw_output = model(image_tensor).cpu().numpy().flatten()[0]

    # Apply calibration
    calibrated_prob = calibrator.predict_proba(np.array([[raw_output]]))[0, 1]
    return raw_output, calibrated_prob


def load_model(model_path):
    """Load a model from a safetensors file"""
    model = StegoCNN().to(DEVICE)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    # Data augmentation and preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Define stego directories
    stego_dirs = {
        "openstego": "ml/openstego/",
        "steghide": "ml/steghide/",
        "outguess": "ml/outguess/"
    }

    # Create datasets
    train_dataset = StegoDataset(
        original_dir=ORIGINAL_DIR,
        original_jpg_dir=ORIGINAL_JPG_DIR,
        stego_dirs=stego_dirs,
        transform=transform_train,
        split='train'
    )

    val_dataset = StegoDataset(
        original_dir=ORIGINAL_DIR,
        original_jpg_dir=ORIGINAL_JPG_DIR,
        stego_dirs=stego_dirs,
        transform=transform_val,
        split='val'
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = StegoCNN().to(DEVICE)

    # Loss function and optimizer
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=15,
        mixup_alpha=0.2,
        save_path="stego_model.safetensors"
    )

    # Load best model
    model = load_model("stego_model.safetensors")

    # Evaluate model
    metrics = evaluate_model(model, val_loader)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity (TPR): {metrics['sensitivity']:.4f}")
    print(f"Specificity (TNR): {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])

    # Tool-specific metrics
    print("\nTool-specific metrics:")
    for tool, tool_metrics in metrics['tool_metrics'].items():
        print(f"\n{tool}:")
        for metric_name, metric_value in tool_metrics.items():
            if metric_name != 'confusion_matrix':
                print(f"  {metric_name}: {metric_value:.4f}")

    # Calibrate model
    calibrator = calibrate_model(model, val_loader)

    # Create a function for predicting new images
    def predict_stego(image_path):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform_val(image).to(DEVICE)

            # Get raw model prediction and calibrated probability
            raw_score, calibrated_prob = predict_with_calibration(model, calibrator, image_tensor)

            # Predict class
            is_stego = calibrated_prob > 0.5

            # Try to identify the tool
            if is_stego:
                tool_identifier = ToolIdentifier(image_path)
                stego_tool = tool_identifier.identify()
            else:
                stego_tool = None

            result = {
                'is_stego': bool(is_stego),
                'raw_score': float(raw_score),
                'calibrated_probability': float(calibrated_prob),
                'stego_tool': stego_tool if is_stego else None,
            }

            return result
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    # Example usage of the predict function
    # result = predict_stego("path/to/test/image.jpg")
    # print(result)

    # Save the calibrator
    import pickle
    with open("calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

    print("Training and evaluation complete. Model and calibrator saved.")


if __name__ == "__main__":
    main()
