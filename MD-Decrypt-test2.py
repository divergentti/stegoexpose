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
                block = img[i:i+8, j:j+8]
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
        if coeffs is None or len(coeffs) == 0:
            print(f"No valid DCT coefficients for {self.image_path}")
            return None
        lsb = (coeffs % 2).astype(int)
        lsb = np.clip(lsb, 0, 1)
        counts = np.bincount(lsb, minlength=2)[:2]
        expected = np.array([len(lsb) / 2] * 2)
        if counts.shape != expected.shape:
            print(f"Shape mismatch in chisquare: counts={counts.shape}, expected={expected.shape}, lsb={np.unique(lsb)}")
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
        ], dtype=torch.float32)
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
    def __init__(self, alpha=0.5, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = torch.tensor([1.5, 1.0]).to(DEVICE)  # Balanced weights

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
    def __init__(self, original_dir: str, original_jpg_dir: str, stego_dirs: dict, transform=None, split='train', train_ratio=0.7):
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
            indices, train_size=train_ratio, stratify=labels, random_state=42
        )
        self.indices = train_indices if split == 'train' else val_indices
        print(f"Dataset: {len(self.data)} total, {len(train_indices)} train, {len(val_indices)} val")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path, label, tool = self.data[actual_idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        if self.transform:
            img = self.transform(img)
            if img.shape[0] != 3:
                raise ValueError(f"Image {img_path} has {img.shape[0]} channels, expected 3")
        return img, torch.tensor(label, dtype=torch.float32)


class Training:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss(alpha=0.5, gamma=1.5)
        self.optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.scaler = torch.amp.GradScaler('cuda')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.best_val_loss = float('inf')
        self.patience = 15
        self.epochs_no_improve = 0
        self.calibrator = None
        self.warmup_epochs = 5
        self.base_lr = 5e-5
        self.warmup_lr = 1e-6

    def calibrate(self, val_loader):
        self.model.eval()
        probs = []
        labels = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images).view(-1)
                probs.extend(torch.sigmoid(outputs).cpu().numpy())
                labels.extend(targets.cpu().numpy())
        probs = np.array(probs).reshape(-1, 1)
        labels = np.array(labels)
        self.calibrator = CalibratedClassifierCV(
            LogisticRegression(), cv=5, method='sigmoid'
        )
        self.calibrator.fit(probs, labels)
        print("Probability calibrator trained")

    def train(self, original_dir: str, original_jpg_dir: str, stego_dirs: dict, epochs: int = 100, batch_size: int = 32):
        train_dataset = StegoDataset(original_dir, original_jpg_dir, stego_dirs, self.transform, split='train', train_ratio=0.7)
        val_dataset = StegoDataset(original_dir, original_jpg_dir, stego_dirs, self.transform, split='val', train_ratio=0.7)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for epoch in range(epochs):
            if epoch < self.warmup_epochs:
                lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()

            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_preds = []
            train_labels = []

            self.optimizer.zero_grad()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(-1)

                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    outputs = outputs.view(-1)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                train_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                if batch_idx == 0:
                    print(f"Epoch {epoch + 1}, Sample Probs: {probs[:4].cpu().detach().numpy()}")
                    print(f"Epoch {epoch + 1}, Sample Labels: {labels[:4].cpu().detach().numpy()}")

            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            cm = confusion_matrix(train_labels, train_preds)
            print(f"Train Confusion Matrix:\n{cm}")

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []
            tool_correct = {"clean": 0, "openstego": 0, "steghide": 0, "outguess": 0}
            tool_total = {"clean": 0, "openstego": 0, "steghide": 0, "outguess": 0}

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    labels = labels.view(-1)
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        outputs = outputs.view(-1)
                        loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    for i, label in enumerate(labels):
                        tool = val_dataset.data[val_dataset.indices[val_total - labels.size(0) + i]][2]
                        tool_total[tool] += 1
                        if preds[i] == label:
                            tool_correct[tool] += 1

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            cm = confusion_matrix(val_labels, val_preds)
            print(f"Val Confusion Matrix:\n{cm}")
            print("Tool-specific Val Acc:")
            for tool in tool_correct:
                acc = tool_correct[tool] / tool_total[tool] if tool_total[tool] > 0 else 0
                print(f"{tool}: {acc:.4f} ({tool_correct[tool]}/{tool_total[tool]})")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_model("model_best.safetensors")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.calibrate(val_loader)
                    break

            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def save_model(self, path: str = "model.safetensors"):
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        save_file(state_dict, path)
        print(f"Model saved to {path}")

    def predict(self, stego_path: str, clean_path: str = None):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        try:
            stego_img = Image.open(stego_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {stego_path}: {e}")
            return None, None
        img = stego_img

        img = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img).view(-1)
            prob = torch.sigmoid(output).item()
            if self.calibrator is not None:
                prob = self.calibrator.predict_proba(np.array([[prob]]))[0, 1]
        prediction = "Stego" if prob >= 0.5 else "Clean"
        return prob, prediction


if __name__ == '__main__':
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    model = StegoCNN()
    train_model = False

    try:
        state_dict = load_file("model.safetensors", device=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Model file not found, starting model training...")
        train_model = True

    if train_model:
        trainer = Training(model, device=DEVICE)
        stego_dirs = {
            "openstego": "ml/stego/",
            "steghide": "ml/stego_steghide/",
            "outguess": "ml/stego_outguess/"
        }
        trainer.train(
            original_dir=ORIGINAL_DIR,
            original_jpg_dir=ORIGINAL_JPG_DIR,
            stego_dirs=stego_dirs,
            epochs=100,
            batch_size=16 # 32 cause CUDA out of memory
        )
        trainer.save_model("model_temp.safetensors")
        os.replace("model_temp.safetensors", "model.safetensors")

    image_paths = [
        "test_openstego.bmp",
        "test_steghide.bmp",
        "test_outguess.jpg",
        "test_unknown.bmp"
    ]

    trainer = Training(model, device=DEVICE)
    clean_phash_map = build_clean_phash_map(ORIGINAL_DIR, ORIGINAL_JPG_DIR)

    for path in image_paths:
        if not os.path.exists(path):
            print(f"{path}: File not found.")
            continue

        identifier = ToolIdentifier(path)
        tool = identifier.identify()
        print(f"{path}: Identified tool → {tool}")

        try:
            stego_phash = phash_image(path)
            if stego_phash is None:
                print(f"{path}: Failed to compute phash")
                continue
            closest_clean = min(clean_phash_map.items(), key=lambda kv: abs(stego_phash - kv[0]))
            if abs(stego_phash - closest_clean[0]) <= 2:
                clean_path = closest_clean[1]
                prob, prediction = trainer.predict(path, clean_path)
                if prob is None:
                    print(f"{path}: Prediction failed")
                    continue
                print(f"{path}: CNN prediction → {prediction} (prob={prob:.4f})")
            else:
                prob, prediction = trainer.predict(path)
                if prob is None:
                    print(f"{path}: Prediction failed")
                    continue
                print(f"{path}: No clean match, CNN prediction → {prediction} (prob={prob:.4f})")
        except Exception as e:
            print(f"{path}: Processing failed: {e}")

    print("\nValidating dataset...")
    sample_paths = [
        "ml/clean/image1.bmp",
        "ml/stego/image1.jpg.bmp",
        "ml/stego_steghide/image1_steghide.bmp",
        "ml/stego_outguess/image1_outguess.jpg"
    ]
    for path in sample_paths:
        if not os.path.exists(path):
            print(f"{path}: File not found")
            continue
        identifier = ToolIdentifier(path)
        tool = identifier.identify()
        print(f"{path}: Identified tool → {tool}")
