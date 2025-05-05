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
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
import torchvision
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
import matplotlib.pyplot as plt

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
            # print(f"OpenStego identified for {self.stego_image_path}") # Less verbose
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
        patterns = [b'steghide', b'.txt\x00']
        for pattern in patterns:
            if pattern in byte_data:
                # print(f"Steghide identified for {self.image_path} with pattern {pattern}") # Less verbose
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
            # print(f"No DCT coefficients extracted for {self.image_path}") # Less verbose
            return None
        coeffs = np.concatenate(coeffs)
        return coeffs

    def identify(self):
        if not self.image_path.lower().endswith('.jpg'):
            return None
        coeffs = self.extract_dct_coefficients()
        if coeffs is None or len(coeffs) == 0:
            # print(f"No valid DCT coefficients for {self.image_path}") # Less verbose
            return None
        # Extract LSBs, avoiding coefficients that are 0 or 1
        valid_coeffs = coeffs[(coeffs != 0) & (coeffs != 1)]
        if len(valid_coeffs) < 100: # Need enough coefficients for statistical test
             return None
        lsb = (valid_coeffs.astype(int) % 2)
        counts = np.bincount(lsb, minlength=2)[:2]

        # Check if counts are too skewed (can happen with heavily processed images)
        if min(counts) < 5: # Chi-square needs minimum expected frequency
             return None

        expected = np.array([len(lsb) / 2] * 2)
        try:
            chi2, p = chisquare(counts, expected)
            # print(f"OutGuess chi-square for {self.image_path}: p={p:.4f}") # Less verbose
            # Use a less strict p-value threshold for OutGuess detection, as its signature is statistical
            if p > 0.01:
                # print(f"OutGuess identified for {self.image_path}") # Less verbose
                return "OutGuess"
        except Exception as e:
            print(f"Chisquare error for {self.image_path}: {e}")
        return None


class ToolIdentifier:
    def __init__(self, image_path):
        self.image_path = image_path
        self.identifiers = [
            OpenStegoRandomLSB(image_path),
            # SteghideIdentifier(image_path), # Often less reliable or conflicts
            OutGuessIdentifier(image_path),
        ]

    def identify(self):
        # Prioritize OutGuess for JPGs
        if self.image_path.lower().endswith('.jpg'):
            try:
                result = self.identifiers[1].identify() # OutGuessIdentifier index
                if result:
                    return result
            except Exception as e:
                 print(f"Identifier error for {self.identifiers[1].__class__.__name__}: {e}")

        # Check other identifiers
        for identifier in self.identifiers:
             # Skip OutGuess if already checked or not JPG
            if isinstance(identifier, OutGuessIdentifier) and self.image_path.lower().endswith('.jpg'):
                continue
            try:
                result = identifier.identify()
                if result:
                    return result
            except Exception as e:
                print(f"Identifier error for {identifier.__class__.__name__}: {e}")
        # print(f"No specific tool identified for {self.image_path}") # Less verbose
        return "Unknown"


class SRMFilter(nn.Module):
    def __init__(self):
        super(SRMFilter, self).__init__()
        # Original SRM kernels, consider adjusting amplification if needed
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
        ], dtype=torch.float32) # * 2.0 <- Removed amplification here, test if needed
        # Normalize kernels
        srm_kernels = srm_kernels / srm_kernels.abs().sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)

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
        # Input channels: 3 (original) + 9 (SRM) = 12
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1) # Reduced filters
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Reduced filters
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # Reduced filters
        self.bn3 = nn.BatchNorm2d(128)
        # self.attn1 = AttentionBlock(128) # Optional: remove attention?
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # Reduced filters
        self.bn4 = nn.BatchNorm2d(256)
        # self.attn2 = AttentionBlock(256) # Optional: remove attention?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) # Slightly reduced dropout
        # Recalculate linear layer input size based on pooling: 256 / 2 / 2 / 2 / 2 = 16x16
        # Input features to FC layer: 256 channels * 16 * 16 spatial dimensions
        self.fc1 = nn.Linear(256 * 16 * 16, 512) # Adjusted FC layer size
        self.fc2 = nn.Linear(512, 1)
        self._initialize_weights()

    def forward(self, x, is_clean=None):
        x_orig = x
        x_srm = self.srm(x_orig)
        # --- Debug SRM Output ---
        # if is_clean is not None:
        #     for i in range(min(len(is_clean), 4)): # Print for first few samples
        #         label = "clean" if is_clean[i] else "stego"
        #         print(f"SRM Output ({label}, idx {i}) Mean: {x_srm[i].mean().item():.4f}, Std: {x_srm[i].std().item():.4f}")
        # --- End Debug ---
        x = torch.cat([x_orig, x_srm], dim=1) # Concatenate original and SRM features

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.attn1(x) # Optional
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        # x = self.attn2(x) # Optional
        x = self.pool(x)

        # print(f"Shape before flatten: {x.shape}") # Debug shape
        x = x.view(x.size(0), -1) # Flatten dynamically
        # print(f"Shape after flatten: {x.shape}") # Debug shape

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
                if m.bias is not None:
                   nn.init.constant_(m.bias, 0)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Alpha can still be tuned
        self.gamma = gamma
        # --- Increased weight for clean class (0) ---
        self.class_weights = torch.tensor([5.0, 1.0]).to(DEVICE)
        # --- End Change ---
        print(f"FocalLoss using class weights: {self.class_weights.cpu().numpy()}")


    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Use weights directly in BCEWithLogitsLoss for stability and clarity
        weights = self.class_weights[targets.long()]
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=weights)

        pt = torch.exp(-BCE_loss) # Careful: BCE_loss here already includes class weights
        # Adjust Focal loss calculation to avoid applying weights twice
        # Option 1: Use original BCE loss to calculate pt
        BCE_loss_unweighted = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt_unweighted = torch.exp(-BCE_loss_unweighted)
        F_loss = self.alpha * (1 - pt_unweighted) ** self.gamma * BCE_loss # Apply weights here via BCE_loss

        # Option 2: Modify alpha based on target class (effectively similar to weights)
        # alpha_t = torch.where(targets > 0.5, 1 - self.alpha, self.alpha) # Inverse alpha logic? Check Focal Loss paper
        # F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss # pt already weighted

        # Let's stick with Option 1 for clarity
        return F_loss.mean()


class StegoDataset(Dataset):
    def __init__(self, original_dir: str, original_jpg_dir: str, stego_dirs: dict, transform=None, split='train', train_ratio=0.7):
        self.transform = transform
        self.original_dir = original_dir
        self.original_jpg_dir = original_jpg_dir
        self.stego_dirs = stego_dirs

        # Collect clean images
        self.original_files = []
        clean_count = 0
        for dir_path in [original_dir, original_jpg_dir]:
             if os.path.exists(dir_path):
                 files = sorted(
                     [f for f in os.listdir(dir_path) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))],
                     key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x
                 )
                 self.original_files.extend([os.path.join(dir_path, f) for f in files])
                 clean_count += len(files)
                 print(f"Found {len(files)} clean images in {dir_path}")
             else:
                print(f"Warning: Directory {dir_path} does not exist")

        if not self.original_files:
            raise ValueError("No clean images found in specified directories")
        print(f"Total clean images found: {clean_count}")


        # Collect stego images
        self.stego_files = {}
        stego_count = 0
        for tool, stego_dir in stego_dirs.items():
            ext = '.jpg' if "outguess" in tool else '.bmp'
            if not os.path.exists(stego_dir):
                print(f"Warning: Stego directory {stego_dir} for tool {tool} does not exist")
                self.stego_files[tool] = []
                continue # Skip if dir not found
                # raise ValueError(f"Stego directory {stego_dir} does not exist")
            stego_files_in_dir = sorted(
                [f for f in os.listdir(stego_dir) if f.lower().endswith(ext)],
                key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x
            )
            self.stego_files[tool] = [os.path.join(stego_dir, f) for f in stego_files_in_dir]
            print(f"{tool}: {len(stego_files_in_dir)} stego images found in {stego_dir}")
            stego_count += len(stego_files_in_dir)
        print(f"Total stego images found: {stego_count}")

        # Build dataset
        self.data = []
        clean_base_names = set()
        # Add clean files first
        for orig_path in self.original_files:
            base_name_match = re.search(r'image(\d+)', os.path.basename(orig_path))
            if not base_name_match:
                print(f"Warning: Could not extract base number from clean filename {os.path.basename(orig_path)}")
                base_name = os.path.splitext(os.path.basename(orig_path))[0] # Fallback
            else:
                base_name = base_name_match.group(1)

            clean_base_names.add(base_name)
            self.data.append({"path": orig_path, "label": 0.0, "tool": "clean", "base_name": base_name})

        # Add stego files, trying to match with clean base names
        for tool, stego_files_list in self.stego_files.items():
            for stego_path in stego_files_list:
                base_name = None
                # Try to extract base number matching the clean image format
                match = re.search(r'image(\d+)', os.path.basename(stego_path))
                if match:
                    base_name = match.group(1)
                else:
                    print(f"Warning: Could not extract base number from stego filename {os.path.basename(stego_path)} for tool {tool}")
                    base_name = os.path.splitext(os.path.basename(stego_path))[0].replace("_" + tool,"").replace(".jpg","") # Fallback guess

                # Check if a corresponding clean image exists (optional strict matching)
                # if base_name in clean_base_names:
                self.data.append({"path": stego_path, "label": 1.0, "tool": tool, "base_name": base_name})
                # else:
                #     print(f"Warning: No clean match found for stego image base name '{base_name}' from {os.path.basename(stego_path)} ({tool}). Skipping.")


        if not self.data:
            raise ValueError("Dataset is empty after processing files.")

        # Log dataset composition before split
        tool_counts = {"clean": 0}
        for tool in stego_dirs.keys():
             tool_counts[tool] = 0
        for item in self.data:
             tool_counts[item["tool"]] = tool_counts.get(item["tool"], 0) + 1
        print("Initial dataset composition:", tool_counts)


        # Stratified train/val split based on labels (0 or 1)
        from sklearn.model_selection import train_test_split
        labels = [item["label"] for item in self.data]
        indices = list(range(len(self.data)))

        try:
            train_indices, val_indices = train_test_split(
                indices,
                train_size=train_ratio,
                stratify=labels, # Stratify by clean/stego label
                random_state=42,
                shuffle=True
            )
        except ValueError as e:
             print(f"Warning: Stratified split failed ({e}). Performing non-stratified split.")
             train_indices, val_indices = train_test_split(
                 indices, train_size=train_ratio, random_state=42, shuffle=True
             )


        # Assign indices based on split
        target_indices = train_indices if split == 'train' else val_indices
        self.split_data = [self.data[i] for i in target_indices]

        # --- Oversample clean images ONLY in the training set ---
        if split == 'train':
            oversample_factor = 3 # Increase this factor if needed
            clean_items_in_train = [item for item in self.split_data if item["label"] == 0.0]
            if clean_items_in_train:
                 original_train_size = len(self.split_data)
                 num_clean_to_add = len(clean_items_in_train) * (oversample_factor - 1)
                 self.split_data.extend(np.random.choice(clean_items_in_train, num_clean_to_add))
                 print(f"Training set: Oversampled clean images. Added {num_clean_to_add} clean samples. New size: {len(self.split_data)} (was {original_train_size})")
            else:
                print("Warning: No clean images found in the training split for oversampling.")
        # --- End Oversampling ---

        # Final check and log for the current split
        final_tool_counts = {"clean": 0}
        for tool in stego_dirs.keys():
             final_tool_counts[tool] = 0
        for item in self.split_data:
             final_tool_counts[item["tool"]] = final_tool_counts.get(item["tool"], 0) + 1
        print(f"Final {split} set composition: {final_tool_counts}")
        print(f"Dataset split '{split}': {len(self.split_data)} samples")


    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        item = self.split_data[idx]
        img_path = item["path"]
        label = item["label"]
        tool = item["tool"]
        try:
            # Ensure image is loaded correctly, handle potential errors
            with Image.open(img_path) as img:
                 image = img.convert('RGB')
            # image = Image.open(img_path).convert('RGB') # Original
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32), tool
        except Exception as e:
            print(f"Error loading image {img_path} (tool: {tool}, label: {label}): {e}. Returning blank image.")
            # Return a blank image of the correct size
            blank = torch.zeros((3, 256, 256), dtype=torch.float32) # Create tensor directly
            # blank = Image.new('RGB', (256, 256), color='black')
            # if self.transform:
            #     blank = self.transform(blank) # Transform might fail on blank PIL
            return blank, torch.tensor(0.0, dtype=torch.float32), "failed"


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=100, save_path=None):
    best_val_f1 = 0.0 # Track best F1 score for saving model
    patience = 15 # Increased patience slightly
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None

    # --- Find optimal threshold on initial validation data before training ---
    print("Calculating initial optimal threshold on validation data...")
    initial_metrics = evaluate_model(model, val_loader, criterion) # Pass criterion for loss calc
    optimal_threshold = initial_metrics['optimal_threshold']
    print(f"Initial optimal threshold: {optimal_threshold:.4f}")
    # ------------------------------------------------------------------------

    for epoch in range(num_epochs):
        # No linear warmup, use scheduler from start or after few epochs
        # if epoch < 5:  # 5-epoch warmup
        #     lr = 1e-6 + (optimizer.defaults['lr'] - 1e-6) * epoch / 5
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # else:
        current_lr = optimizer.param_groups[0]['lr']


        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_preds_at_opt, train_labels = [], []

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # Mixed precision training
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16, enabled=scaler is not None):
                # Pass is_clean flag if you still want debug prints in the model
                outputs = model(inputs) #, is_clean=(targets == 0))
                loss = criterion(outputs, targets)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Adjusted max_norm
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() # Already mean loss from criterion
            probs = torch.sigmoid(outputs.detach()) # Use detach for metrics calculation
            # Use the pre-calculated optimal threshold for training accuracy monitoring
            preds = (probs > optimal_threshold).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

            train_preds_at_opt.extend(preds.cpu().numpy())
            train_labels.extend(targets.cpu().numpy())

            # Optional: Print sample probs/labels less frequently
            # if batch_idx == 0 and epoch % 5 == 0:
            #     print(f"Epoch {epoch + 1}, Sample Probs: {probs[:4].cpu().detach().numpy().flatten()}")
            #     print(f"Epoch {epoch + 1}, Sample Labels: {targets[:4].cpu().detach().numpy()}")

        # Average loss over batches
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f"Epoch {epoch + 1}/{num_epochs} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f} | Train Acc (thr={optimal_threshold:.2f}): {train_acc:.4f}")
        # Show train confusion matrix less often if needed
        if (epoch + 1) % 5 == 0:
             cm_train = confusion_matrix(train_labels, train_preds_at_opt, labels=[0, 1])
             print(f"Train Confusion Matrix (thr={optimal_threshold:.2f}):\n{cm_train}")

        # Validation step (run every epoch or less frequently)
        if (epoch + 1) % 1 == 0: # Validate every epoch
            val_metrics = evaluate_model(model, val_loader, criterion)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            optimal_threshold = val_metrics['optimal_threshold'] # Update optimal threshold

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            print(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {val_loss:.4f} | Val Acc (thr={optimal_threshold:.2f}): {val_acc:.4f} | Val F1: {val_f1:.4f} | Optimal Thr: {optimal_threshold:.4f}")
            print(f"Val Confusion Matrix (thr={optimal_threshold:.2f}):\n{val_metrics['confusion_matrix']}")

            # --- Plot validation confusion matrix ---
            plt.figure(figsize=(6, 4))
            cm_val = val_metrics['confusion_matrix']
            plt.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Epoch {epoch + 1} Val CM (thr={optimal_threshold:.2f})")
            plt.colorbar()
            plt.xticks([0, 1], ['Clean', 'Stego'])
            plt.yticks([0, 1], ['Clean', 'Stego'])
            thresh = cm_val.max() / 2.
            for i in range(cm_val.shape[0]):
                for j in range(cm_val.shape[1]):
                    plt.text(j, i, format(cm_val[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm_val[i, j] > thresh else "black")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f"confusion_matrix_epoch_{epoch + 1}.png")
            plt.close()
            # --- End Plotting ---

            # --- Save model based on best validation F1 score ---
            if val_f1 > best_val_f1 and save_path:
                print(f"Validation F1 improved from {best_val_f1:.4f} to {val_f1:.4f}. Saving model...")
                best_val_f1 = val_f1
                epochs_no_improve = 0
                # Save model state
                state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                save_file(state_dict, save_path)
                # Save the optimal threshold from this epoch
                with open("optimal_threshold.pkl", "wb") as f:
                    pickle.dump(optimal_threshold, f)
                print(f"Saved best model to {save_path} and threshold {optimal_threshold:.4f} to optimal_threshold.pkl")

            else:
                epochs_no_improve += 1
                print(f"Validation F1 did not improve for {epochs_no_improve} epoch(s). Best F1: {best_val_f1:.4f}")

            # --- Early stopping check ---
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation F1 score.")
                break

        # Step the scheduler
        scheduler.step()


    # --- Plot Training History ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label=f'Train Acc (at final thr={optimal_threshold:.2f})')
    plt.plot(history['val_acc'], label='Val Acc (at optimal thr)')
    plt.plot(history['val_f1'], label='Val F1 (at optimal thr)')
    plt.title('Accuracy & F1 History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()
    # --- End Plotting ---

    print(f"Training finished. Best validation F1 score: {best_val_f1:.4f}")
    return history, best_val_f1 # Return history and best F1


def find_optimal_threshold(targets, scores):
    """Finds the optimal threshold to maximize F1 score."""
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(targets, (scores >= t).astype(int)) for t in thresholds]
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx]
    return optimal_threshold, f1_scores[best_f1_idx]

    # Alternative: Maximize Youden's J statistic (Sensitivity + Specificity - 1)
    # fpr, tpr, thresholds = roc_curve(targets, scores)
    # youden_j = tpr - fpr
    # best_j_idx = np.argmax(youden_j)
    # optimal_threshold_j = thresholds[best_j_idx]
    # return optimal_threshold_j, youden_j[best_j_idx]

def evaluate_model(model, test_loader, criterion=None):
    """Evaluates the model and finds the optimal threshold."""
    model.eval()
    all_targets, all_scores, all_tools = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, tools in test_loader:
            inputs = inputs.to(DEVICE)
            targets_dev = targets.to(DEVICE) # Targets on device for loss calculation

            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16, enabled=DEVICE=='cuda'):
                outputs = model(inputs)
                # Calculate loss if criterion is provided
                if criterion:
                    loss = criterion(outputs, targets_dev)
                    total_loss += loss.item() # Already mean loss

            scores = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_targets.extend(targets.numpy().flatten())
            all_scores.extend(scores)
            all_tools.extend(tools)

            # Optional: Print eval sample probs less frequently
            # if len(all_targets) <= test_loader.batch_size * 2:
            #     print(f"Eval Sample Probs: {scores[:4]}")
            #     print(f"Eval Sample Labels: {targets.numpy()[:4]}")

    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    # --- Find Optimal Threshold ---
    optimal_threshold, max_f1 = find_optimal_threshold(all_targets, all_scores)
    print(f"Optimal threshold found: {optimal_threshold:.4f} (maximizes F1 score: {max_f1:.4f})")
    # --- End Finding Optimal Threshold ---

    # Calculate metrics using the optimal threshold
    all_preds_opt = (all_scores >= optimal_threshold).astype(int)

    accuracy = (all_preds_opt == all_targets).mean()
    cm = confusion_matrix(all_targets, all_preds_opt, labels=[0, 1])

    # Ensure cm has shape (2, 2) even if one class is missing in predictions/targets for a tool
    if cm.shape == (1, 1):
        # Infer the missing class based on the present label
        present_label = np.unique(np.concatenate((all_targets, all_preds_opt)))[0]
        if present_label == 0: # Only negatives predicted/true
            cm = np.array([[cm[0,0], 0], [0, 0]])
        else: # Only positives predicted/true
            cm = np.array([[0, 0], [0, cm[0,0]]])
    elif cm.shape != (2, 2):
         # Handle unexpected shapes if necessary, though less likely with labels=[0,1]
         print(f"Warning: Unexpected confusion matrix shape {cm.shape}. Setting default.")
         cm = np.array([[0, 0], [0, 0]]) # Default safe matrix

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # TPR, Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # F1 score calculated using sklearn's function on predictions at optimal threshold
    f1 = f1_score(all_targets, all_preds_opt)


    # Plot probability histogram
    clean_scores = all_scores[all_targets == 0]
    stego_scores = all_scores[all_targets == 1]
    plt.figure(figsize=(8, 6))
    if len(clean_scores) > 0:
         plt.hist(clean_scores, bins=50, alpha=0.6, label=f'Clean (N={len(clean_scores)})', color='blue', density=True)
    if len(stego_scores) > 0:
        plt.hist(stego_scores, bins=50, alpha=0.6, label=f'Stego (N={len(stego_scores)})', color='red', density=True)
    plt.axvline(optimal_threshold, color='k', linestyle='dashed', linewidth=1, label=f'Optimal Thr={optimal_threshold:.2f}')
    plt.title('Validation Probability Distribution')
    plt.xlabel('Predicted Probability (Stego)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('probability_histogram.png')
    plt.close()

    tool_metrics = {}
    unique_tools = sorted(list(set(t for t in all_tools if t != "failed")))
    for tool in unique_tools:
        tool_indices = [i for i, t in enumerate(all_tools) if t == tool]
        if not tool_indices:
            continue

        tool_targets = all_targets[tool_indices]
        tool_scores = all_scores[tool_indices]
        tool_preds_opt = (tool_scores >= optimal_threshold).astype(int)

        # Ensure labels=[0, 1] for consistency, even if only one class present for a tool
        tool_cm = confusion_matrix(tool_targets, tool_preds_opt, labels=[0, 1])

        # Handle potential 1x1 matrix if only one class exists for the tool
        if tool_cm.shape == (1, 1):
             present_label = np.unique(tool_targets)[0]
             if present_label == 0:
                 tool_cm = np.array([[tool_cm[0,0], 0], [0, 0]])
             else:
                 tool_cm = np.array([[0, 0], [0, tool_cm[0,0]]])
        elif tool_cm.shape != (2,2):
             print(f"Warning: Unexpected confusion matrix shape {tool_cm.shape} for tool {tool}. Setting default.")
             tool_cm = np.array([[0, 0], [0, 0]])


        tn_t, fp_t, fn_t, tp_t = tool_cm.ravel()

        tool_metrics[tool] = {
            'accuracy': (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t) if (tp_t + tn_t + fp_t + fn_t) > 0 else 0,
            'sensitivity': tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0,
            'specificity': tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0,
            'precision': tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0,
            'f1': f1_score(tool_targets, tool_preds_opt), # Use sklearn's F1
            'confusion_matrix': tool_cm,
            'count': len(tool_indices)
        }

    metrics = {
        'loss': total_loss / len(test_loader) if criterion and len(test_loader) > 0 else float('nan'),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1, # Overall F1 at optimal threshold
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold,
        'tool_metrics': tool_metrics,
        # Return raw scores and targets if needed elsewhere
        # 'predictions_opt': all_preds_opt,
        # 'targets': all_targets,
        # 'scores': all_scores,
        # 'tools': all_tools
    }

    return metrics


def calibrate_model(model, val_loader):
    """Calibrates the model using Logistic Regression (Sigmoid scaling)."""
    print("Calibrating model...")
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(DEVICE)
            # Use float32 for calibration input features
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32, enabled=False):
                 outputs = model(inputs).cpu().numpy().flatten()
            features.extend(outputs)
            labels.extend(targets.cpu().numpy().flatten())

    features = np.array(features).reshape(-1, 1)
    labels = np.array(labels)

    # Use Logistic Regression for calibration (similar to Platt scaling)
    # Consider adjusting C (inverse of regularization strength)
    lr = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear') # Added balanced weights
    # Fit directly using LogisticRegression which gives probabilities
    lr.fit(features, labels)
    print("Calibration complete.")
    # Return the fitted Logistic Regression model itself as the calibrator
    return lr


def predict_with_calibration(model, calibrator, image_tensor):
    """Makes predictions using the model and the calibrator."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        # Ensure model runs in eval mode without unnecessary grads etc.
        # Use float32 for input to calibrator model
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32, enabled=False):
             raw_output = model(image_tensor).cpu().numpy().flatten()[0]

    # Use the fitted Logistic Regression calibrator to get calibrated probability
    # The calibrator's predict_proba gives [[prob_class_0, prob_class_1]]
    calibrated_prob = calibrator.predict_proba(np.array([[raw_output]]))[0, 1]

    return raw_output, calibrated_prob


def load_model(model_path):
    model = StegoCNN().to(DEVICE)
    print(f"Loading model state dict from: {model_path}")
    try:
        state_dict = load_file(model_path, device='cpu') # Load to CPU first
        model.load_state_dict(state_dict)
        model.to(DEVICE) # Move model to target device
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise # Re-raise exception


def predict_stego(image_path, model, calibrator, optimal_threshold, transform_val):
    """Predicts if an image is stego using the model, calibrator, and optimal threshold."""
    try:
        # Ensure image is loaded correctly
        with Image.open(image_path) as img:
             image = img.convert('RGB')
        image_tensor = transform_val(image) # Apply validation transforms
        raw_score, calibrated_prob = predict_with_calibration(model, calibrator, image_tensor)

        # --- Use optimal threshold for decision ---
        is_stego = calibrated_prob > optimal_threshold
        # --- End Change ---

        # Identify tool only if predicted as stego
        stego_tool = "None"
        if is_stego:
             tool_identifier = ToolIdentifier(image_path)
             stego_tool = tool_identifier.identify()


        result = {
            'is_stego': bool(is_stego),
            'raw_score': float(raw_score),
            'calibrated_probability': float(calibrated_prob),
            'stego_tool': stego_tool,
            'threshold_used': float(optimal_threshold)
        }
        return result
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # Optionally, provide more details for debugging
        # import traceback
        # traceback.print_exc()
        return None


def main():
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")


    # Define transforms (consider adjusting augmentations)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)), # Resize first
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.3), # Less vertical flip?
        transforms.RandomRotation(10), # Reduced rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Mild color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Keep normalization consistent
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # --- Define Directories ---
    stego_dirs = {
        "openstego": "ml/stego/",
        "steghide": "ml/stego_steghide/",
        "outguess": "ml/stego_outguess/"
    }
    model_save_path = "stego_model_v2.safetensors"
    calibrator_save_path = "calibrator_v2.pkl"
    threshold_save_path = "optimal_threshold_v2.pkl"

    # --- Create Datasets and DataLoaders ---
    print("\n--- Creating Training Dataset ---")
    train_dataset = StegoDataset(
        original_dir=ORIGINAL_DIR,
        original_jpg_dir=ORIGINAL_JPG_DIR,
        stego_dirs=stego_dirs,
        transform=transform_train,
        split='train',
        train_ratio=0.8 # Use 80% for training
    )
    print("\n--- Creating Validation Dataset ---")
    val_dataset = StegoDataset(
        original_dir=ORIGINAL_DIR,
        original_jpg_dir=ORIGINAL_JPG_DIR,
        stego_dirs=stego_dirs,
        transform=transform_val,
        split='val',
        train_ratio=0.8 # Must match train_ratio for correct split
    )

    # Check dataset sizes
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
         print("Error: One of the datasets is empty. Check file paths and dataset creation logic.")
         return

    train_loader = DataLoader(
        train_dataset,
        batch_size=32, # Adjust batch size based on GPU memory
        shuffle=True,
        num_workers=4, # Adjust based on CPU cores
        pin_memory=True,
        drop_last=True # Drop last incomplete batch for consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32, # Match train batch size or use larger for validation
        shuffle=False, # No need to shuffle validation data
        num_workers=4,
        pin_memory=True
    )

    # --- Initialize Model, Criterion, Optimizer, Scheduler ---
    print("\n--- Initializing Model and Training Components ---")
    model = StegoCNN().to(DEVICE)
    # --- Optional: Load weights from previous run if needed ---
    # if os.path.exists(model_save_path):
    #     print(f"Loading existing model weights from {model_save_path}")
    #     model = load_model(model_save_path)

    criterion = FocalLoss(alpha=0.5, gamma=2.0) # Keep alpha/gamma, rely on class weights
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2) # AdamW often works well, slightly higher LR
    # Scheduler: Reduce LR on plateau based on validation F1 score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)

    # --- Train the Model ---
    print("\n--- Starting Model Training ---")
    history, best_metric = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler, # Pass scheduler here (or step it manually in loop)
        num_epochs=50, # Reduced epochs for example, adjust as needed
        save_path=model_save_path
    )
    print("\n--- Training Complete ---")


    # --- Load Best Model for Final Evaluation and Calibration ---
    print(f"\n--- Loading Best Model from {model_save_path} ---")
    try:
         model = load_model(model_save_path)
    except FileNotFoundError:
         print(f"Error: Best model file {model_save_path} not found. Cannot proceed with final evaluation.")
         return
    except Exception as e:
         print(f"Error loading best model: {e}")
         return


    # --- Final Evaluation on Validation Set ---
    print("\n--- Final Evaluation on Validation Set using Best Model ---")
    final_metrics = evaluate_model(model, val_loader, criterion) # Use the same val_loader

    print(f"\nFinal Validation Metrics (using threshold={final_metrics['optimal_threshold']:.4f}):")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Sensitivity (TPR): {final_metrics['sensitivity']:.4f}")
    print(f"  Specificity (TNR): {final_metrics['specificity']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  F1 Score: {final_metrics['f1']:.4f}")
    print("  Confusion Matrix:")
    print(final_metrics['confusion_matrix'])
    print("\n  Tool-specific metrics (Validation Set):")
    for tool, tool_metrics_val in final_metrics['tool_metrics'].items():
        print(f"\n  {tool} (N={tool_metrics_val['count']}):")
        for metric_name, metric_value in tool_metrics_val.items():
            if metric_name == 'confusion_matrix':
                print(f"    {metric_name}:\n{metric_value}")
            elif metric_name != 'count':
                print(f"    {metric_name}:{metric_value:.4f}")

    # Use the optimal threshold found during the final evaluation
    optimal_threshold = final_metrics['optimal_threshold']
    # Save this definitive threshold
    with open(threshold_save_path, "wb") as f:
        pickle.dump(optimal_threshold, f)
        print(f"Saved final optimal threshold {optimal_threshold:.4f} to {threshold_save_path}")


    # --- Calibrate the Best Model ---
    calibrator = calibrate_model(model, val_loader) # Calibrate on the validation set
    with open(calibrator_save_path, "wb") as f:
        pickle.dump(calibrator, f)
        print(f"Saved calibrator to {calibrator_save_path}")


    # --- Test Predictions on Sample Images ---
    print("\n--- Testing Predictions on Sample Images ---")
    # Load necessary components if not already in memory
    try:
        if 'model' not in locals(): model = load_model(model_save_path)
        if 'calibrator' not in locals():
            with open(calibrator_save_path, "rb") as f: calibrator = pickle.load(f)
        if 'optimal_threshold' not in locals():
             with open(threshold_save_path, "rb") as f: optimal_threshold = pickle.load(f)
        print(f"Using loaded threshold for prediction: {optimal_threshold:.4f}")
    except FileNotFoundError as e:
         print(f"Error loading components for prediction: {e}. Cannot proceed.")
         return
    except Exception as e:
         print(f"Error loading components: {e}")
         return


    clean_phash_map = build_clean_phash_map(ORIGINAL_DIR, ORIGINAL_JPG_DIR)
    image_paths = [
        # Add paths to your specific test images here
        "test_openstego.bmp",
        "test_steghide.bmp",
        "test_outguess.jpg",
        "test_unknown.bmp", # Example clean image
        # Add more test files as needed
        "ml/clean/image1.bmp", # Example from dataset (should be clean)
        "ml/stego/image1.jpg.bmp", # Example from dataset (should be stego)
    ]

    for path in image_paths:
        if not os.path.exists(path):
            print(f"\n{path}: File not found.")
            continue

        print(f"\nProcessing: {path}")
        prediction_result = predict_stego(
             image_path=path,
             model=model,
             calibrator=calibrator,
             optimal_threshold=optimal_threshold,
             transform_val=transform_val # Pass the correct transform
         )

        if prediction_result:
            print(f"  Prediction:")
            print(f"    Is Stego: {prediction_result['is_stego']} (Prob: {prediction_result['calibrated_probability']:.4f} vs Thr: {prediction_result['threshold_used']:.4f})")
            # print(f"    Raw Score: {prediction_result['raw_score']:.4f}") # Less important now
            print(f"    Identified Tool (if stego): {prediction_result['stego_tool']}")

            # Find closest clean match (optional)
            stego_phash = phash_image(path)
            if stego_phash and clean_phash_map:
                closest_clean_phash, closest_clean_path = min(clean_phash_map.items(), key=lambda kv: abs(stego_phash - kv[0]))
                phash_dist = abs(stego_phash - closest_clean_phash)
                # Adjust threshold for "match" based on testing
                if phash_dist <= 4:
                    print(f"    Closest Clean Match (pHash dist={phash_dist}): {closest_clean_path}")
                else:
                    print(f"    No close clean match found (min pHash dist={phash_dist})")
            elif not clean_phash_map:
                 print("    Clean phash map is empty, cannot find match.")
            else:
                 print("    Could not compute phash for the image.")
        else:
             print("  Prediction failed.")


    print("\n--- Script Finished ---")
    # Return final metrics for potential further use
    return final_metrics


if __name__ == "__main__":
    final_validation_metrics = main()
    if final_validation_metrics:
         print("\nFinal script run metrics summary (on validation set):")
         print(f"  Optimal Threshold: {final_validation_metrics['optimal_threshold']:.4f}")
         print(f"  F1 Score: {final_validation_metrics['f1']:.4f}")
         print(f"  Accuracy: {final_validation_metrics['accuracy']:.4f}")
         print(f"  Specificity: {final_validation_metrics['specificity']:.4f}")
         print(f"  Sensitivity: {final_validation_metrics['sensitivity']:.4f}")
