import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import save_file, load_file
import numpy as np
import torch.nn.functional as fu
import torch.amp
import re

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {DEVICE}")
ORIGINAL_DIR = "ml/clean/"


class JavaRandom:
    def __init__(self, init_seed):
        self.seed = (init_seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def next(self, bits):
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
        self.stego_image = Image.open(stego_image_path).convert("RGB")
        self.stego_np = np.array(self.stego_image)
        self.magic = b"OPENSTEGO"

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

            if channel_mapping:
                mapped_c = channel_mapping[c]
            else:
                mapped_c = c

            bit = stego_np[y, x, mapped_c] & 1
            extracted_bits.append(int(bit))

            if len(extracted_bits) <= 10:
                r, g, b = stego_np[y, x]
                print(f"DEBUG: Pixel at x={x}, y={y}: R={r}, G={g}, B={b}")
                print(
                    f"DEBUG: Bit {len(extracted_bits) - 1}: x={x}, y={y}, channel={c}, mapped_channel={mapped_c}, bitPos={bit_pos}, value={bit}")

        return extracted_bits

    def identify_tool(self, seed: int = 98234782, bits_to_extract: int = 336) -> str:
        channel_mapping = [2, 1, 0]
        print(f"Checking for OpenStego with seed {seed} and channel mapping {channel_mapping}")

        extracted_bits = self.extract_bits_randomly(self.stego_np, seed, bits_to_extract,
                                                    channel_mapping=channel_mapping)
        print(f"DEBUG: First 100 bits: {extracted_bits[:100]}")
        extracted_bytes = self.bits_to_bytes(extracted_bits)
        print(f"DEBUG: First 50 bytes of header (hex): {extracted_bytes[:50].hex()}")

        idx = extracted_bytes.find(self.magic)
        if idx != -1:
            print(f"Detected OpenStego! Magic number found at byte {idx}")
            return "OpenStego"
        print("OpenStego magic number not found.")
        return None


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
        attention = fu.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        out = self.gamma * out + x
        return out


class StegoCNN(nn.Module):
    def __init__(self):
        super(StegoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
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
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.attn1(x)
        x = self.pool(x)
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.attn2(x)
        x = self.pool(x)
        x = x.view(-1, 256 * 16 * 16)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class StegoDataset(Dataset):
    def __init__(self, original_dir: str, stego_dirs: dict, transform=None, split='train', train_ratio=0.8):
        self.transform = transform
        self.original_files = sorted(
            [f for f in os.listdir(original_dir) if f.endswith('.bmp')],
            key=lambda x: int(re.search(r'\d+', x).group())
        )
        self.stego_dirs = stego_dirs

        if len(self.original_files) != 197:
            raise ValueError(f"Expected 197 original BMP images, found {len(self.original_files)}")

        self.stego_files = {}
        for tool, stego_dir in stego_dirs.items():
            if "outguess" in tool:
                stego_files = sorted(
                    [f for f in os.listdir(stego_dir) if f.endswith('.jpg')],
                    key=lambda x: int(re.search(r'\d+', x).group())
                )
            else:
                stego_files = sorted(
                    [f for f in os.listdir(stego_dir) if f.endswith('.bmp')],
                    key=lambda x: int(re.search(r'\d+', x).group())
                )
            if len(stego_files) != 197:
                raise ValueError(f"Expected 197 stego images in {stego_dir}, found {len(stego_files)}")
            self.stego_files[tool] = stego_files

        for tool, stego_files in self.stego_files.items():
            for orig, stego in zip(self.original_files, stego_files):
                orig_base = orig.split('.')[0]
                if "openstego" in tool:
                    stego_base = stego.split('.')[0]
                else:
                    stego_base = stego.split('_')[0]
                if orig_base != stego_base:
                    raise ValueError(f"Mismatched base filenames: {orig} vs {stego} in {tool}")

        self.data = []
        for orig in self.original_files:
            orig_path = os.path.join(original_dir, orig)
            self.data.append((orig_path, 0.0, "clean"))
            for tool, stego_files in self.stego_files.items():
                stego_base = orig.split('.')[0]
                stego_file = next(f for f in stego_files if f.startswith(stego_base))
                stego_path = os.path.join(stego_dirs[tool], stego_file)
                self.data.append((stego_path, 1.0, tool))

        total = len(self.data)
        train_size = int(total * train_ratio)
        indices = list(range(total))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        self.indices = train_indices if split == 'train' else val_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path, label, tool = self.data[actual_idx]

        base_name = os.path.basename(img_path).split('_')[0] if label == 1.0 else os.path.basename(img_path).split('.')[
            0]
        if "openstego" in tool:
            base_name = os.path.basename(img_path).split('.')[0]
        clean_path = os.path.join(ORIGINAL_DIR, f"{base_name}.bmp")

        clean_img = Image.open(clean_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')

        clean_np = np.array(clean_img, dtype=np.float32) / 255.0
        stego_np = np.array(img, dtype=np.float32) / 255.0

        residuals = np.zeros_like(clean_np)
        for c in range(3):
            residuals[:, :, c] = clean_np[:, :, c] - stego_np[:, :, c]
        residuals = np.clip(residuals * 100.0, -1, 1)
        residuals = (residuals * 255).astype(np.uint8)
        img = Image.fromarray(residuals)

        if self.transform:
            img = self.transform(img)
            if img.shape[0] != 3:
                raise ValueError(f"Image {img_path} has {img.shape[0]} channels, expected 3")
        return img, torch.tensor(label)


class Training:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.scaler = torch.amp.GradScaler('cuda')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ])

    def train(self, original_dir: str, stego_dirs: dict, epochs: int = 30, batch_size: int = 2, accum_steps: int = 4):
        global batch_idx
        train_dataset = StegoDataset(original_dir, stego_dirs, self.transform, split='train', train_ratio=0.8)
        val_dataset = StegoDataset(original_dir, stego_dirs, self.transform, split='val', train_ratio=0.8)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            self.optimizer.zero_grad()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.amp.autocast('cuda'):
                    outputs = self.model(images).squeeze()
                    loss = self.criterion(outputs, labels)
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()

                train_loss += loss.item() * images.size(0) * accum_steps
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

                if (batch_idx + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if batch_idx == 0:
                    print(f"Epoch {epoch + 1}, Sample Probs: {probs[:4].cpu().detach().numpy()}")
                    print(f"Epoch {epoch + 1}, Sample Labels: {labels[:4].cpu().detach().numpy()}")

            if (batch_idx + 1) % accum_steps != 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images).squeeze()
                        loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            self.scheduler.step(val_loss)
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']}")

    def save_model(self, path: str = "model.safetensors"):
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        save_file(state_dict, path)
        print(f"Model saved to {path}")

    def predict(self, clean_path: str, stego_path: str):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ])
        clean_img = Image.open(clean_path).convert('RGB')
        stego_img = Image.open(stego_path).convert('RGB')

        clean_np = np.array(clean_img, dtype=np.float32) / 255.0
        stego_np = np.array(stego_img, dtype=np.float32) / 255.0
        residuals = np.zeros_like(clean_np)
        for c in range(3):
            residuals[:, :, c] = clean_np[:, :, c] - stego_np[:, :, c]
        residuals = np.clip(residuals * 100.0, -1, 1)
        residuals = (residuals * 255).astype(np.uint8)
        img = Image.fromarray(residuals)

        img = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img).squeeze()
            prob = torch.sigmoid(output).item()
        return prob, "Stego" if prob >= 0.5 else "Clean"


if __name__ == '__main__':
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    model = StegoCNN()

    try:
        state_dict = load_file("model.safetensors", device=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully")
    except FileNotFoundError:
        torch.cuda.empty_cache()
        model = StegoCNN()
        print("Starting model training... wait...")
        trainer = Training(model, device=DEVICE)

        stego_dirs = {
            "openstego": "ml/stego/",
            "steghide": "ml/stego_steghide/",
            "outguess": "ml/stego_outguess/"
        }

        trainer.train(
            original_dir='ml/clean/',
            stego_dirs=stego_dirs,
            epochs=30,
            batch_size=2,
            accum_steps=4
        )

        trainer.save_model("model.safetensors")

    stego_path = "test_stego_new.bmp"
    identifier = OpenStegoRandomLSB(stego_path)
    tool = identifier.identify_tool()
    if tool:
        print(f"Identified steganography tool: {tool}")
    else:
        print("Could not identify the steganography tool.")