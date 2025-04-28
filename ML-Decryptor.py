"""
What is correct and tested:

1. Random Number Generator
2. Correct Bit Packing: The test [1, 0, 1, 0, 0, 0, 0, 1] -> 0xa1 confirms that np.packbits is correctly converting bits
3. Valid Stego Image

Issue:
The Decryptor class fails to find the "OPENSTEGO" magic number in the extracted bits, despite the CNN model
correctly identifying the image as stego (probability 1.0).

The most likely causes are:

    Incorrect Seed: The script assumes SEED = 0, but OpenStego might use a different default or derived seed.
    Position Selection Mismatch: The order or method of selecting (x, y, channel) coordinates may differ from
    OpenStego’s RandomLSB.

    Header Format: The assumed header structure might not match your OpenStego version’s format.

"""


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
from torch.cuda.amp import GradScaler, autocast


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {DEVICE}")

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
        if (n & -n) == n:  # Power of 2
            return (n * self.next(31)) >> 31
        bits = self.next(31)
        val = bits % n
        while bits - val + (n - 1) < 0:
            bits = self.next(31)
            val = bits % n
        return val


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
    def __init__(self, original_dir: str, stego_dir: str, transform=None, split='train', train_ratio=0.8):
        self.transform = transform
        self.original_files = sorted([f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.bmp'))])
        self.stego_files = sorted([f for f in os.listdir(stego_dir) if f.endswith(('.jpg', '.bmp'))])

        assert len(self.original_files) == len(
            self.stego_files), f"Mismatched image counts: {len(self.original_files)} vs {len(self.stego_files)}"
        for orig, stego in zip(self.original_files, self.stego_files):
            orig_base = orig.split('.')[0]
            stego_base = stego.split('.')[0]
            assert orig_base == stego_base, f"Mismatched base filenames: {orig} vs {stego}"

        self.data = []
        for orig, stego in zip(self.original_files, self.stego_files):
            self.data.append((os.path.join(original_dir, orig), 0.0))
            self.data.append((os.path.join(stego_dir, stego), 1.0))

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
        img_path, label = self.data[actual_idx]
        clean_idx = actual_idx ^ 1
        clean_img = Image.open(self.data[clean_idx][0]).convert('RGB') if self.data[clean_idx][1] == 0.0 else Image.open(self.data[actual_idx][0]).convert('RGB')
        stego_img = Image.open(img_path).convert('RGB') if label == 1.0 else clean_img

        clean_np = np.array(clean_img, dtype=np.float32) / 255.0
        stego_np = np.array(stego_img, dtype=np.float32) / 255.0

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
        self.scaler = GradScaler()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ])

    def train(self, original_dir: str, stego_dir: str, epochs: int = 30, batch_size: int = 2, accum_steps: int = 4):
        global batch_idx
        train_dataset = StegoDataset(original_dir, stego_dir, self.transform, split='train', train_ratio=0.8)
        val_dataset = StegoDataset(original_dir, stego_dir, self.transform, split='val', train_ratio=0.8)
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

                with autocast(device_type=DEVICE):
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
                    with autocast(device_type=DEVICE):
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

class Decryptor:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ])

    def extract_bits_sequentially(self, stego_np, num_bits):
        height, width, channels = stego_np.shape
        bits = []
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if len(bits) >= num_bits:
                        return bits
                    bits.append(stego_np[y, x, c] & 1)
        return bits

    def extract_bits_randomly(self, stego_np, seed: int, num_bits: int) -> list:
        height, width, channels = stego_np.shape
        rand = JavaRandom(seed)
        extracted_bits = []

        # Alternative position calculation that might match OpenStego
        while len(extracted_bits) < num_bits:
            y = rand.nextInt(height)
            x = rand.nextInt(width)
            c = rand.nextInt(channels)
            extracted_bits.append(stego_np[y, x, c] & 1)

        return extracted_bits


        # Alternative 2: Flattened array approach
        # total_pixels = height * width * channels
        # while len(extracted_bits) < num_bits:
        #     idx = rand.nextInt(total_pixels)
        #     c = idx % 3
        #     idx //= 3
        #     x = idx % width
        #     y = idx // width
        #     key = (x, y, c)
        #     if key in used_positions:
        #         continue
        #     used_positions.add(key)
        #     extracted_bits.append(stego_np[y, x, c] & 1)



    def bits_to_bytes(self, bits: list) -> bytes:
        return np.packbits(bits, axis=0).tobytes()

    def predict(self, stego_path):
        stego_img = Image.open(stego_path).convert('RGB')
        img = self.transform(stego_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img).squeeze()
            prob = torch.sigmoid(output).item()

        return prob, "Stego" if prob >= 0.5 else "Clean"


    def decrypt(self, stego_path):
        global extracted_bits, seed, actual_pos, extracted_bytes
        prob, label = self.predict(stego_path)
        print(f"Predicted: {label} with probability {prob:.4f}")

        if label != "Stego":
            print("No steganographic content detected.")
            return b''

        print("Starting OpenStego extraction...")
        stego_img = Image.open(stego_path).convert('RGB')
        stego_np = np.array(stego_img, dtype=np.uint8)
        magic = b'OPENSTEGO'

        # Expanded seed list including image-derived seeds
        height, width, _ = stego_np.shape
        seeds_to_try = [
            0, 1, 42, 12345, 1000, 9999,
            314159, 271828, 1337, 6969,
            hash((height, width)) & 0xFFFFFFFF,
            hash(tuple(stego_np.flatten()[:100])) & 0xFFFFFFFF  # Seed from image content
        ]

        # Remove duplicates while preserving order
        seen = set()
        seeds_to_try = [x for x in seeds_to_try if not (x in seen or seen.add(x))]

        found = False
        for seed in seeds_to_try:
            print(f"\nTrying seed {seed}...")

            # Extract enough bits for header (magic + reasonable header size)
            extracted_bits = self.extract_bits_randomly(stego_np, seed, len(magic) * 8 + 1024 * 8)
            extracted_bytes = self.bits_to_bytes(extracted_bits)

            # Debug output
            print(f"First 20 bytes (hex): {extracted_bytes[:20].hex()}")
            print(f"First 20 bytes (ASCII): {extracted_bytes[:20]}")

            # Check all possible byte alignments for magic number
            for i in range(8):
                shifted_bytes = extracted_bytes[i:]
                idx = shifted_bytes.find(magic)
                if idx != -1:
                    actual_pos = idx + i
                    print(f"Magic number found at byte {actual_pos} with alignment offset {i}")
                    found = True
                    break

            if found:
                break

        if not found:
            print("\nMagic number 'OPENSTEGO' not found in any tested seeds.")
            print("Possible causes:")
            print("- Different seed generation algorithm used by OpenStego")
            print("- Different header format (magic string might be different)")
            print("- The steganography method might not be OpenStego's RandomLSB")
            return b''

        byte_pos = actual_pos + len(magic)
        try:
            version = extracted_bytes[byte_pos]
            byte_pos += 1
            data_length = int.from_bytes(extracted_bytes[byte_pos:byte_pos + 4], byteorder='little')
            byte_pos += 4
            channel_bits_used = extracted_bytes[byte_pos]
            byte_pos += 1
            filename_length = extracted_bytes[byte_pos]
            byte_pos += 1
            compression_flag = extracted_bytes[byte_pos]
            byte_pos += 1
            encryption_flag = extracted_bytes[byte_pos]
            byte_pos += 1
            encryption_algo = extracted_bytes[byte_pos:byte_pos + 8].decode('utf-8', errors='ignore').strip()
            byte_pos += 8
            filename = extracted_bytes[byte_pos:byte_pos + filename_length].decode('utf-8', errors='ignore').strip()
            byte_pos += filename_length
            print(f"Parsed header: version={version}, filename='{filename}', data_length={data_length}")


            total_data_bytes = byte_pos + data_length
            total_bits_needed = (byte_pos + data_length) * 8
            if len(extracted_bits) < total_bits_needed:
                extracted_bits = self.extract_bits_randomly(stego_np, seed, total_bits_needed)
                extracted_bytes = self.bits_to_bytes(extracted_bits)

            content_bytes = extracted_bytes[byte_pos:byte_pos + data_length]

            if encryption_flag:
                print("Warning: decryption needed but not implemented.")
                return content_bytes

            if compression_flag:
                try:
                    import gzip
                    import io
                    with gzip.GzipFile(fileobj=io.BytesIO(content_bytes)) as gz:
                        content_bytes = gz.read()
                    print("Decompressed successfully.")
                except Exception as e:
                    print(f"Decompression failed: {e}")

            output_dir = "extracted_files"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename or "extracted_message.txt")
            with open(output_path, 'wb') as f:
                f.write(content_bytes)
            print(f"Extracted file saved to {output_path}")

            return content_bytes

        except Exception as e:
            print(f"Error during parsing: {e}")
            return b''


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

        trainer.train(
            original_dir='ml/clean',  # original images
            stego_dir='ml/stego',  # encrypted images
            epochs=30,
            batch_size=2,
            accum_steps=4
        )

        trainer.save_model("model.safetensors")

    decryptor = Decryptor(model, device=DEVICE)

    # Decrypt the stego image
    stego_path = "test_stego_new.bmp"
    extracted_data = decryptor.decrypt(stego_path)

    # Save or process the extracted data
    with open("extracted_message.bin", "wb") as f:
        f.write(extracted_data)

    print("Decryption complete. Extracted data saved to extracted_message.bin")
