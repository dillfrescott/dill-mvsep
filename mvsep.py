import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prodigyopt import Prodigy
from mir_eval import separation
import numpy as np

# Define a simpler CNN model with configurable number of layers for spectrograms
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, hidden_size=512, num_layers=5, dilation_rate=1):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dilation_rate = dilation_rate

        # First layer
        self.conv1 = nn.Conv2d(in_channels, hidden_size, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(hidden_size)

        # Intermediate layers with residual connections and dilated convolutions
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dilation_rates = [dilation_rate ** i for i in range(num_layers - 2)]
        for dilation in self.dilation_rates:
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), padding=(dilation, dilation), dilation=(dilation, dilation)))
            self.bns.append(nn.BatchNorm2d(hidden_size))

        # Last layer
        self.conv_last = nn.Conv2d(hidden_size, in_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        # First layer
        x = F.relu(self.bn1(self.conv1(x)))

        # Intermediate layers with residual connections and dilated convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = x
            x = F.relu(bn(conv(x)))
            x = x + residual  # Residual connection

        # Last layer
        x = self.conv_last(x)

        return x

# Custom Dataset class with normalization and spectrogram conversion
class MUSDBDataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=264600, n_fft=2048, hop_length=512, segment=True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment = segment
        self.tracks = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]
        mixture, _ = torchaudio.load(os.path.join(track_path, 'mixture.wav'))
        vocals, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))

        # Ensure both signals have the same number of channels
        if mixture.shape[0] != 2 or vocals.shape[0] != 2:
            raise ValueError("Audio files must have 2 channels.")

        # Ensure both signals have the same length
        min_length = min(mixture.shape[1], vocals.shape[1])
        mixture = mixture[:, :min_length]
        vocals = vocals[:, :min_length]

        # Convert to spectrograms
        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=False)
        vocals_spec = torch.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=False)

        # Ensure the last dimension is of size 2 (real and imaginary parts)
        if mixture_spec.shape[-1] != 2 or vocals_spec.shape[-1] != 2:
            raise ValueError("The last dimension of spectrograms must be of size 2 (real and imaginary parts).")

        # Convert to magnitude spectrograms
        mixture_spec = torch.abs(torch.view_as_complex(mixture_spec))
        vocals_spec = torch.abs(torch.view_as_complex(vocals_spec))

        # Normalize to zero mean and unit variance
        mixture_spec = (mixture_spec - mixture_spec.mean(dim=(1, 2), keepdim=True)) / (mixture_spec.std(dim=(1, 2), keepdim=True) + 1e-8)
        vocals_spec = (vocals_spec - vocals_spec.mean(dim=(1, 2), keepdim=True)) / (vocals_spec.std(dim=(1, 2), keepdim=True) + 1e-8)

        # Optionally segment the signals
        if self.segment and self.segment_length:
            if mixture_spec.shape[2] >= self.segment_length // self.hop_length:
                start = torch.randint(0, mixture_spec.shape[2] - self.segment_length // self.hop_length, (1,))
                mixture_spec = mixture_spec[:, :, start:start + self.segment_length // self.hop_length]
                vocals_spec = vocals_spec[:, :, start:start + self.segment_length // self.hop_length]
            else:
                mixture_spec = F.pad(mixture_spec, (0, self.segment_length // self.hop_length - mixture_spec.shape[2]))
                vocals_spec = F.pad(vocals_spec, (0, self.segment_length // self.hop_length - vocals_spec.shape[2]))

        return mixture_spec, vocals_spec, mixture, vocals

# Training function with loss logging
def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps, checkpoint_path=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    loss_log = []
    progress_bar = tqdm(total=epochs * len(dataloader))

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        avg_loss = checkpoint['avg_loss']
        loss_log = checkpoint['loss_log']
        print(f"Resuming training from step {step} with average loss {avg_loss:.4f}")

    model.train()
    for epoch in range(epochs):
        for mixture_spec, vocals_spec, _, _ in dataloader:
            mixture_spec = mixture_spec.to(device)
            vocals_spec = vocals_spec.to(device)

            optimizer.zero_grad()
            vocals_spec_pred = model(mixture_spec)
            loss = loss_fn(vocals_spec_pred, vocals_spec)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step the scheduler

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            loss_log.append(loss.item())
            step += 1
            progress_bar.update(1)
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f}"

            if step % checkpoint_steps == 0:
                # Save checkpoint
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'loss_log': loss_log
                }, f"checkpoint_step_{step}.pt")

            progress_bar.set_description(desc)

    # Save final loss log
    torch.save({'loss_log': loss_log}, 'loss_log.pt')
    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path, output_instrumentals_path,
              chunk_size=16384, overlap=4096, device='cpu', n_fft=2048, hop_length=512):
    # Load model weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # Load input audio
    input_audio, sr = torchaudio.load(input_wav_path)
    if input_audio.shape[0] != 2:
        raise ValueError("Input audio must have 2 channels.")
    input_audio = input_audio.to(device)

    # Initialize variables for chunk processing
    total_length = input_audio.shape[1]
    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    instrumentals = torch.zeros_like(input_audio)

    # Define cross-fade length
    cross_fade_length = overlap // 2

    # Create a Hann window
    window = torch.hann_window(n_fft).to(device)

    # Process audio in chunks with a sliding window and cross-fade
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            chunk = input_audio[:, i:i + chunk_size]

            # Convert chunk to spectrogram
            chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            chunk_mag = torch.abs(chunk_spec)
            chunk_phase = torch.angle(chunk_spec)

            # Normalize chunk magnitude spectrogram
            chunk_mag_mean = chunk_mag.mean(dim=(1, 2), keepdim=True)
            chunk_mag_std = chunk_mag.std(dim=(1, 2), keepdim=True)
            chunk_mag_normalized = (chunk_mag - chunk_mag_mean) / (chunk_mag_std + 1e-8)

            # Add batch dimension
            chunk_mag_normalized = chunk_mag_normalized.unsqueeze(0)

            # Inference
            with torch.no_grad():
                vocals_mag_pred = model(chunk_mag_normalized)
                inst_mag = chunk_mag_normalized - vocals_mag_pred

            # Remove batch dimension
            inst_mag = inst_mag.squeeze(0)

            # Denormalize the output
            inst_mag = inst_mag * chunk_mag_std + chunk_mag_mean

            # Reconstruct the complex spectrogram
            inst_spec = inst_mag * torch.exp(1j * chunk_phase)

            # Convert spectrogram back to waveform
            inst_chunk = torch.istft(inst_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=chunk_size, return_complex=False)

            # Cross-fade the overlapping regions
            if i > 0:
                fade_in = torch.linspace(0, 1, cross_fade_length).to(device)
                fade_out = torch.linspace(1, 0, cross_fade_length).to(device)

                # Apply fade-in to the new chunk
                inst_chunk[:, :cross_fade_length] *= fade_in

                # Apply fade-out to the previous chunk
                instrumentals[:, i:i + cross_fade_length] *= fade_out

                # Add the cross-faded regions
                instrumentals[:, i:i + cross_fade_length] += inst_chunk[:, :cross_fade_length]

            # Place the non-overlapping part of the chunk in the final instrumentals tensor
            instrumentals[:, i + cross_fade_length:i + chunk_size] = inst_chunk[:, cross_fade_length:]

            pbar.update(1)

    # Apply clipping to avoid extreme values
    instrumentals = torch.clamp(instrumentals, -1.0, 1.0)

    # Save the separated instrumentals
    torchaudio.save(output_instrumentals_path, instrumentals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for music voice separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='path/to/musdb18', help='Path to MUSDB18 training dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--checkpoint_steps', type=int, default=1000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--segment_length', type=int, default=264600, help='Segment length for training')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in the CNN model')
    parser.add_argument('--n_fft', type=int, default=2048, help='Number of FFT bins for STFT')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, optimizer, and loss function
    model = SimpleCNN(in_channels=2, hidden_size=256, num_layers=args.num_layers)
    optimizer = Prodigy(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    loss_fn = nn.MSELoss()

    if args.train:
        # Create training dataset and dataloader
        train_dataset = MUSDBDataset(root_dir=args.data_dir, segment_length=args.segment_length, n_fft=args.n_fft, hop_length=args.hop_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Initialize scheduler
        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # Start training
        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs,
              args.checkpoint_steps, checkpoint_path=args.checkpoint_path)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        # Ensure the model architecture matches the one used during training
        model = SimpleCNN(in_channels=2, hidden_size=256, num_layers=args.num_layers)
        # Run inference
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, device=device, n_fft=args.n_fft, hop_length=args.hop_length)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
