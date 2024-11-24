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

# Define a simpler CNN model with configurable number of layers
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, hidden_size=512, num_layers=5, dilation_rate=1):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dilation_rate = dilation_rate

        # First layer
        self.conv1 = nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # Intermediate layers with residual connections and dilated convolutions
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dilation_rates = [dilation_rate ** i for i in range(num_layers - 2)]
        for dilation in self.dilation_rates:
            self.convs.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=dilation, dilation=dilation))
            self.bns.append(nn.BatchNorm1d(hidden_size))

        # Last layer
        self.conv_last = nn.Conv1d(hidden_size, in_channels, kernel_size=3, padding=1)

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

# Custom Dataset class with normalization
class MUSDBDataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=264600):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
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

        # Normalize to zero mean and unit variance
        mixture = (mixture - mixture.mean(dim=1, keepdim=True)) / (mixture.std(dim=1, keepdim=True) + 1e-8)
        vocals = (vocals - vocals.mean(dim=1, keepdim=True)) / (vocals.std(dim=1, keepdim=True) + 1e-8)

        # Define a scaling factor to reduce the volume (e.g., 0.5 for half the volume)
        scaling_factor = 0.05

        # Apply the scaling factor to make the audio quieter
        mixture_quiet = mixture * scaling_factor
        vocals_quiet = vocals * scaling_factor

        # Optionally segment the signals
        if self.segment_length:
            if mixture.shape[1] >= self.segment_length:
                start = torch.randint(0, mixture.shape[1] - self.segment_length, (1,))
                mixture = mixture[:, start:start+self.segment_length]
                vocals = vocals[:, start:start+self.segment_length]
            else:
                mixture = F.pad(mixture, (0, self.segment_length - mixture.shape[1]))
                vocals = F.pad(vocals, (0, self.segment_length - vocals.shape[1]))

        return mixture, vocals

# Training function with loss logging
def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps,
          val_dataloader=None, checkpoint_path=None):
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
        for mixture, vocals in dataloader:
            mixture = mixture.to(device)
            vocals = vocals.to(device)

            optimizer.zero_grad()
            vocals_pred = model(mixture)
            loss = loss_fn(vocals_pred, vocals)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step the scheduler

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            loss_log.append(loss.item())
            step += 1
            progress_bar.update(1)
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f}"

            if step % checkpoint_steps == 0:
                if val_dataloader:
                    avg_val_loss, avg_sdr_vocals, avg_sdr_instrumentals = validate(model, val_dataloader, loss_fn, device)
                    desc += f" - Val Loss: {avg_val_loss:.4f}, SDR Vocals: {avg_sdr_vocals:.4f}, SDR Instrumentals: {avg_sdr_instrumentals:.4f}"
                else:
                    avg_val_loss, avg_sdr_vocals, avg_sdr_instrumentals = None, None, None
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'sdr_vocals': avg_sdr_vocals,
                    'sdr_instrumentals': avg_sdr_instrumentals,
                    'loss_log': loss_log
                }, f"checkpoint_step_{step}.pt")
            progress_bar.set_description(desc)

    # Save final loss log
    torch.save({'loss_log': loss_log}, 'loss_log.pt')
    progress_bar.close()

def validate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    sdr_vocals = []
    sdr_instrumentals = []

    with torch.no_grad():
        # Create a tqdm progress bar
        pbar = tqdm(total=len(dataloader), desc="Validation Progress")

        for batch_idx, (mixture, vocals) in enumerate(dataloader):
            mixture = mixture.to(device)
            vocals = vocals.to(device)
            vocals_pred = model(mixture)
            loss = loss_fn(vocals_pred, vocals)
            val_loss += loss.item()

            # Convert tensors to numpy arrays
            mixture_np = mixture.cpu().numpy()
            vocals_np = vocals.cpu().numpy()
            vocals_pred_np = vocals_pred.cpu().numpy()

            # Compute instrumentals
            instrumentals_true_np = mixture_np - vocals_np
            instrumentals_pred_np = mixture_np - vocals_pred_np

            # Compute SDR for each sample in the batch
            batch_sdr_vocals = []
            batch_sdr_instrumentals = []
            for j in range(mixture_np.shape[0]):
                # For sample j
                ref_vocals = vocals_np[j]
                ref_instrumentals = instrumentals_true_np[j]
                est_vocals = vocals_pred_np[j]
                est_instrumentals = instrumentals_pred_np[j]

                # Detect and ignore silent areas
                ref_vocals_mask = np.abs(ref_vocals).mean(axis=0) > 1e-5
                ref_instrumentals_mask = np.abs(ref_instrumentals).mean(axis=0) > 1e-5
                est_vocals_mask = np.abs(est_vocals).mean(axis=0) > 1e-5
                est_instrumentals_mask = np.abs(est_instrumentals).mean(axis=0) > 1e-5

                # Apply masks to ignore silent areas
                ref_vocals = ref_vocals[:, ref_vocals_mask]
                ref_instrumentals = ref_instrumentals[:, ref_instrumentals_mask]
                est_vocals = est_vocals[:, est_vocals_mask]
                est_instrumentals = est_instrumentals[:, est_instrumentals_mask]

                # Ensure all arrays have the same number of dimensions
                ref_vocals = ref_vocals[np.newaxis, :, :] if ref_vocals.ndim == 1 else ref_vocals
                ref_instrumentals = ref_instrumentals[np.newaxis, :, :] if ref_instrumentals.ndim == 1 else ref_instrumentals
                est_vocals = est_vocals[np.newaxis, :, :] if est_vocals.ndim == 1 else est_vocals
                est_instrumentals = est_instrumentals[np.newaxis, :, :] if est_instrumentals.ndim == 1 else est_instrumentals

                # Stack references and estimates
                ref = np.vstack([ref_vocals, ref_instrumentals])
                est = np.vstack([est_vocals, est_instrumentals])

                # Compute BSS eval metrics
                sdr, _, _, _ = separation.bss_eval_sources(ref, est)

                # Collect SDR for vocals and instrumentals
                batch_sdr_vocals.append(sdr[0])
                batch_sdr_instrumentals.append(sdr[1])

            # Update the progress bar with the current batch SDRs
            avg_batch_sdr_vocals = sum(batch_sdr_vocals) / len(batch_sdr_vocals)
            avg_batch_sdr_instrumentals = sum(batch_sdr_instrumentals) / len(batch_sdr_instrumentals)
            pbar.set_postfix({
                "SDR (Vocals)": f"{avg_batch_sdr_vocals:.4f}",
                "SDR (Instrumentals)": f"{avg_batch_sdr_instrumentals:.4f}"
            })
            pbar.update(1)

            # Append batch SDRs to the overall SDR lists
            sdr_vocals.extend(batch_sdr_vocals)
            sdr_instrumentals.extend(batch_sdr_instrumentals)

        pbar.close()

    model.train()
    avg_val_loss = val_loss / len(dataloader)
    avg_sdr_vocals = sum(sdr_vocals) / len(sdr_vocals)
    avg_sdr_instrumentals = sum(sdr_instrumentals) / len(sdr_instrumentals)
    print(f"Validation - Avg Loss: {avg_val_loss:.4f}")
    print(f"Avg SDR (Vocals): {avg_sdr_vocals:.4f}")
    print(f"Avg SDR (Instrumentals): {avg_sdr_instrumentals:.4f}")
    return avg_val_loss, avg_sdr_vocals, avg_sdr_instrumentals

def inference(model, checkpoint_path, input_wav_path, output_instrumentals_path,
              chunk_size=16384, overlap=4096, device='cpu'):
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

    # Normalize input audio
    input_audio_mean = input_audio.mean(dim=1, keepdim=True)
    input_audio_std = input_audio.std(dim=1, keepdim=True)
    input_audio_normalized = (input_audio - input_audio_mean) / (input_audio_std + 1e-8)

    # Initialize variables for chunk processing
    total_length = input_audio_normalized.shape[1]
    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    instrumentals = torch.zeros_like(input_audio_normalized)

    # Define cross-fade length
    cross_fade_length = overlap // 2

    # Process audio in chunks with a sliding window and cross-fade
    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            chunk = input_audio_normalized[:, i:i + chunk_size]
            chunk = chunk.unsqueeze(0)  # Add batch dimension

            # Inference
            with torch.no_grad():
                vocals_pred = model(chunk)
                inst_chunk = chunk - vocals_pred

            # Remove batch dimension
            inst_chunk = inst_chunk.squeeze(0)

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

    # Denormalize the output
    instrumentals = instrumentals * input_audio_std + input_audio_mean

    # Apply clipping to avoid extreme values
    instrumentals = torch.clamp(instrumentals, -1.0, 1.0)

    # Save the separated instrumentals
    torchaudio.save(output_instrumentals_path, instrumentals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for music voice separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='path/to/musdb18', help='Path to MUSDB18 training dataset')
    parser.add_argument('--val_dir', type=str, default=None, help='Path to MUSDB18 validation dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--checkpoint_steps', type=int, default=1000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--segment_length', type=int, default=264600, help='Segment length for training')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in the CNN model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, optimizer, and loss function
    model = SimpleCNN(in_channels=2, hidden_size=512, num_layers=args.num_layers)
    optimizer = Prodigy(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    loss_fn = nn.MSELoss()

    if args.train:
        # Create training dataset and dataloader
        train_dataset = MUSDBDataset(root_dir=args.data_dir, segment_length=args.segment_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Create validation dataset and dataloader if provided
        val_dataloader = None
        if args.val_dir:
            val_dataset = MUSDBDataset(root_dir=args.val_dir, segment_length=args.segment_length)
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)  # Always batch size of 1

        # Initialize scheduler
        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # Start training
        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs,
              args.checkpoint_steps, val_dataloader=val_dataloader, checkpoint_path=args.checkpoint_path)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        # Ensure the model architecture matches the one used during training
        model = SimpleCNN(in_channels=2, hidden_size=512, num_layers=args.num_layers)
        # Run inference
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, device=device)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()
