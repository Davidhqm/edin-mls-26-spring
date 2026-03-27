"""
Audio Feature Extraction (Whisper-style Mel Spectrogram)
Educational implementation from scratch using PyTorch only
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from config import AudioProcessorConfig


def create_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> torch.Tensor:
    """
    Create a Mel filterbank matrix.

    This converts FFT frequency bins to Mel scale, which better matches
    human perception of pitch.
    """
    if f_max is None:
        f_max = sample_rate / 2

    # Mel scale conversion functions
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Create mel points equally spaced in mel scale
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bin numbers
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return torch.from_numpy(filterbank).float()


class WhisperFeatureExtractor:
    """
    Extracts log-mel spectrogram features from audio waveforms.

    This matches the Whisper feature extraction used by GLM-ASR.
    """

    def __init__(self, config: AudioProcessorConfig = None):
        if config is None:
            config = AudioProcessorConfig()

        self.sampling_rate = config.sampling_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_mels = config.feature_size
        self.chunk_length = config.chunk_length
        self.n_samples = config.n_samples
        self.nb_max_frames = config.nb_max_frames
        self.padding_value = config.padding_value

        # Create mel filterbank
        self.mel_filters = create_mel_filterbank(
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            sample_rate=self.sampling_rate
        )

        # Create Hann window for STFT
        self.window = torch.hann_window(self.n_fft)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform.

        Args:
            waveform: Audio tensor of shape (num_samples,)

        Returns:
            Complex STFT tensor of shape (n_freqs, num_frames)
        """
        # Ensure waveform is 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Pad to center frames
        # Note: F.pad with mode='reflect' requires at least 2D tensor
        # So we add a batch dimension, pad, then squeeze
        pad_amount = self.n_fft // 2
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        waveform = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')
        waveform = waveform.squeeze(0).squeeze(0)  # (samples + 2*pad,)

        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.device),
            center=False,
            return_complex=True
        )

        return stft

    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram from waveform.

        Args:
            waveform: Audio tensor of shape (num_samples,)

        Returns:
            Log-mel spectrogram of shape (n_mels, num_frames)
        """
        # Compute STFT magnitude
        stft = self._stft(waveform)
        magnitudes = torch.abs(stft) ** 2

        # Apply mel filterbank
        mel_filters = self.mel_filters.to(magnitudes.device)
        mel_spec = torch.matmul(mel_filters, magnitudes)

        # Convert to log scale (with small epsilon for numerical stability)
        log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))

        # Normalize to match Whisper's preprocessing
        log_mel_spec = torch.clamp(log_mel_spec, min=log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0

        return log_mel_spec

    def __call__(
        self,
        raw_audio: Union[np.ndarray, torch.Tensor],
        sampling_rate: int = None,
        padding: str = "max_length",
        max_length: int = None,
        return_tensors: str = "pt"
    ) -> dict:
        """
        Extract features from raw audio.

        Args:
            raw_audio: Raw audio waveform (numpy array or tensor)
            sampling_rate: Sample rate of input audio (should be 16000)
            padding: Padding strategy ("max_length" or "do_not_pad")
            max_length: Maximum number of frames
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Dictionary with 'input_features' key containing mel spectrogram
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Audio must be {self.sampling_rate}Hz, got {sampling_rate}Hz. "
                "Please resample the audio first."
            )

        # Convert to tensor if needed
        if isinstance(raw_audio, np.ndarray):
            waveform = torch.from_numpy(raw_audio).float()
        else:
            waveform = raw_audio.float()

        # Compute mel spectrogram
        mel_spec = self._compute_mel_spectrogram(waveform)

        # Determine max length
        if max_length is None:
            max_length = self.nb_max_frames

        # Pad or truncate to max_length
        num_frames = mel_spec.shape[1]

        if padding == "max_length":
            if num_frames < max_length:
                # Pad with padding_value
                pad_size = max_length - num_frames
                mel_spec = F.pad(mel_spec, (0, pad_size), value=self.padding_value)
            elif num_frames > max_length:
                # Truncate
                mel_spec = mel_spec[:, :max_length]

        # Add batch dimension
        mel_spec = mel_spec.unsqueeze(0)  # (1, n_mels, num_frames)

        # Transpose to (batch, num_frames, n_mels) as expected by encoder
        mel_spec = mel_spec.transpose(1, 2)

        return {
            "input_features": mel_spec,
            "attention_mask": torch.ones(1, mel_spec.shape[1], dtype=torch.long)
        }


def load_audio_file(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    This is a simple implementation using scipy.
    For production, consider using librosa or torchaudio.
    """
    try:
        from scipy.io import wavfile
        from scipy import signal

        sr, audio = wavfile.read(file_path)

        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.float64:
            audio = audio.astype(np.float32)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)

        return audio

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")


if __name__ == "__main__":
    # Test the feature extractor
    config = AudioProcessorConfig()
    extractor = WhisperFeatureExtractor(config)

    # Create a test signal (1 second of 440Hz sine wave)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    features = extractor(test_audio, sampling_rate=sr)
    print(f"Input audio shape: {test_audio.shape}")
    print(f"Output features shape: {features['input_features'].shape}")
    print(f"Expected: (1, num_frames, 128)")
