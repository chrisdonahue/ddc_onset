from typing import Optional
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import WEIGHTS_DIR

AUDIO_NUM_CHANNELS = 1
AUDIO_FS = 44100
FEATS_HOP = 441
FEATS_FS = 100

_FFT_FRAME_LENGTHS = [1024, 2048, 4096]
_NUM_MEL_BANDS = 80
_LOG_EPS = 1e-16


class SpectrogramExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        for i in _FFT_FRAME_LENGTHS:
            window = nn.Parameter(
                torch.zeros(i, dtype=torch.float32, requires_grad=False),
                requires_grad=False,
            )
            setattr(self, "window_{}".format(i), window)
            mel = nn.Parameter(
                torch.zeros(
                    ((i // 2) + 1, 80), dtype=torch.float32, requires_grad=False
                ),
                requires_grad=False,
            )
            setattr(self, "mel_{}".format(i), mel)
        self.load_state_dict(
            torch.load(pathlib.Path(WEIGHTS_DIR, "spectrogram_extractor.bin"))
        )

    def forward(
        self, x: torch.Tensor, frame_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Extracts a log mel spectrogram from a waveform.

        Args:
            x: 44.1kHz waveforms as float32 [batch_size, num_samples].
            frame_chunk_size: Number of frames to process at a time. If None, process all frames at once.
        Returns:
            Log mel spectrograms as float32 [batch_size, num_frames, num_mel_bands, num_fft_frame_lengths].
        """
        # NOTE: This was originally implemented as [samps, batch] but [batch, samps] is better type signature.
        waveform = x.transpose(1, 0)
        feats = []
        for i, fft_frame_length in enumerate(_FFT_FRAME_LENGTHS):
            # Pad waveform to center spectrogram
            fft_frame_length_half = fft_frame_length // 2
            waveform_padded = F.pad(waveform, (0, 0, fft_frame_length_half, 0))
            waveform_padded_len = waveform_padded.shape[0]

            # Params for this FFT size
            window = getattr(self, "window_{}".format(fft_frame_length)).view(1, -1, 1)
            mel_w = getattr(self, "mel_{}".format(fft_frame_length))

            # Chunk up waveform to save memory at cost of some efficiency
            if frame_chunk_size is None:
                chunk_hop = waveform_padded_len
            else:
                chunk_hop = frame_chunk_size * FEATS_HOP

            chunk_feats = []
            for c in range(0, waveform_padded_len, chunk_hop):
                frames = []
                for s in range(c, min(c + chunk_hop, waveform_padded_len), FEATS_HOP):
                    # Slice waveform into frames
                    # TODO: Change this to range(0, waveform.shape[0], FEATS_HOP) to make num feats = ceil(num_samps / 441)? Make sure frames are equal after doing this
                    frame = waveform_padded[s : s + fft_frame_length]
                    padding_amt = fft_frame_length - frame.shape[0]
                    if padding_amt > 0:
                        frame = F.pad(frame, (0, 0, padding_amt, 0))
                    frames.append(frame)
                frames = torch.stack(frames, dim=0)

                # Apply window
                frames *= window

                # Copying questionable "zero phase" windowing nonsense from essentia
                # https://github.com/MTG/essentia/blob/master/src/algorithms/standard/windowing.cpp#L85
                frames_half_one = frames[:, :fft_frame_length_half]
                frames_half_two = frames[:, fft_frame_length_half:]
                frames = torch.cat([frames_half_two, frames_half_one], dim=1)

                # Perform FFT
                frames = torch.transpose(frames, 1, 2)
                spec = torch.fft.rfft(frames)

                # Compute power spectrogram
                spec_r = torch.real(spec)
                spec_i = torch.imag(spec)
                pow_spec = torch.pow(spec_r, 2) + torch.pow(spec_i, 2)

                # Compute mel spectrogram
                mel_spec = torch.matmul(pow_spec, mel_w)

                # Compute log mel spectrogram
                log_mel_spec = torch.log(mel_spec + _LOG_EPS)
                log_mel_spec = torch.transpose(log_mel_spec, 1, 2)
                chunk_feats.append(log_mel_spec)

            chunk_feats = torch.cat(chunk_feats, dim=0)
            # Cut off extra chunk_feats for larger FFT lengths
            # TODO: Don't slice these chunk_feats to begin with
            if i == 0:
                feats_num_timesteps = chunk_feats.shape[0]
            else:
                assert chunk_feats.shape[0] > feats_num_timesteps
                chunk_feats = chunk_feats[:feats_num_timesteps]

            feats.append(chunk_feats)

        feats = torch.stack(feats, dim=2)

        # [feats..., batch] -> [batch, feats...]
        return feats.permute(3, 0, 1, 2)
