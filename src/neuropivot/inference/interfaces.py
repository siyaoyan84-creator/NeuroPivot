"""Real EEG encoding and physiological residual conditioning; no AffectGPT import."""
import torch
from neuropivot.models.ordinal import Gate3EEGPivotV3
from neuropivot.models.residual import ResidualAdapter

class EEGEncoder:
    def __init__(self, model, mean, std, dataset="eav"):
        if dataset not in {"eav", "refed"}:
            raise ValueError(dataset)
        self.model = model.eval()
        self.dataset = dataset
        parameter = next(model.parameters())
        self.mean = torch.as_tensor(mean, device=parameter.device, dtype=torch.float32)
        self.std = torch.as_tensor(std, device=parameter.device, dtype=torch.float32)
        channels = 30 if dataset == "eav" else 64
        if self.mean.shape != (1,channels,1) or self.std.shape != self.mean.shape or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Expected finite train-only channel statistics [1,C,1]")

    @classmethod
    def from_checkpoint(cls, path, dataset="eav", device="cpu", normalization=None):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("public_format_version", 1) != 1:
            raise ValueError("Unsupported public checkpoint version")
        fixed_config = {"embedding_dim":128,"shared_dim":64,"dropout":0.25}
        if checkpoint.get("model_config", fixed_config) != fixed_config:
            raise ValueError("Unsupported checkpoint model configuration")
        model = Gate3EEGPivotV3().to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        stats = normalization if normalization is not None else checkpoint.get("normalization")
        if stats is None:
            raise ValueError("Legacy checkpoint has no channel statistics. Supply normalization={'mean': ..., 'std': ...} fitted on its exact training subjects; never fit inference data.")
        if stats.get("dataset", dataset) != dataset:
            raise ValueError("Checkpoint normalization dataset mismatch")
        return cls(model, stats["mean"], stats["std"], dataset)

    @torch.inference_mode()
    def encode_eeg(self, eeg):
        channels, samples = (30,500) if self.dataset == "eav" else (64,200)
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (channels,samples):
            raise ValueError(f"Expected [B,{channels},{samples}] after documented preprocessing")
        x = eeg.to(device=self.mean.device, dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite EEG")
        x = (x - self.mean) / self.std
        output = self.model.encode_eav(x) if self.dataset == "eav" else self.model.encode_refed(x)
        va, latent = output["va_pred"], output["shared"]
        if va.shape[1] != 2 or latent.shape[1] != 64:
            raise ValueError("Unsupported model dimensions")
        return {"va": va, "latent": latent, "pivot": torch.cat([va,latent],dim=1)}

class AVConditioner:
    def __init__(self, adapter, mean, std):
        self.adapter = adapter.eval()
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=next(adapter.parameters()).device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=self.mean.device)
        if self.mean.shape != (66,) or self.std.shape != (66,) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 1e-6).any():
            raise ValueError("Expected final training-only 66D feature statistics")

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["feature_dim"] != 66 or checkpoint["representation"] != "va_pivot" or checkpoint["mode"] != "real":
            raise ValueError("Expected the main 66D aligned physiology checkpoint")
        adapter = ResidualAdapter(66).to(device)
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        return cls(adapter, checkpoint["feature_mean"], checkpoint["feature_std"])

    @torch.inference_mode()
    def condition_av(self, av_embedding, physiology):
        pivot = physiology["pivot"] if isinstance(physiology,dict) else physiology
        if pivot.ndim != 2 or pivot.shape[1] != 66:
            raise ValueError("Expected raw physiology [B,66]")
        if av_embedding.ndim not in {2,3} or av_embedding.shape[-1] != 3584 or av_embedding.shape[0] != pivot.shape[0] or (av_embedding.ndim == 3 and av_embedding.shape[1] != 1):
            raise ValueError("Expected fused AV [B,3584] or [B,1,3584]")
        z = (pivot.to(self.mean.device, dtype=torch.float32) - self.mean) / self.std
        delta = self.adapter(z).to(device=av_embedding.device, dtype=av_embedding.dtype)
        if av_embedding.ndim == 3:
            delta = delta.unsqueeze(1)
        return av_embedding + delta

def encode_eeg(encoder, eeg):
    return encoder.encode_eeg(eeg)

def condition_av(conditioner, av_embedding, physiology):
    return conditioner.condition_av(av_embedding, physiology)
