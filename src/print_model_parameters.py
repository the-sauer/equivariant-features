from aef import models

for name, model, param in [
    ("logpolar+maxpool", models.HardNetLogPolar, {"in_channels": 1, "head": "maxpool", "learned_mask": False}),
    ("logpolar+fft", models.HardNetLogPolar, {"in_channels": 1, "head": "fft", "learned_mask": False}),
    ("effient8", models.BlobDescriptorEfficient, {"n_rotations": 8, "in_channels": 1, "head": "attention", "learned_mask": False}),
    ("effient4", models.BlobDescriptorEfficient, {"n_rotations": 4, "in_channels": 1, "head": "attention", "learned_mask": False}),
]:
    model = model(**param)
    model.train()
    print(f"{name}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")  
