import torch
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmengine.registry import DefaultScope

cfg = Config.fromfile("configs/clrernet/culane/clrernet_culane_dla34_ema.py")

# Register CLRerNet custom modules
if hasattr(cfg, "custom_imports"):
    import_modules_from_strings(**cfg.custom_imports)

# Set default scope to mmdet so MODELS registry resolves correctly
DefaultScope.get_instance("test", scope_name="mmdet")

# Build model
model = MODELS.build(cfg.model)
model.eval()

# Load weights
load_checkpoint(model, "clrernet_culane_dla34_ema.pth", map_location="cpu")
model = model.cuda()

dummy_input = torch.randn(1, 3, 320, 800).cuda()

torch.onnx.export(
    model,
    dummy_input,
    "clrernet.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=[
        "logits_0",
        "anchors_0",
        "lengths_0",
        "xs_0",
        "logits_1",
        "anchors_1",
        "lengths_1",
        "xs_1",
        "logits_2",
        "anchors_2",
        "lengths_2",
        "xs_2",
    ],
    verbose=False,
)

print("ONNX export done!")
