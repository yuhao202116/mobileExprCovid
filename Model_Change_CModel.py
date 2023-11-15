import torch
from model_v3 import mobilenet_v3_large

ckptfile = "./MobileNetV3.pth"
savedfile = "./human_seg.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mobilenet_v3_large(num_classes=4).to(device)

model.load_state_dict(torch.load(ckptfile, map_location=device))

model.eval()  # 这一步会将参数固化，不能省。否则会报AssertionError('batchnorm with training is not support. Please set model.eval() before export.')
x = torch.rand(1, 1, 299, 299).to(device)
traced_script_module = torch.jit.trace(model, x)


torch.onnx.export(traced_script_module, x, "model.onnx")
torch.jit.save(traced_script_module, "jit_model.pt")
traced_script_module.save("model.pt")
