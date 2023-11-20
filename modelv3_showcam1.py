import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model_v3 import mobilenet_v3_large
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import cv2

img_path = "C:\\Users\\yuhao\\Desktop\\test\\Normal-1.png"
model_weight_path = "./MobileNetV3_1115.pth"
def main():

    model = mobilenet_v3_large(num_classes=4)
    state_dict = torch.load(model_weight_path)
    model.load_state_dict(state_dict)


    target_layers = [model.features[-1]]



    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5094], [0.2314])])
    # load image

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 2 #interested classilfy label
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()