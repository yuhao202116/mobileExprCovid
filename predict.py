import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v3 import mobilenet_v3_large


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(299),
         transforms.CenterCrop(299),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize([0.5094, ], [0.2314, ])])

    # load image
    img_path = "D:\\deep_learning\\Mobilenet\\data_set\\lung_data\\lung_photos\\Lung_Opacity\\Lung_Opacity-1.png"

    assert os.path.exists("output.png"), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('L')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # model = mobilenet_v3_large(num_classes=4).to(device)
    # load model weights
    model_weight_path = "./jit_model.pt"
    # model.load_state_dict(torch.jit.load(model_weight_path, map_location=device))
    model = torch.load(model_weight_path, map_location=device)
    # target_layers = [model.features[-1]]
    target_category = 1
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
