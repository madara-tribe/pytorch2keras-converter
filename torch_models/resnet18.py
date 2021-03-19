import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision import transforms
from torchsummary import summary
import numpy as np
import cv2

# Define the architecture by modifying resnet.
# Original code is here
# https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
class FullyConvolutionalResnet18(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):

        # Start with standard resnet18 defined here
        # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
        super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L153-L154
        self.avgpool = nn.AvgPool2d((7, 7))

        # Add final Convolution Layer.
        self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    # Reimplementing forward pass.
    # Replacing the following code
    # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L197-L213
    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # Notice, there is no forward pass
        # through the original fully connected layer.
        # Instead, we forward pass through the last conv layer
        x = self.last_conv(x)
        return x

if __name__=='__main__':
    path = '../images/sample.jpg'
    model = FullyConvolutionalResnet18(pretrained=True).eval()
    original_image = cv2.imread(path)
    # Convert original image to RGB format
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
                 transforms.ToTensor(), #Convert image to tensor.
                 transforms.Normalize(
                 mean=[0.485, 0.456, 0.406],   # Subtract mean
                 std=[0.229, 0.224, 0.225]     # Divide by standard deviation
                 )])

    image = transform(image)
    image = image.unsqueeze(0)
    print(image.shape)
    with torch.no_grad():
        preds = model(image)
        preds = torch.softmax(preds, dim=1)
        pred, class_idx = torch.max(preds, dim=1)
        print(class_idx)
        row_max, row_idx = torch.max(pred, dim=1)
        col_max, col_idx = torch.max(row_max, dim=1)
        predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]
        score_map = preds[0, predicted_class, :, :].cpu().numpy()

        # Binarize score map
        #cv2.imwrite('scoremap.png', score_map)
