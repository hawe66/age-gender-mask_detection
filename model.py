import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class BaseModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes, hidden_layer = 128, freeze = False, pretrained_model = resnet50):
        super(MyModel, self).__init__()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # pretrained with age-estimation
        self.model = pretrained_model(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 101, bias=False)
        model_path = "/opt/ml/pretrained/age_pretrained.pth"
        self.model.load_state_dict(torch.load(model_path))
        # freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        # add hidden layer @ classifier
        self.num_classes = num_classes
        self.hidden_layer = hidden_layer
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, self.hidden_layer),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(self.hidden_layer, num_classes),
            )
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
