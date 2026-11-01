import torch.nn as nn


def conv_block(cin, cout):
    """Conv - BN - ReLU (x2) - MaxPool: giảm kích thước không gian đi 2 lần."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """CNN tự thiết kế, huấn luyện từ đầu. 224 -> 112 -> 56 -> 28 -> 14."""

    def __init__(self, num_classes, channels=(32, 64, 128, 256), dropout=0.4):
        super().__init__()
        layers, cin = [], 3
        for c in channels:
            layers.append(conv_block(cin, c))
            cin = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cin, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
