import torch


class MNIST_Classifier(torch.nn.Module):
    def __init__(self, backbone, mlp, temperature=1.0):
        super(MNIST_Classifier, self).__init__()

        self.backbone = backbone
        self.mlp = mlp
        self.temperature = temperature
        self.center = 0.0

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)
        x = self.mlp(x)

        x = x - self.center

        return torch.nn.functional.softmax(x / self.temperature, dim=1)
