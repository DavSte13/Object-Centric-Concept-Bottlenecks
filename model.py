import torch
import torch.nn as nn

# Random seed for reproducibility
torch.manual_seed(0)


class LinearPredictor(nn.Module):
    def __init__(self, num_classes, explicit_encoding_dim, device="cuda"):
        super(LinearPredictor, self).__init__()
        self.classifier = nn.Linear(explicit_encoding_dim, num_classes)
        self.device = device

    def forward(self, encodings):
        # Pass the encodings through the linear classifier
        outputs = self.classifier(encodings)
        return outputs


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=10):
        super(MLP, self).__init__()
        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPPredictor(nn.Module):
    def __init__(self, num_classes, implicit_encoding_dim, device="cuda"):
        super(MLPPredictor, self).__init__()
        # initialize the residual module, that creates continuous concept embedding
        self.MLP = MLP(
            input_dim=implicit_encoding_dim,
            hidden_dims=[512, 256],
            output_dim=num_classes,
        )
        self.device = device

    def forward(self, clip_encs):
        # Pass implicit clip encodings to residual module
        y = self.MLP(clip_encs)
        return y
