import torch
import torch.nn as nn


# Random seed for reproducibility
torch.manual_seed(0)

# Define the linear model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    

class ConceptBottleneckModelWithEncs(nn.Module):
    def __init__(self, num_classes, explicit_encoding_dim, device='cuda'):
        super(ConceptBottleneckModelWithEncs, self).__init__()
        self.classifier = LinearClassifier(explicit_encoding_dim, num_classes)
        self.device = device

    def forward(self, encodings):
        # Pass the encodings through the linear classifier
        outputs = self.classifier(encodings)
        return outputs
