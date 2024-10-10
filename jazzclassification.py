# This python file contains the code to set up the Jazz Classification model. 
# It was developed with assitance from ChatGTP and my thesis supervisor.
# The main classification model is from Pytorch. 
# https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html


import torch
import torch.nn as nn
import torch.nn.functional as F

# Labels for jazz artists
labels = ('BennyCarter', 'BennyGoodman', 'CharlieParker', 'ChetBaker', 'CurtisFuller', 
          'DexterGordon', 'DizzyGillespie', 'FreddieHubbard', 'HankMobley', 'HerbieHancock',
          'JoeHenderson', 'JohnColtrane', 'KaiWinding', 'KidOry', 'LouisArmstrong',
          'MilesDavis', 'SonnyRollins', 'WayneShorter', 'WoodyShaw')

def label_to_index(word):
    return torch.tensor(labels.index(word))

def index_to_label(index):
    return labels[index]

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=19, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        
        x = F.avg_pool1d(x, x.shape[-1])  # Global average pooling
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)  # Output probabilities

# Load the model
model = M5(n_input=1, n_output=len(labels))  # Adjust n_input if needed
model.load_state_dict(torch.load(r'jazz_classification_model.pth'))
model.eval()
device = torch.device('CUDA')
model.to(device)

def predict(waveform):
    waveform = waveform.to(device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(waveform)
    
    # Get the probabilities using exp() to undo log_softmax
    probabilities = output.exp().squeeze(0)
    
    # Get the top 2 predicted indices and their probabilities
    top_probs, top_indices = torch.topk(probabilities, 2, dim=1)

    # Convert indices to labels
    top_labels_probs = [(index_to_label(idx.item()), prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])]

    print(top_labels_probs)
    return top_labels_probs
