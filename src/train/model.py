import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=(1, 2), dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return self.dropout(x)

class DrumCRNN(nn.Module):
    """
    40GB VRAM에 최적화된 CRNN 모델
    Input: [Batch, 1, 229 (Mels), Time]
    Output: [Batch, Time, 9 (Classes)]
    """
    def __init__(self, num_classes=9, num_mels=229):
        super().__init__()
        
        # Feature Extractor (3-Layer CNN)
        # Input: [B, 1, 229, T]
        self.conv1 = ConvBlock(1, 32, pool_size=(2, 1)) # [B, 32, 114, T]
        self.conv2 = ConvBlock(32, 64, pool_size=(2, 1)) # [B, 64, 57, T]
        self.conv3 = ConvBlock(64, 128, pool_size=(2, 1)) # [B, 128, 28, T]
        
        # Calculate resulting feature size along frequency axis
        self.freq_dim = 128 * (num_mels // 8) # 128 * 28 = 3584
        
        # Temporal Context (2-Layer BiGRU)
        self.gru = nn.GRU(
            input_size=self.freq_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Multi-label Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: [Batch, 1, Mels, Time]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # [Batch, Channels, Freqs, Time] -> [Batch, Time, Channels * Freqs]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        
        # GRU
        x, _ = self.gru(x)
        
        # Classifier
        logits = self.classifier(x) # [Batch, Time, num_classes]
        
        # BCELossWithLogits를 사용할 것이므로 sigmoid는 생략
        return logits
