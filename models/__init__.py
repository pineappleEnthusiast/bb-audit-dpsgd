from .lr import LR
from .cnn import CNN
from .wideresnet import WideResNet
from .lstm import LSTM

Models = {
    'lr': LR,
    'cnn': CNN,
    'wideresnet': WideResNet,
    'lstm': LSTM
}
