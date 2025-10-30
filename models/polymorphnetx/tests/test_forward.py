
import torch
from model.model import PolymorphNetX
from model.config import CONF

def test_forward_shape():
    m = PolymorphNetX()
    t = torch.randint(0, CONF.vocab_size, (1, 32))
    y = m(t)
    assert y.shape == (1, 32, CONF.vocab_size)
