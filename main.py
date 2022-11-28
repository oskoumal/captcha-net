import torch

from training.captcha_coder import CaptchaCoder, Coding
from training.trainer import train, test


def test_captcha_coder():
    coder = CaptchaCoder(Coding.full_one_hot)
    print(coder._encode_full_one_hot_vector("1"))
    print(coder._encode_full_one_hot_vector("2"))
    print(coder._encode_full_one_hot_vector("3"))

    print(torch.zeros(33 * 5).dtype)

    print(coder.encode("12ack"))
    print(coder.decode(coder.encode("12ack")))


# test_captcha_coder()

#train()

test()
