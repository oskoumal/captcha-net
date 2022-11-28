import numpy as np
import torch
from enum import Enum


class Coding(Enum):
    simple_single_char = "simple single char"
    full_one_hot = 'full one hot'


class CaptchaCoder:
    """
        Class takes care of coding of CAPTCHA characters to numeric tensors.
    """
    def __init__(self, coding):
        # Alphabet for testing
        self.sample_alphabet = ['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm',
                                'n', 'p', 'w', 'x', 'y']
        # Problem defined alphabet
        self.alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e',
                         'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u',
                         'v', 'w', 'x', 'y', 'z']
        self.alphabet_size = len(self.alphabet)
        # Number of characters in one image
        self.captcha_dimension = 5
        # Encoding of CAPTCHA string to tensor
        self.captcha_encoder = {}
        # Decoding of output tensor to CAPTCHA string
        self.captcha_decoder = {}
        # Enum of type of coding used by the class
        self.coding = coding
        if coding == Coding.simple_single_char:
            self._compute_simple_single_char_coding()

    def _compute_simple_single_char_coding(self):
        for index, class_char in enumerate(self.alphabet):
            self.captcha_encoder[class_char] = index
            self.captcha_decoder[index] = class_char
        print(self.captcha_encoder)

    def encode(self, captcha_string):
        if self.coding == Coding.simple_single_char:
            return int(self.captcha_encoder[captcha_string])
        elif self.coding == Coding.full_one_hot:
            composed_vector = torch.empty(0)
            for char in captcha_string:
                composed_vector = torch.cat((composed_vector, self._encode_full_one_hot_vector(char)))
            return composed_vector

    def _encode_full_one_hot_vector(self, char):
        vector = np.zeros(self.alphabet_size, dtype=np.float32)
        vector[self.alphabet.index(char)] = 1
        return torch.from_numpy(vector)

    def decode(self, label):
        captcha_string = ""
        label = torch.squeeze(label)
        composed_vector = torch.split(label, self.alphabet_size)
        for vector in composed_vector:
            vector = vector.numpy()
            captcha_string += self.alphabet[np.where(vector == 1)[0][0]]
        return captcha_string

    def decode_raw_output(self, output):
        captcha_string = ""
        output = torch.squeeze(output)
        composed_output = torch.split(output, self.alphabet_size)

        for one_hot_vec in composed_output:
            _, out_index = torch.max(one_hot_vec, 0)
            captcha_string += self.alphabet[out_index.item()]

        return captcha_string
