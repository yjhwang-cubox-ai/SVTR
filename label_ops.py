import copy
import numpy as np

class BaseRecLabelEncode:
    """ Convert between text-label and text-index """
    
    def __init__(self,
                 max_text_length=25,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False):
    
        self.max_text_length = max_text_length
        self.beg_str = 'sos'
        self.end_str = 'eos'
        self.lower = lower
        
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        self.idx2char = {y: x for x, y in self.dict.items()}
    
    def add_special_char(self, dict_character):
        return dict_character
    
    def encode(self, text):
        if len(text) == 0 or len(text) > self.max_text_length:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list

    def decode(self, arr):
        if len(arr) == 0 or max(arr) > len(self.character):
            return None
        text_list = []
        for idx in arr:
            if idx not in self.idx2char:
                continue
            text_list.append(self.idx2char[idx])
        if len(text_list) == 0:
            return None
        print(text_list)
        return "".join(text_list)

class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_length - len(text))
        data['label'] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character