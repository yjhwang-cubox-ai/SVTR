import torch

class CTCLabelConverter():
    def __init__(self, character):
        # dict_character = list(character)
        
        self.dict = {}
        for i, char in enumerate(character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTC loss
            self.dict[char] = i + 1
        
        self.character = ['[CTCblank]'] + character # dummy '[CTCblank]' token for CTCLoss (index 0)
        
    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in each batch. 25 by default
        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        
        return (batch_text, torch.IntTensor(length))
    
    def decode(self, text_index, Length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(Length):
            t = text_index[index, :]
            
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            
            texts.append(text)
        return texts