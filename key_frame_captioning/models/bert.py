from transformers import BertModel
import torch
from logzero import logger

class Bert:
    def __init__(self,config, device):
        self.bert = BertModel.from_pretrained(config["bert_config"]).to(device)
        self.config = config

        if config["update_params"] is False:
            self.freeze_params()

    
    def __call__(self, sentence_ids, sentence_masks):
        out = self.bert(sentence_ids, attention_mask=sentence_masks)
        if self.config["use_feat"] == 'mean':
            masks = sentence_masks.view(*sentence_masks.size(),1)
            out = out[0] * masks
            out = out[:,  1:, :] #CLSを除外
            out = torch.mean(out, -2)
        
        elif self.config["use_feat"] == 'cls':
            out = out[1]
        else:
            raise ValueError

        return out

    def freeze_params(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()
        logger.info("bert params is freezed")