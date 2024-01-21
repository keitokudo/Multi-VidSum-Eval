import torch
from torch import nn

from models.bert import Bert
from logzero import logger


class BaselineModel(nn.Module):
    '''
    とりあえずresnetの特徴量とbniの特徴量とキャプションのsentence idを入力として各フレーム数分の出力をする仮のモデル
    '''

    def __init__(self, config, device):
        super(BaselineModel, self).__init__()
        self.config = config
        self.resnet_emb = nn.Linear(
            config["net_params"]["resnet_input_dim"], config["net_params"]["d_model"] // 2)
        self.bni_emb = nn.Linear(
            config["net_params"]["bni_input_dim"], config["net_params"]["d_model"] // 2)

        self.relu = nn.ReLU()

        if "d_linear" in config["net_params"]:
            self.linear = nn.Linear(config["net_params"]["d_model"] + config["language_model"]["d_hidden"], config["net_params"]["d_linear"])
            self.contextualized_input = config["net_params"]["d_linear"]
        else:
            self.contextualized_input = config["net_params"]["d_model"] + config["language_model"]["d_hidden"]

        if config["net_params"]["artchitecture"] == 'lstm':
            logger.info("lstm is used")
            self.contextualized_output = config["net_params"]["d_hidden"]
            self.lstm = nn.LSTM(self.contextualized_input, self.contextualized_output, bidirectional=config["net_params"]["bidirectinal"], batch_first=True)
            if self.config["net_params"]["use_feat"] == "mean":
                self.contextualized_output = config["net_params"]["d_hidden"]
            elif self.config["net_params"]["use_feat"] == "concat":
                self.contextualized_output = config["net_params"]["d_hidden"] * 2
            else:
                raise ValueError(f"{self.config['net_params']['use_feat']} is unknown")
        elif config["net_params"]["artchitecture"] == "transformer":
            logger.info("transformer is used")
            self.contextualized_output = self.contextualized_input
            from models.pos_encoding import PositionalEncoding
            self.pos_encoder = PositionalEncoding(
                d_model=self.contextualized_input)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.contextualized_input, nhead=config["net_params"]["nhead"], batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config["net_params"]["num_layers"])
        else:
            raise ValueError

        self.logits = nn.Linear(self.contextualized_output, 1)
        self.lm = Bert(config["language_model"], device)

    def forward(self, resnet, bni, sentence_ids, frame_masks, sent_masks):
        out = self.predict(resnet, bni, sentence_ids, frame_masks, sent_masks)
        return out

    def predict(self, resnet, bni, sentence_ids, frame_masks, sent_masks):
        batch_size, frame_cnt, _ = resnet.size()
        y_resnet = self.resnet_emb(resnet)
        y_bni = self.bni_emb(bni)
        vis_feat = torch.cat((y_resnet, y_bni), 2)

        # frame_cnt分ブロードキャスト
        sent_feat = self.lm(sentence_ids, sent_masks).view(
            batch_size, 1, -1).tile((1, frame_cnt, 1))

        out = torch.cat((vis_feat, sent_feat), 2)

        out = self.relu(out)

        if "d_linear" in self.config["net_params"]:
            out = self.linear(out)
            out = self.relu(out)

        out = self.contextualize_layer(out, frame_masks)

        out = self.relu(out)

        out = self.logits(out)
        out[frame_masks == 0] = -1e8  # padding分の出力を-infにする
        out = out.view(bni.size(0), -1)  # (batch, 1, frame_size) -> (batch, frame_size)
        return out

    def contextualize_layer(self, out, masks):
        # 混ぜる
        if self.config["net_params"]["artchitecture"] == 'lstm':
            sentence_len = masks.sum(-1).to("cpu")
            out = nn.utils.rnn.pack_padded_sequence(
                out, sentence_len, batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(out)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            if self.config["net_params"]["bidirectinal"] is True:
                if self.config["net_params"]["use_feat"] == "mean":
                    out = out[:, :, :self.config["net_params"]["d_hidden"]] + \
                        out[:, :, self.config["net_params"]["d_hidden"]:]
                    out = out.contiguous()
                elif self.config["net_params"]["use_feat"] == "concat":
                    None
                else:
                    raise ValueError(f"{self.config['net_params']['use_feat']} is unknown")
        elif self.config["net_params"]["artchitecture"] == "transformer":
            # out = out * math.sqrt(self.contextualized_input)
            out = self.pos_encoder(out)

            # The shape of the 2D attn_mask is torch.Size([2, 264]), but should be (tensor(264), tensor(264))
            _masks = (torch.clone(masks) == 0).bool()
            out = self.transformer_encoder(out, src_key_padding_mask=_masks)
        return out

# import torch
# from torch import nn

# import math
# from models.bert import Bert
# from logzero import logger


# class BaselineModel(nn.Module):
#     '''
#     とりあえずresnetの特徴量とbniの特徴量とキャプションのsentence idを入力として各フレーム数分の出力をする仮のモデル
#     '''

#     def __init__(self, config, device):
#         super(BaselineModel, self).__init__()
#         self.config = config
#         self.resnet_emb = nn.Linear(
#             config["net_params"]["resnet_input_dim"], config["net_params"]["d_model"] // 2)
#         self.bni_emb = nn.Linear(
#             config["net_params"]["bni_input_dim"], config["net_params"]["d_model"] // 2)

#         if config["net_params"]["artchitecture"] == 'lstm':
#             logger.info("lstm is used")
#             self.lstm = nn.LSTM(input_size=config["net_params"]["d_model"] + config["language_model"]["d_hidden"],
#                                 hidden_size=config["net_params"]["d_hidden"],
#                                 bidirectional=config["net_params"]["bidirectinal"], batch_first=True)
#         elif config["net_params"]["artchitecture"] == "transformer":
#             logger.info("transformer is used")
#             from models.pos_encoding import PositionalEncoding
#             self.pos_encoder = PositionalEncoding(
#                 d_model=config["net_params"]["d_hidden"])
#             self.pre_transformer_fc = nn.Linear(
#                 config["net_params"]["d_model"] + config["language_model"]["d_hidden"],
#                 config["net_params"]["d_hidden"])
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=config["net_params"]["d_hidden"], nhead=config["net_params"]["nhead"], batch_first=True)
#             self.transformer_encoder = nn.TransformerEncoder(
#                 encoder_layer, num_layers=6)

#         else:
#             raise ValueError

#         self.logits = nn.Linear(config["net_params"]["d_hidden"], 1)
#         self.relu = nn.ReLU()
#         self.lm = Bert(config["language_model"], device)

#     def forward(self, resnet, bni, sentence_ids, frame_masks, sent_masks):
#         out = self.predict(resnet, bni, sentence_ids, frame_masks, sent_masks)
#         return out

#     def predict(self, resnet, bni, sentence_ids, frame_masks, sent_masks):
#         batch_size, frame_cnt, _ = resnet.size()
#         y_resnet = self.resnet_emb(resnet)
#         y_bni = self.bni_emb(bni)
#         vis_feat = torch.cat((y_resnet, y_bni), 2)
#         vis_feat = self.relu(vis_feat)

#         # frame_cnt分ブロードキャスト
#         sent_feat = self.lm(sentence_ids, sent_masks).view(
#             batch_size, 1, -1).tile((1, frame_cnt, 1))

#         out = torch.cat((vis_feat, sent_feat), 2)

#         out = self.contextualize_layer(out, frame_masks)

#         out = self.relu(out)

#         out = self.logits(out)
#         out[frame_masks == 0] = -1e8  # padding分の出力を-infにする
#         out = out.view(bni.size(0), -1)  # (batch, 1, frame_size) -> (batch, frame_size)
#         return out

#     def contextualize_layer(self, out, masks):
#         # 混ぜる
#         if self.config["net_params"]["artchitecture"] == 'lstm':
#             sentence_len = masks.sum(-1).to("cpu")
#             out = nn.utils.rnn.pack_padded_sequence(
#                 out, sentence_len, batch_first=True, enforce_sorted=False)
#             out, _ = self.lstm(out)
#             out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
#             if self.config["net_params"]["bidirectinal"] is True:
#                 out = out[:, :, :self.config["net_params"]["d_hidden"]] + \
#                     out[:, :, self.config["net_params"]["d_hidden"]:]
#                 out = out.contiguous()
#         elif self.config["net_params"]["artchitecture"] == "transformer":
#             out = self.pre_transformer_fc(out)
#             out = self.relu(out)
#             out = out * math.sqrt(self.config["net_params"]["d_hidden"])
#             out = self.pos_encoder(out)

#             # The shape of the 2D attn_mask is torch.Size([2, 264]), but should be (tensor(264), tensor(264))
#             _masks = (torch.clone(masks) == 0).bool()
#             out = self.transformer_encoder(out, src_key_padding_mask=_masks)
#         return out
