"""
    Adapted from https://github.com/google/prompt-to-prompt/tree/main
        and
    https://github.com/silent-chen/layout-guidance
"""

import abc
import math
import torch
from typing import Tuple


LOW_RESOURCE = False


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] == self.block_res[place_in_unet] ** 2:
            self.step_store[key].append(attn)
        return attn
     
    def set_to_mask(self, masking):
        self.mask = masking

    def between_steps(self):
        if not self.mask:
            if len(self.attention_store) == 0:
                if not self.sum_blocks[0]:
                    for key in self.step_store:
                        self.step_store[key] = [torch.cat(self.step_store[key], dim=0)]
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    if self.sum_blocks[0]:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
                    else:
                        concat_attn = torch.cat(self.step_store[key], dim=0)
                        self.attention_store[key] += [concat_attn]
        else:
            if len(self.mask_attention_store) == 0:
                if not self.sum_blocks[1]:
                    for key in self.step_store:
                        self.step_store[key] = [torch.cat(self.step_store[key], dim=0)]
                self.mask_attention_store = self.step_store
            else:
                for key in self.mask_attention_store:
                    if self.sum_blocks[1]:
                        for i in range(len(self.mask_attention_store[key])):
                            self.mask_attention_store[key][i] += self.step_store[key][i]
                    else:
                        concat_attn = torch.cat(self.step_store[key], dim=0)
                        self.mask_attention_store[key] += [concat_attn]

        self.step_store = self.get_empty_store()

    def get_average_attention(self, mask=False):
        if not mask:
            average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        else:
            average_attention = {key: [item / self.cur_step for item in self.mask_attention_store[key]] for key in self.mask_attention_store}
        return average_attention

    def get_attention(self, mask=False):
        if not mask:
            return self.attention_store
        else:
            return self.mask_attention_store

    def reset(self, default=True, mask=True):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        if default:
            self.attention_store = {}
        if mask:
            self.mask_attention_store = {}

    def __init__(self, down_res: int = 16, mid_res: int = 8, up_res: int = 16, sum_blocks: Tuple[bool, bool] = (True, True)):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.mask_attention_store = {}
        self.mask = False
        self.sum_blocks = sum_blocks # Whether to sum or concat attentions for: (attention_store, mask_attention_store)

        self.block_res = {"down": down_res, "mid": mid_res, "up": up_res}


def compute_ca_loss(attention_dict, bboxes, object_positions):
    loss = 0
    object_number = len(bboxes)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    
    for attn_map in attention_dict['mid_cross']:
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:

                x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss/len(object_positions[obj_idx]))

    for attn_map in attention_dict['up_cross']:
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))

    loss = loss / (object_number * (len(attention_dict['up_cross'])) + len(attention_dict['mid_cross']))
    return loss
