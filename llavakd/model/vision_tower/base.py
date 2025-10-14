import os
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None
        self._image_processor = None
        self.config = cfg
    

    def load_model(self, vision_tower_name, **kwargs):
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False)



        
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_vision_tower_path')
        if isinstance(self._vision_tower, PreTrainedModel): # hf model
            if pretrained_vision_tower_path is not None:
                vision_tower_name = pretrained_vision_tower_path
            
            # self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)
            
            checkpoint_path = os.path.join(vision_tower_name, "model.safetensors")
            pre_checkpoint = load_file(checkpoint_path, device='cuda')

            filtered_pretrained_dict = {k: v.to('cuda') for k, v in pre_checkpoint.items() if ('position_embedding' not in k and ('text_model' not in k) and ('logit_bias' not in k) and ('logit_scale' not in k))}
            model_dict = self._vision_tower.state_dict()
            model_dict.update(filtered_pretrained_dict)
            self._vision_tower.load_state_dict(model_dict)

            # # Determine the device of the model
            # device = next(self._vision_tower.parameters()).device

            old_position_embedding = pre_checkpoint['vision_model.embeddings.position_embedding.weight'].to('cuda')
            new_position_embedding = self._vision_tower.vision_model.embeddings.position_embedding.weight
            
            new_num_patches = new_position_embedding.shape[0]
            old_num_patches = old_position_embedding.shape[0]

            if new_num_patches != old_num_patches:
                print(f"Interpolating position embedding from {old_num_patches} to {new_num_patches}")
                old_position_embedding = old_position_embedding.unsqueeze(0).permute(0, 2, 1)
                new_position_embedding = F.interpolate(old_position_embedding, size=new_num_patches, mode='linear')
                new_position_embedding = new_position_embedding.permute(0, 2, 1).squeeze(0).contiguous()

                # 将插值后的pe赋值给vision tower
                self._vision_tower.vision_model.embeddings.position_embedding.weight.data = new_position_embedding
            else:
                # self._vision_tower.vision_model.embeddings.position_embedding.weight.data = old_position_embedding
                self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)
            self._vision_tower = self._vision_tower.to('cuda')

        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)

        print("Loading vision tower from ", vision_tower_name)
        


    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features
        

    
    @property
    def vision_tower(self):
        return self._vision_tower
        
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower
        
    
