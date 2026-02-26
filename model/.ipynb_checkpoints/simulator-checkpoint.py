from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os

class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device, hidden_size=128, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size =  node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        # 修改点：将 hidden_size 传递给 EncoderProcesserDecoder
        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num, 
            node_input_size=node_input_size, 
            edge_input_size=edge_input_size,
            hidden_size=hidden_size
        ).to(device)
        
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print(f'Simulator model initialized with hidden_size={hidden_size}')

    def update_node_attr(self, frames, types:torch.Tensor):
        node_feature = []

        node_feature.append(frames) #velocity
        node_type = torch.squeeze(types.long())
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    def velocity_to_accelation(self, noised_frames, next_velocity):

        acc_next = next_velocity - noised_frames
        return acc_next


    def forward(self, graph:Data, velocity_sequence_noise=None):
        # 注意：原代码 forward 签名可能不一样，这里保持兼容性
        # 如果是训练代码直接调用 model.model(graph)，那么这个 forward 实际上可能没被用到
        # 但为了完整性，这里保留原逻辑
        
        if self.training:
            # 注意：这里的逻辑似乎是针对特定任务的（带噪声训练），
            # 如果你的训练脚本直接调用的 self.model(graph)，那下面这段其实不走。
            # 假设保持原样：
            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            target = graph.y
            
            # 如果没传噪声，给个默认零
            if velocity_sequence_noise is None:
                velocity_sequence_noise = torch.zeros_like(frames)

            noised_frames = frames + velocity_sequence_noise
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x = node_attr
            predicted = self.model(graph)

            target_acceration = self.velocity_to_accelation(noised_frames, target)
            target_acceration_normalized = self._output_normalizer(target_acceration, self.training)

            return predicted, target_acceration_normalized

        else:

            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            node_attr = self.update_node_attr(frames, node_type)
            graph.x = node_attr
            predicted = self.model(graph)

            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update

            return predicted_velocity

    def load_checkpoint(self, ckpdir=None):
        
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            # 这里的恢复逻辑可能需要注意，如果是复杂的对象嵌套可能会有问题
            # 但针对 normalizer 应该是没问题的
            if hasattr(self, k):
                obj = getattr(self, k)
                # 假设 v 是 normalizer 的参数字典
                if hasattr(obj, 'load_state_dict'): # 如果有标准的 load 方法
                     # 这里原代码是逐个属性设置，这比较危险，建议标准化
                     pass 
                
                # 保持原代码的逻辑：
                for para, value in v.items():
                    setattr(obj, para, value)

        print("Simulator model loaded checkpoint %s"%ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir=self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)