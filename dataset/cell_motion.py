# -*- encoding: utf-8 -*-
"""
细胞运动预测数据集
用于处理时间序列的细胞2D坐标和邻接关系，预测下一时刻的细胞位置
"""

from torch.utils.data import IterableDataset
import os
import numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
import torch
import math


class CellMotionBase():
    """
    细胞运动数据集基类
    处理时间序列的细胞位置和邻接关系数据
    """

    def __init__(self, max_epochs=1, files=None, sequence_length=50, enable_shuffle=True, predict_steps=1):
        """
        初始化细胞运动数据集基类
        
        参数:
            max_epochs: 最大训练轮数
            files: HDF5文件句柄字典
            sequence_length: 输入的历史时间步长度
            enable_shuffle: 是否启用数据打乱功能，默认为True
        """
        self.open_tra_num = 50  # 同时打开的轨迹数量
        self.file_handle = files  # HDF5文件句柄
        self.sequence_length = sequence_length  # 输入序列长度
        self.enable_shuffle = enable_shuffle  # 是否启用shuffle
        self.predict_steps = predict_steps  # 需要预测的未来步数
        
        if self.enable_shuffle:
            self.shuffle_file()  # 打乱文件顺序

        # 数据集中包含的关键字段
        self.data_keys = ("pos", "edge_index")  # 位置和邻接关系
        
        self.tra_index = 0  # 当前轨迹索引
        self.epoch_num = 1  # 当前epoch编号
        
        # 数据集属性（将根据实际数据动态设置）
        self.time_interval = 1.0  # 时间间隔（单位可以是分钟、小时等）
        
        # 管理已打开的轨迹
        self.opened_tra = []  # 已打开的轨迹列表
        self.opened_tra_readed_index = {}  # 每条轨迹已读取的索引
        self.opened_tra_readed_random_index = {}  # 每条轨迹的随机读取顺序
        self.tra_data = {}  # 缓存的轨迹数据
        self.tra_lengths = {}  # 每条轨迹的长度
        self.max_epochs = max_epochs  # 最大训练轮数

    def open_tra(self):
        """
        打开新的轨迹数据
        维护一定数量的打开轨迹，当轨迹数量不足时从数据集中加载新的轨迹
        """
        while len(self.opened_tra) < self.open_tra_num:
            tra_index = self.datasets[self.tra_index]
            
            # 如果该轨迹尚未打开，则打开它
            if tra_index not in self.opened_tra:
                # 获取该轨迹的实际长度
                data = self.file_handle[tra_index]
                tra_len = data['pos'].shape[0]
                self.tra_lengths[tra_index] = tra_len
                
                self.opened_tra.append(tra_index)
                self.opened_tra_readed_index[tra_index] = -1
                # 生成随机帧顺序（确保有足够的帧用于序列和未来预测）
                valid_start_frames = tra_len - self.sequence_length - self.predict_steps
                if valid_start_frames > 0:
                    self.opened_tra_readed_random_index[tra_index] = np.random.permutation(valid_start_frames)
                else:
                    # 轨迹太短，跳过
                    self.opened_tra.remove(tra_index)
                    continue
            
            self.tra_index += 1
            
            # 检查是否到达epoch末尾
            if self.check_if_epoch_end():
                self.epoch_end()
                break

    def check_and_close_tra(self):
        """
        检查并关闭已读取完毕的轨迹
        释放已读完的轨迹占用的内存
        """
        to_del = []
        # 找出所有已读完的轨迹
        for tra in self.opened_tra:
            tra_len = self.tra_lengths.get(tra, 0)
            valid_frames = tra_len - self.sequence_length - self.predict_steps
            if self.opened_tra_readed_index[tra] >= (valid_frames - 1):
                to_del.append(tra)
        
        # 关闭并删除这些轨迹的相关数据
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
                del self.tra_data[tra]
                del self.tra_lengths[tra]
            except Exception as e:
                print(f"Error closing trajectory {tra}: {e}")

    def shuffle_file(self):
        """
        打乱数据集文件顺序
        用于每个epoch开始时随机化训练数据顺序
        """
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epoch_end(self):
        """
        处理epoch结束逻辑
        重置轨迹索引，重新打乱数据，增加epoch计数
        """
        self.tra_index = 0
        self.shuffle_file()
        self.epoch_num = self.epoch_num + 1

    def check_if_epoch_end(self):
        """
        检查当前epoch是否已结束
        
        返回:
            bool: 如果所有轨迹都已遍历则返回True
        """
        if self.tra_index >= len(self.file_handle):
            return True
        return False

    @staticmethod
    def datas_to_graph(datas, sequence_length):
        """
        将原始数据转换为PyTorch Geometric图数据格式
        
        参数:
            datas: 包含以下元素的字典
                'pos_sequence': 位置序列 (sequence_length, N, 2)
                'pos_target': 目标位置 (N, 2)
                'edge_index': 邻接关系 (2, E)
                'time': 时间戳
        
        返回:
            Data: PyTorch Geometric图数据对象
        """
        pos_sequence = datas['pos_sequence']  # (seq_len, N, 2)
        pos_target = datas['pos_target']  # (N, 2)
        edge_index = datas['edge_index']  # (2, E)
        time = datas['time']
        pos_future = datas.get('future_pos', None)
        
        num_nodes = pos_sequence.shape[1]
        seq_len = pos_sequence.shape[0]
        
        # 构建节点特征：将序列展平为特征
        # 形状: (N, sequence_length * 2)
        pos_flat = pos_sequence.transpose(1, 0, 2).reshape(num_nodes, -1)
        
        # 可选：添加时间特征
        time_vector = np.ones((num_nodes, 1)) * time
        
        # 节点特征 = [历史位置序列, 时间]
        node_attr = np.hstack([pos_flat, time_vector])
        
        # 转换为张量
        node_attr = torch.as_tensor(node_attr, dtype=torch.float32)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        pos_target = torch.as_tensor(pos_target, dtype=torch.float32)
        pos = torch.as_tensor(pos_sequence[-1], dtype=torch.float32)  # 当前位置
        if pos_future is not None:
            pos_future = torch.as_tensor(pos_future, dtype=torch.float32)
        
        # 创建图数据对象
        g = Data(
            x=node_attr,  # 节点特征
            edge_index=edge_index,  # 边索引
            y=pos_target,  # 目标位置
            pos=pos  # 当前位置（用于可视化）
        )
        
        if pos_future is not None:
            g.future_pos = pos_future
        
        return g

    def __next__(self):
        """
        迭代器的下一个元素
        返回一个图数据样本
        
        返回:
            Data: PyTorch Geometric图数据对象
        
        异常:
            StopIteration: 当达到最大epoch数时停止迭代
        """
        # 检查并关闭已读完的轨迹
        self.check_and_close_tra()
        # 打开新的轨迹以保持足够的打开数量
        self.open_tra()
        
        # 如果超过最大epoch数，停止迭代
        if self.epoch_num > self.max_epochs:
            raise StopIteration
        
        # 如果没有可用的轨迹，停止迭代
        if len(self.opened_tra) == 0:
            raise StopIteration
        
        # 从已打开的轨迹中随机选择一条
        selected_tra = np.random.choice(self.opened_tra)
        
        # 获取或加载该轨迹的数据
        data = self.tra_data.get(selected_tra, None)
        if data is None:
            data = self.file_handle[selected_tra]
            self.tra_data[selected_tra] = data
        
        # 获取该轨迹的当前读取索引和随机帧顺序
        selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
        selected_start_frame = self.opened_tra_readed_random_index[selected_tra][selected_tra_readed_index + 1]
        self.opened_tra_readed_index[selected_tra] += 1
        
        # 读取序列数据
        # pos_sequence: 从selected_start_frame开始的sequence_length帧
        pos_sequence = np.array(
            data['pos'][selected_start_frame:selected_start_frame + self.sequence_length],
            dtype=np.float32
        )  # (sequence_length, N, 2)
        
        # future positions: sequence_length 之后的多步位置
        future_start = selected_start_frame + self.sequence_length
        future_end = future_start + self.predict_steps
        pos_future = np.array(
            data['pos'][future_start:future_end],
            dtype=np.float32
        )
        pos_target = pos_future[0]  # (N, 2)
        
        # edge_index: 邻接关系（假设在整个轨迹中保持不变）
        edge_index = np.array(data['edge_index'], dtype=np.int64)  # (2, E)
        
        # 时间戳
        time = np.array([self.time_interval * selected_start_frame], dtype=np.float32)
        
        # 组装数据
        datas = {
            'pos_sequence': pos_sequence,
            'pos_target': pos_target,
            'edge_index': edge_index,
            'time': time,
            'future_pos': pos_future
        }
        
        # 转换为图数据格式
        g = self.datas_to_graph(datas, self.sequence_length)
        
        return g

    def __iter__(self):
        """返回迭代器自身"""
        return self


class CellMotion(IterableDataset):
    """
    细胞运动数据集类
    继承自PyTorch的IterableDataset，用于训练和验证
    支持多进程数据加载
    """
    
    def __init__(self, max_epochs, dataset_dir, split='train', sequence_length=10, predict_steps=1):
        """
        初始化细胞运动数据集
        
        参数:
            max_epochs: 最大训练轮数
            dataset_dir: 数据集目录路径
            split: 数据集划分，可选 'train', 'valid', 'test'
            sequence_length: 输入的历史时间步长度
        """
        super().__init__()
        
        dataset_path = osp.join(dataset_dir, split + '.h5')
        self.max_epochs = max_epochs
        self.dataset_dir = dataset_path
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps
        
        assert os.path.isfile(dataset_path), f'{dataset_path} not exist'
        
        # 以只读模式打开HDF5文件
        self.file_handle = h5py.File(dataset_path, "r")
        print(f'CellMotion Dataset {self.dataset_dir} Initialized (sequence_length={sequence_length})')

    def __iter__(self):
        """
        创建迭代器
        支持多进程数据加载时的数据分片
        
        返回:
            CellMotionBase: 数据集迭代器实例
        """
        # 获取数据加载器的worker信息（用于多进程加载）
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # 单进程模式：使用全部数据
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            # 多进程模式：将数据分片给各个worker
            per_worker = int(math.ceil(len(self.file_handle) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))
        
        # 为当前worker分配数据子集
        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        
        return CellMotionBase(
            max_epochs=self.max_epochs,
            files=files,
            sequence_length=self.sequence_length,
            predict_steps=self.predict_steps
        )


class CellMotionRollout(IterableDataset):
    """
    细胞运动rollout数据集类
    用于模型推理和长时间序列预测
    按顺序遍历完整的轨迹序列
    """
    
    def __init__(self, dataset_dir, split='test', sequence_length=10, predict_steps=1):
        """
        初始化细胞运动rollout数据集
        
        参数:
            dataset_dir: 数据集目录路径
            split: 数据集划分，通常为 'test'
            sequence_length: 输入的历史时间步长度
        """
        dataset_path = osp.join(dataset_dir, split + '.h5')
        self.dataset_dir = dataset_path
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps
        
        assert os.path.isfile(dataset_path), f'{dataset_path} not exist'
        
        # 以只读模式打开HDF5文件
        self.file_handle = h5py.File(dataset_path, "r")
        self.data_keys = ("pos", "edge_index")
        self.time_interval = 1.0  # 时间间隔
        
        # 初始化属性，避免AttributeError
        self.cur_tra = None
        self.cur_trajectory_length = 0
        self.cur_trajectory_index = 0
        self.current_file_index = 0  # 添加文件索引追踪
        
        self.load_dataset()

    def load_dataset(self):
        """加载数据集的所有轨迹键"""
        datasets = list(self.file_handle.keys())
        self.datasets = datasets

    def change_file(self, file_index):
        """
        切换到指定的轨迹文件
        
        参数:
            file_index: 轨迹在数据集中的索引
        """
        if file_index >= len(self.datasets):
            raise IndexError(f"文件索引 {file_index} 超出范围，总文件数: {len(self.datasets)}")
        
        file_key = self.datasets[file_index]
        self.cur_tra = self.file_handle[file_key]  # 当前轨迹数据
        self.cur_trajectory_length = self.cur_tra['pos'].shape[0]  # 轨迹长度
        self.cur_trajectory_index = 0  # 当前帧索引
        self.current_file_index = file_index

    def __next__(self):
        """
        获取当前轨迹的下一帧数据
        按顺序遍历整个轨迹（不随机）
        
        返回:
            Data: PyTorch Geometric图数据对象
        
        异常:
            StopIteration: 当到达轨迹末尾时停止迭代
        """
        # 检查是否已初始化轨迹
        if self.cur_tra is None:
            raise StopIteration
        
        # 如果已到达轨迹末尾（需要预留足够的序列长度和未来帧）
        if self.cur_trajectory_index >= (self.cur_trajectory_length - self.sequence_length - self.predict_steps + 1):
            raise StopIteration
        
        data = self.cur_tra
        start_frame = self.cur_trajectory_index
        
        # 读取序列数据
        pos_sequence = np.array(
            data['pos'][start_frame:start_frame + self.sequence_length],
            dtype=np.float32
        )
        
        # 未来位置序列
        future_start = start_frame + self.sequence_length
        future_end = future_start + self.predict_steps
        pos_future = np.array(
            data['pos'][future_start:future_end],
            dtype=np.float32
        )
        pos_target = pos_future[0]
        
        # 邻接关系
        edge_index = np.array(data['edge_index'], dtype=np.int64)
        
        # 时间戳
        time = np.array([self.time_interval * start_frame], dtype=np.float32)
        
        # 移动到下一帧
        self.cur_trajectory_index += 1
        
        # 组装数据
        datas = {
            'pos_sequence': pos_sequence,
            'pos_target': pos_target,
            'edge_index': edge_index,
            'time': time,
            'future_pos': pos_future
        }
        
        # 转换为图数据格式
        g = CellMotionBase.datas_to_graph(datas, self.sequence_length)
        
        return g

    def __iter__(self):
        """
        返回迭代器自身
        每次迭代开始时，选择一个轨迹文件并重置索引
        """
        # 选择第一个轨迹文件（或者可以随机选择）
        if len(self.datasets) > 0:
            # 方案1: 总是从第一个文件开始
            self.change_file(0)
            
            # 方案2: 随机选择一个文件（如果需要随机化）
            # import random
            # file_idx = random.randint(0, len(self.datasets) - 1)
            # self.change_file(file_idx)
        else:
            # 如果没有数据集，设置为空状态
            self.cur_tra = None
            self.cur_trajectory_length = 0
            self.cur_trajectory_index = 0
        
        return self
    
    def __len__(self):
        """返回数据集中的轨迹数量"""
        return len(self.datasets)
    
    def get_trajectory_length(self, file_index):
        """
        获取指定轨迹的长度
        
        参数:
            file_index: 轨迹索引
            
        返回:
            int: 轨迹中的帧数
        """
        file_key = self.datasets[file_index]
        return self.file_handle[file_key]['pos'].shape[0]
    
    def __del__(self):
        """析构函数，关闭HDF5文件"""
        if hasattr(self, 'file_handle') and self.file_handle is not None:
            self.file_handle.close()
