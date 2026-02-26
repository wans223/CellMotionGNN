from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
import torch
import math
import time

class FPCBase():
    """
    FPC数据集基类，用于处理流体动力学模拟数据（Flow Past Cylinder - 圆柱绕流）
    实现了轨迹数据的迭代加载和预处理
    """

    def __init__(self, max_epochs=1, files=None):
        """
        初始化FPC数据集基类
        
        参数:
            max_epochs: 最大训练轮数
            files: HDF5文件句柄字典
        """

        self.open_tra_num = 10  # 同时打开的轨迹数量
        self.file_handle = files  # HDF5文件句柄
        self.shuffle_file()  # 打乱文件顺序

        # 数据集中包含的关键字段
        self.data_keys =  ("pos", "node_type", "velocity", "cells", "pressure")
        self.out_keys = list(self.data_keys)  + ['time']

        self.tra_index = 0  # 当前轨迹索引
        self.epcho_num = 1  # 当前epoch编号
        self.tra_readed_index = -1  # 已读取的轨迹索引

        # 数据集属性
        self.tra_len = 600  # 每条轨迹的长度（时间步数）
        self.time_iterval = 0.01  # 时间间隔

        # 管理已打开的轨迹
        self.opened_tra = []  # 已打开的轨迹列表
        self.opened_tra_readed_index = {}  # 每条轨迹已读取的索引
        self.opened_tra_readed_random_index = {}  # 每条轨迹的随机读取顺序
        self.tra_data = {}  # 缓存的轨迹数据
        self.max_epochs = max_epochs  # 最大训练轮数

    
    def open_tra(self):
        """
        打开新的轨迹数据
        维护一定数量的打开轨迹，当轨迹数量不足时从数据集中加载新的轨迹
        """
        while(len(self.opened_tra) < self.open_tra_num):

            tra_index = self.datasets[self.tra_index]

            # 如果该轨迹尚未打开，则打开它
            if tra_index not in self.opened_tra:
                self.opened_tra.append(tra_index)
                self.opened_tra_readed_index[tra_index] = -1
                # 为该轨迹生成随机的帧读取顺序（用于数据增强）
                self.opened_tra_readed_random_index[tra_index] = np.random.permutation(self.tra_len - 2)

            self.tra_index += 1

            # 检查是否到达epoch末尾
            if self.check_if_epcho_end():
                self.epcho_end()
                print('Epcho Finished')
    
    def check_and_close_tra(self):
        """
        检查并关闭已读取完毕的轨迹
        释放已读完的轨迹占用的内存
        """
        to_del = []
        # 找出所有已读完的轨迹（读取索引超过轨迹长度-3）
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - 3):
                to_del.append(tra)
        # 关闭并删除这些轨迹的相关数据
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
                del self.tra_data[tra]
            except Exception as e:
                print(e)
                


    def shuffle_file(self):
        """
        打乱数据集文件顺序
        用于每个epoch开始时随机化训练数据顺序
        """
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epcho_end(self):
        """
        处理epoch结束逻辑
        重置轨迹索引，重新打乱数据，增加epoch计数
        """
        self.tra_index = 0
        self.shuffle_file()
        self.epcho_num = self.epcho_num + 1

    def check_if_epcho_end(self):
        """
        检查当前epoch是否已结束
        
        返回:
            bool: 如果所有轨迹都已遍历则返回True
        """
        if self.tra_index >= len(self.file_handle):
            return True
        return False

    @staticmethod
    def datas_to_graph(datas):
        """
        将原始数据转换为PyTorch Geometric图数据格式
        
        参数:
            datas: 包含以下元素的列表
                [0] pos: 节点位置坐标
                [1] node_type: 节点类型
                [2] velocity: 速度 [当前帧, 下一帧]
                [3] cells: 网格单元（面）
                [4] pressure: 压力 [当前帧, 下一帧]
                [5] time: 时间戳
        
        返回:
            Data: PyTorch Geometric图数据对象
        """
        # 创建时间向量，所有节点共享同一时间戳
        time_vector = np.ones((datas[0].shape[0], 1))*datas[5]
        # 拼接节点属性：节点类型、当前速度、当前压力、时间
        node_attr = np.hstack((datas[1], datas[2][0], datas[4][0], time_vector))
        # 节点属性包含: node_type, cur_v, pressure, time
        crds = torch.as_tensor(datas[0], dtype=torch.float)
        # senders = edge_index[0].numpy()
        # receivers = edge_index[1].numpy()
        # crds_diff = crds[senders] - crds[receivers]
        # crds_norm = np.linalg.norm(crds_diff, axis=1, keepdims=True)
        # edge_attr = np.concatenate((crds_diff, crds_norm), axis=1)

        # 目标是下一时刻的速度（用于预测加速度）
        target = datas[2][1]
        # 转换为张量
        node_attr = torch.as_tensor(node_attr, dtype=torch.float32)
        # edge_attr = torch.from_numpy(edge_attr)
        target = torch.from_numpy(target)
        # 网格面信息需要转置以符合PyG格式
        face = torch.as_tensor(datas[3].T, dtype=torch.long)
        # 创建图数据对象：x=节点特征, face=面, y=目标, pos=位置
        g = Data(x=node_attr, face=face, y=target, pos=crds)
        # g = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=target, pos=crds)
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
        if self.epcho_num > self.max_epochs:
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
        selected_frame = self.opened_tra_readed_random_index[selected_tra][selected_tra_readed_index+1]
        self.opened_tra_readed_index[selected_tra] += 1

        # 加载选中帧的所有数据
        datas = []
        for k in self.data_keys:
            if k in ["velocity", "pressure"]:
                # 速度和压力需要两帧数据（当前帧和下一帧）
                r = np.array((data[k][selected_frame], data[k][selected_frame+1]), dtype=np.float32)
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)
        # 添加时间戳
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))
        # 数据顺序: ("pos", "node_type", "velocity", "cells", "pressure", "time")
        g = self.datas_to_graph(datas)
  
        return g

    def __iter__(self):
        """返回迭代器自身"""
        return self


class FPC(IterableDataset):
    """
    FPC数据集类（Flow Past Cylinder - 圆柱绕流）
    继承自PyTorch的IterableDataset，用于训练和验证
    支持多进程数据加载
    """
    def __init__(self, max_epochs, dataset_dir, split='train') -> None:
        """
        初始化FPC数据集
        
        参数:
            max_epochs: 最大训练轮数
            dataset_dir: 数据集目录路径
            split: 数据集划分，可选 'train', 'valid', 'test'
        """
        super().__init__()

        dataset_dir = osp.join(dataset_dir, split+'.h5')
        self.max_epochs = max_epochs
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        # 以只读模式打开HDF5文件
        self.file_handle = h5py.File(dataset_dir, "r")
        print('Dataset '+  self.dataset_dir + ' Initilized')

    def __iter__(self):
        """
        创建迭代器
        支持多进程数据加载时的数据分片
        
        返回:
            FPCBase: 数据集迭代器实例
        """
        # 获取数据加载器的worker信息（用于多进程加载）
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单进程模式：使用全部数据
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            # 多进程模式：将数据分片给各个worker
            per_worker = int(math.ceil(len(self.file_handle)/float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))

        # 为当前worker分配数据子集
        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        return FPCBase(max_epochs=self.max_epochs, files=files)


class FPC_ROLLOUT(IterableDataset):
    """
    FPC rollout数据集类
    用于模型推理和长时间序列预测（rollout）
    按顺序遍历完整的轨迹序列
    """
    def __init__(self, dataset_dir, split='test', name='flow pass a cylinder'):
        """
        初始化FPC rollout数据集
        
        参数:
            dataset_dir: 数据集目录路径
            split: 数据集划分，通常为 'test'
            name: 数据集名称描述
        """
        dataset_dir = osp.join(dataset_dir, split+'.h5')
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        # 以只读模式打开HDF5文件
        self.file_handle = h5py.File(dataset_dir, "r")
        self.data_keys =  ("pos", "node_type", "velocity", "cells", "pressure")
        self.time_iterval = 0.01  # 时间间隔
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
        file_index = self.datasets[file_index]
        self.cur_tra = self.file_handle[file_index]  # 当前轨迹数据
        self.cur_targecity_length = self.cur_tra['velocity'].shape[0]  # 轨迹长度
        self.cur_tragecity_index = 0  # 当前帧索引
        self.edge_index = None

    def __next__(self):
        """
        获取当前轨迹的下一帧数据
        按顺序遍历整个轨迹（不随机）
        
        返回:
            Data: PyTorch Geometric图数据对象
        
        异常:
            StopIteration: 当到达轨迹末尾时停止迭代
        """
        # 如果已到达轨迹末尾，停止迭代
        if self.cur_tragecity_index == (self.cur_targecity_length - 1):
            raise StopIteration

        datas = []
        data = self.cur_tra
        selected_frame = self.cur_tragecity_index

        # 加载当前帧的所有数据
        datas = []
        for k in self.data_keys:
            if k in ["velocity", "pressure"]:
                # 速度和压力需要两帧数据（当前帧和下一帧）
                r = np.array((data[k][selected_frame], data[k][selected_frame+1]), dtype=np.float32)
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)
        # 添加时间戳
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))

        # 移动到下一帧
        self.cur_tragecity_index += 1
        # 转换为图数据格式
        g = FPCBase.datas_to_graph(datas)
        # self.edge_index = g.edge_index
        return g


    def __iter__(self):
        """返回迭代器自身"""
        return self

