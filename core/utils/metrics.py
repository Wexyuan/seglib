import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, List, Optional
from tabulate import tabulate

class Metrics:
    """
    语义分割评估指标计算类
    
    该类提供了计算各种语义分割评估指标的功能，包括：
    - 混淆矩阵计算
    - 精确度(Precision)、召回率(Recall)、F1分数
    - 交并比(IoU)、Dice系数
    - 总体准确率(OA)、Kappa系数

    混淆矩阵说明：
        预测值 →
    真实值 ↓  P    N
        P    TP   FP
        N    FN   TN
    
    Attributes:
        num_classes: 类别数量
        hist: 混淆矩阵，形状为(num_classes, num_classes)
        eps: 数值稳定性的小常数
    """
    
    def __init__(self, num_classes: int) -> None:
        """
        初始化评估指标计算器
        
        Args:
            num_classes: 类别数量
        """
        if num_classes <= 0:
            raise ValueError("Number of classes must be positive")
            
        self.num_classes = num_classes
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.eps = 1e-8

    def reset(self) -> None:
        """重置混淆矩阵"""
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def add_batch(self, gt_image: Union[torch.Tensor, np.ndarray], pre_image: Union[torch.Tensor, np.ndarray]) -> None:
        """
        添加一个批次的预测结果和真实标签
        
        Args:
            gt_image: 真实标签，形状为(b, h, w)或(h, w)
            pre_image: 预测结果，形状为(b, c, h, w)、(b, h, w)或(h, w)
            
        Raises:
            ValueError: 当输入张量维度不匹配时
        """
        # 转换为numpy数组
        if isinstance(gt_image, torch.Tensor):
            gt_image = gt_image.cpu().detach().numpy()
        if isinstance(pre_image, torch.Tensor):
            pre_image = pre_image.cpu().detach().numpy()

        # 处理预测图像的形状，确保其为 (b, h, w)
        if pre_image.ndim == 4:  # (b, c, h, w)
            pre_image = np.argmax(pre_image, axis=1)

        # 验证输入形状
        if gt_image.ndim not in [2, 3]:
            raise ValueError(f"gt_image should have 2 or 3 dimensions, got {gt_image.ndim}")
        if pre_image.ndim not in [2, 3]:
            raise ValueError(f"pre_image should have 2 or 3 dimensions, got {pre_image.ndim}")

        # 处理批次数据
        if pre_image.ndim == gt_image.ndim == 3:  # (b, h, w)
            for i in range(gt_image.shape[0]):
                self.hist += self._compute_hist(
                    np.asarray(gt_image[i]), np.asarray(pre_image[i])
                )
        elif pre_image.ndim == gt_image.ndim == 2:  # (h, w)
            self.hist += self._compute_hist(
                np.asarray(gt_image), np.asarray(pre_image)
            )
        else:
            raise ValueError(
                f"gt_image and pre_image should have the same number of dimensions, "
                f"but got {gt_image.ndim} and {pre_image.ndim}"
            )

    def _compute_hist(self, gt_image: np.ndarray, pre_image: np.ndarray) -> np.ndarray:
        """
        计算单个图像的混淆矩阵
        
        Args:
            gt_image: 真实标签，形状为(h, w)
            pre_image: 预测结果，形状为(h, w)
            
        Returns:
            np.ndarray: 混淆矩阵，形状为(num_classes, num_classes)
        """
        if gt_image.shape != pre_image.shape:
            raise ValueError(
                f"gt_image and pre_image should have the same shape, "
                f"but got {gt_image.shape} and {pre_image.shape}"
            )
            
        # 过滤掉不在类别范围内的像素
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        if not np.any(mask):
            return np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            
        # 计算每个像素的标签
        label = self.num_classes * gt_image[mask].astype(np.int64) + pre_image[mask]
        
        # 统计每个标签的数量
        count = np.bincount(label, minlength=self.num_classes ** 2)
        
        # 将统计结果转换为混淆矩阵的形状
        hist = count.reshape(self.num_classes, self.num_classes)
        return hist

    def _get_tp_fp_tn_fn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算真阳性(TP)、假阳性(FP)、真阴性(TN)、假阴性(FN)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                TP, FP, TN, FN数组，每个数组的形状为(num_classes,)
        """
        tp = np.diag(self.hist)  # TP
        fp = self.hist.sum(axis=0) - np.diag(self.hist)  # FP
        fn = self.hist.sum(axis=1) - np.diag(self.hist)  # FN
        tn = self.hist.sum() - (tp + fp + fn)  # TN
        return tp, fp, tn, fn

    def precision(self) -> np.ndarray:
        """
        计算精确度(Precision)
        Precision = TP / (TP + FP)
        
        Returns:
            np.ndarray: 每个类别的精确度，形状为(num_classes,)
        """
        tp, fp, _, _ = self._get_tp_fp_tn_fn()
        precision = tp / (tp + fp + self.eps)
        return precision

    def recall(self) -> np.ndarray:
        """
        计算召回率(Recall)
        Recall = TP / (TP + FN)
        
        Returns:
            np.ndarray: 每个类别的召回率，形状为(num_classes,)
        """
        tp, _, fn, _ = self._get_tp_fp_tn_fn()
        recall = tp / (tp + fn + self.eps)
        return recall

    def f1_score(self) -> np.ndarray:
        """
        计算F1分数(F1 Score)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Returns:
            np.ndarray: 每个类别的F1分数，形状为(num_classes,)
        """
        precision = self.precision()
        recall = self.recall()
        f1 = (2.0 * precision * recall) / (precision + recall + self.eps)
        return f1

    def overall_accuracy(self) -> float:
        """
        计算总体准确率(Overall Accuracy, OA)
        OA = (TP + TN) / (TP + TN + FP + FN)
        
        Returns:
            float: 总体准确率
        """
        oa = np.diag(self.hist).sum() / (self.hist.sum() + self.eps)
        return oa

    def intersection_over_union(self) -> np.ndarray:
        """
        计算交并比(Intersection over Union, IoU)
        IoU = TP / (TP + FP + FN)
        
        Returns:
            np.ndarray: 每个类别的IoU，形状为(num_classes,)
        """
        tp, fp, _, fn = self._get_tp_fp_tn_fn()
        iou = tp / (tp + fn + fp + self.eps)
        return iou

    def dice_coefficient(self) -> np.ndarray:
        """
        计算Dice系数(Dice Coefficient)
        Dice = 2 * TP / (2 * TP + FP + FN)
        
        Returns:
            np.ndarray: 每个类别的Dice系数，形状为(num_classes,)
        """
        tp, fp, _, fn = self._get_tp_fp_tn_fn()
        dice = 2 * tp / ((tp + fp) + (tp + fn) + self.eps)
        return dice

    def pixel_accuracy_class(self) -> np.ndarray:
        """
        计算每个类别的像素准确率(Pixel Accuracy per Class)
        Pixel Accuracy per Class = TP / (TP + FP)
        
        Returns:
            np.ndarray: 每个类别的像素准确率，形状为(num_classes,)
        """
        acc = np.diag(self.hist) / (self.hist.sum(axis=0) + self.eps)
        return acc

    def kappa_coefficient(self) -> float:
        """
        计算Kappa系数
        Kappa = (Po - Pe) / (1 - Pe)
        
        Returns:
            float: Kappa系数
        """
        sum0 = np.sum(self.hist, axis=0)
        sum1 = np.sum(self.hist, axis=1)
        total = np.sum(sum0)
        
        if total == 0:
            return 0.0
            
        expected = np.outer(sum0, sum1) / total
        w_mat = np.ones([self.num_classes, self.num_classes], dtype=int)
        w_mat.flat[:: self.num_classes + 1] = 0

        k = np.sum(w_mat * self.hist) / np.sum(w_mat * expected)
        return 1 - k

    def compute_metrics(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        计算所有评估指标
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, np.ndarray]]: 
                总体指标字典和每个类别的指标字典
        """
        perclass_metrics = {
            "Precision": np.round(self.precision(), 4),
            "Recall": np.round(self.recall(), 4),
            "F1": np.round(self.f1_score(), 4),
            "IoU": np.round(self.intersection_over_union(), 4),
            "Dice": np.round(self.dice_coefficient(), 4),
        }

        overall_metrics = {
            "OA": np.round(self.overall_accuracy(), 4),
            "Kappa": np.round(self.kappa_coefficient(), 4),
            "mPrecision": np.round(np.nanmean(self.precision()), 4),
            "mRecall": np.round(np.nanmean(self.recall()), 4),
            "mF1": np.round(np.nanmean(self.f1_score()), 4),
            "mIoU": np.round(np.nanmean(self.intersection_over_union()), 4),
            "mDice": np.round(np.nanmean(self.dice_coefficient()), 4),
        }

        return overall_metrics, perclass_metrics


if __name__ == '__main__':
    # 示例用法
    num_classes = 6
    
    # 创建Metrics实例
    metrics = Metrics(num_classes)
    
    # 模拟一些预测结果和真实标签
    batch_size, height, width = 2, 512, 512
    
    # 真实标签 (b, h, w)
    gt_images = np.random.randint(0, num_classes, size=(batch_size, height, width))
    
    # 预测结果 (b, c, h, w) - 假设每个像素的预测是一个概率分布
    pre_images = np.random.rand(batch_size, num_classes, height, width)
    
    # 添加批次数据
    metrics.add_batch(gt_images, pre_images)
    
    # 计算指标
    overall_metrics, perclass_metrics = metrics.compute_metrics()
    
    # 打印结果
    print("Overall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value}")
    
    print("\nPer-Class Metrics:")
    for metric, values in perclass_metrics.items():
        print(f"{metric}: {values}")
    
