import re
import shutil
import inspect
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
from loguru import logger
from typing import Union, Optional, Dict, Any, List, Tuple
from prettytable import PrettyTable
from tensorboardX import SummaryWriter

class Tracker:
    """
    训练过程跟踪器类
    
    提供训练过程中的指标记录、日志管理、模型备份等功能。
    支持自动版本管理、指标可视化、配置和代码备份。
    
    Attributes:
        name: 实验名称
        path: 结果根路径
        work_path: 工作路径
        version_name: 版本名称
        version_path: 版本路径
        backup_path: 备份路径
        logfiles_path: 日志文件路径
        checkpoints_path: 检查点路径
        metrics_history: 指标历史记录
        metrics_filepath: 指标文件路径
        logger: 日志记录器
        writer: TensorBoard写入器
    """

    def __init__(
        self,
        name: str,
        path: Union[str, Path],
        version: Optional[Union[int, str]] = None
    ) -> None:
        """
        初始化跟踪器
        
        Args:
            name: 实验名称
            path: 结果根路径
            version: 版本号，如果为None则自动生成
        """
        self.name = name
        self.path = Path(path)
        self.work_path = self.path / self.name

        self.version_name = str(version) if version is not None else self._get_version_name()
        self.version_path = self.work_path / self.version_name

        self.backup_path = self.version_path / 'backups'
        self.logfiles_path = self.version_path / 'logfiles'
        self.checkpoints_path = self.version_path / 'checkpoints'

        self.metrics_history: Dict[str, List[Any]] = {}
        self.metrics_filepath = self.logfiles_path / 'metrics.csv'

        self.logger, self.writer = self._init()

    def _get_version_name(self) -> str:
        """
        自动生成版本名称
        
        Returns:
            str: 版本名称，格式为'version_X'
        """
        version_pattern = re.compile(r'version_(\d+)$')
        if not self.work_path.exists():
            version_name = 'version_1'
        else:
            version_files = [f for f in self.work_path.glob('*') if version_pattern.match(f.name)]
            version_list = []
            for f in version_files:
                match = re.match(version_pattern, f.name)
                if match:
                    version_list.append(int(match.group(1)))
            
            version_number = 1 if len(version_list) == 0 else max(version_list) + 1
            version_name = f'version_{version_number}'
        return version_name

    def _init(self) -> Tuple[Any, SummaryWriter]:
        """
        初始化目录结构和日志记录器
        
        Returns:
            Tuple[logger, SummaryWriter]: 日志记录器和TensorBoard写入器
        """
        # 创建目录结构
        self.work_path.mkdir(exist_ok=True, parents=True)
        self.backup_path.mkdir(exist_ok=True, parents=True)
        self.logfiles_path.mkdir(exist_ok=True, parents=True)
        self.checkpoints_path.mkdir(exist_ok=True, parents=True)

        # 配置日志记录器
        logfile_name = self.logfiles_path / "train.log"
        logger.add(
            sink=logfile_name,
            colorize=True,
            format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{file}|{function}|{line}|{message}",
            rotation="10 MB",
            retention="10 days"
        )

        # 初始化TensorBoard写入器
        writer = SummaryWriter(str(self.logfiles_path))

        # 加载已有的指标历史
        if self.metrics_filepath.exists():
            try:
                metrics_data = pd.read_csv(self.metrics_filepath).to_dict(orient='list')
                # 确保所有值都是列表
                for key, value in metrics_data.items():
                    if not isinstance(value, list):
                        metrics_data[key] = [value]
                self.metrics_history = {str(k): v for k, v in metrics_data.items()}
            except Exception as e:
                logger.warning(f"Failed to load existing metrics: {e}")
                self.metrics_history = {}

        return logger, writer

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        记录训练指标
        
        Args:
            metrics: 指标字典，包含指标名称和值
        """
        if not metrics:
            return
            
        # 获取当前历史记录的最大长度
        history_max_length = max((len(lst) for lst in self.metrics_history.values()), default=0)
        col_fill = [] if history_max_length == 0 else [None] * history_max_length
        
        # 更新指标历史
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = col_fill + [value]
            else:
                self.metrics_history[name].append(value)
            
            # 记录到TensorBoard（排除epoch和iter）
            if name not in ['epoch', 'iter']:
                self.writer.add_scalar(name, value, history_max_length + 1)

        # 保存到CSV文件
        try:
            metrics_history = self.metrics_history.copy()
            max_length = max(len(lst) for lst in metrics_history.values())
            
            # 填充较短的列表
            for metric in metrics_history.keys():
                if len(metrics_history[metric]) < max_length:
                    metrics_history[metric].extend([None] * (max_length - len(metrics_history[metric])))

            df = pd.DataFrame(metrics_history)
            df.to_csv(self.metrics_filepath, index=False)
        except Exception as e:
            logger.error(f"Failed to save metrics to CSV: {e}")

    def print_table(self,data: Dict[str, Any],show_index: Optional[List[str]] = None) -> None:
        """
        以表格形式打印数据
        
        Args:
            data: 要打印的数据字典
            show_index: 可选的索引列
        """
        table = PrettyTable()
        
        # 设置表格样式
        table.horizontal_char = '─'
        table.vertical_char = '│'
        table.top_junction_char = '┬'
        table.bottom_junction_char = '┴'
        table.bottom_left_junction_char = '└'
        table.bottom_right_junction_char = '┘'
        table.left_junction_char = '├'
        table.right_junction_char = '┤'
        table.top_left_junction_char = '┌'
        table.top_right_junction_char = '┐'
        table.junction_char = '┼'
        table.align = 'l'
        
        # 添加索引列
        if isinstance(show_index, list) and show_index:
            if len(show_index) == len(list(data.values())[0]) if data else 0:
                table.add_column('class', show_index)
            else:
                logger.warning("Index column length doesn't match data column length")
        
        # 处理数据
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, (list, np.ndarray)):
                processed_data[k] = v
            else:
                processed_data[k] = [v]
        
        # 添加列
        for k, v in processed_data.items():
            table.add_column(k, v)
        
        self.logger.info(f'\n{table}')

    def backup_config(self, config: Dict[str, Any]) -> None:
        """
        备份配置文件
        
        Args:
            config: 配置字典
        """
        try:
            backup_file = self.backup_path / "config_backup.py"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write("# Configuration Backup\n")
                f.write(f"# Generated at: {pd.Timestamp.now()}\n\n")
                
                def _write_dict(d: Dict[str, Any], indent: int = 0) -> None:
                    for key, value in d.items():
                        if isinstance(value, dict):
                            f.write(f"{'  ' * indent}{key} = {{\n")
                            _write_dict(value, indent + 1)
                            f.write(f"{'  ' * indent}}}\n")
                        else:
                            value_str = repr(value)
                            f.write(f"{'  ' * indent}{key} = {value_str}\n")
                
                _write_dict(config)
            
            self.logger.info(f"Configuration backed up to {backup_file}")
        except Exception as e:
            self.logger.error(f"Failed to backup configuration: {str(e)}")

    def backup_model_code(self, model: nn.Module) -> None:
        """
        备份模型代码
        
        Args:
            model: PyTorch模型
        """
        try:
            model_file = inspect.getfile(model.__class__)
            source_path = Path(model_file).resolve()
            
            if source_path.exists():
                backup_path = self.backup_path / "model_backup.py"
                shutil.copy2(source_path, backup_path)
                self.logger.info(f"Model code file backed up to {backup_path}")
            else:
                self.logger.warning(f"Model source file not found: {source_path}")
        except Exception as e:
            self.logger.error(f"Failed to backup model code: {str(e)}")

    @property
    def metrics_result(self) -> pd.DataFrame:
        """
        获取指标结果的DataFrame
        
        Returns:
            pd.DataFrame: 包含所有指标历史的DataFrame
        """
        return pd.DataFrame(self.metrics_history)

    @property
    def metrics_avg(self, metric: Optional[str] = None) -> Dict[str, float]:
        """
        获取指标的平均值
        
        Args:
            metric: 指标名称，如果为None则返回所有指标的平均值
            
        Returns:
            Dict[str, float]: 指标平均值字典
            
        Raises:
            KeyError: 当指定的指标不存在时
        """
        df = pd.DataFrame(self.metrics_history)
        if metric is None:
            return df.mean().to_dict()
        elif metric in df.columns:
            return {metric: df[metric].mean()}
        else:
            raise KeyError(f"metric '{metric}' is not in the DataFrame columns: {df.columns}")

    @property
    def metrics_last(self, metric: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指标的最后一个值
        
        Args:
            metric: 指标名称，如果为None则返回所有指标的最后一个值
            
        Returns:
            Dict[str, Any]: 指标最后一个值字典
            
        Raises:
            KeyError: 当指定的指标不存在时
        """
        df = pd.DataFrame(self.metrics_history)
        if df.empty:
            return {}
            
        if metric is None:
            return df.iloc[-1].to_dict()
        elif metric in df.columns:
            return {metric: df[metric].iloc[-1]}
        else:
            raise KeyError(f"metric '{metric}' is not in the DataFrame columns: {df.columns}")

    def metrics_best(self, metric: str, mode: str = 'max') -> Dict[str, Any]:
        """
        获取指标的最佳值
        
        Args:
            metric: 指标名称
            mode: 模式，'max'表示最大值，'min'表示最小值
            
        Returns:
            Dict[str, Any]: 包含最佳值和索引的字典
            
        Raises:
            ValueError: 当模式不是'max'或'min'时
            KeyError: 当指定的指标不存在时
        """
        df = pd.DataFrame(self.metrics_history)
        if metric not in df.columns:
            raise KeyError(f"metric '{metric}' is not in the DataFrame columns: {df.columns}")
            
        if mode == 'max':
            max_value = df[metric].max()
            max_index = df[metric].idxmax()
            return {metric: max_value, 'index': int(max_index) + 1}
        elif mode == 'min':
            min_value = df[metric].min()
            min_index = df[metric].idxmin()
            return {metric: min_value, 'index': int(min_index) + 1}
        else:
            raise ValueError("Mode must be either 'max' or 'min'.")

    def tracker_finish(self) -> None:
        """完成跟踪，关闭TensorBoard写入器"""
        try:
            self.writer.close()
            logger.info(f"Tracker finished successfully for {self.name} v{self.version_name}")
        except Exception as e:
            logger.error(f"Error while finishing tracker: {e}")

if __name__ == '__main__':
    # 示例用法
    tracker = Tracker(path='results', name='Unet')
    
    # 模拟训练过程
    for i in range(20):
        if i % 3 == 0:
            tracker.log_metrics({
                'train_loss': 0.1,
                'train_acc': 0.9,
                'val_loss': 0.1 * (i + 1),
                'val_acc': 0.9,
                'epoch': i + 1
            })
        else:
            tracker.log_metrics({
                'train_loss': 0.2,
                'train_acc': 0.8,
                'epoch': i + 1
            })
    
    # 获取最佳指标
    best_score = tracker.metrics_best('val_loss', 'min')
    print(f"Best validation loss: {best_score}")
    
    # 打印表格
    tracker.print_table({
        'model': ['UNet', 'DeepLab', 'PSPNet'],
        'miou': [0.75, 0.78, 0.77],
        'pixel_acc': [0.95, 0.96, 0.955]
    }, show_index=['seg', 'seg', 'seg'])
    
    # 完成跟踪
    tracker.tracker_finish()
    logger.success("Tracker finished successfully")
