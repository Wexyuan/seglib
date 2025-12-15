import re
import sys
import yaml
import importlib
from addict import Dict
from pathlib import Path
from typing import Union, Optional, Dict, Any
from argparse import ArgumentParser


class ConfigDict(Dict):
    """
    配置字典类，继承自addict.Dict，提供更友好的属性访问方式。
    
    当访问不存在的键时，会抛出KeyError而不是返回空字典。
    自动将嵌套字典转换为ConfigDict对象，支持链式属性访问。
    """
    
    def __missing__(self, name: str) -> None:
        """当访问不存在的键时抛出KeyError"""
        raise KeyError(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """设置属性值，同时更新字典"""
        self[name] = value

    def __getattr__(self, name: str) -> Any:
        """重写属性访问方法，提供更清晰的错误信息"""
        try:
            value = self[name]
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            raise ex
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """重写设置项方法，自动转换嵌套字典为ConfigDict"""
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super().__setitem__(key, value)
    
    def update(self, *args, **kwargs) -> None:  # type: ignore
        """重写更新方法，确保嵌套字典也被转换"""
        other = dict(*args, **kwargs)
        for key, value in other.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                other[key] = ConfigDict(value)
        super().update(other)


class Config:
    """
    配置管理类，支持从多种来源加载和合并配置。
    
    支持的配置来源：
    1. 命令行参数（优先级最高）
    2. Python字典
    3. 配置文件（YAML或Python格式）
    
    配置文件支持通过_base_字段继承其他配置文件。
    """
    
    def __init__(
        self,
        load_console_args: bool = True,
        parser: Optional[ArgumentParser] = None,
        data_dict: Optional[Dict[str, Any]] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        初始化配置管理器
        
        Args:
            load_console_args: 是否加载命令行参数，默认为True
            parser: 命令行参数解析器
            data_dict: 直接传入的配置字典
            file_path: 配置文件路径
        """
        if load_console_args:
            self._console_dict = self._load_console_dict(parser)
            self._file_path = self._console_dict.get('config_path', file_path)
        else:
            self._console_dict = {}
            self._file_path = file_path
            
        self._data_dict = self._load_data_dict(data_dict)
        self._file_dict = self._load_file_dict(self._file_path)
        self._config_dict = self._merge_config_dict()

    def get_config_dict(self) -> ConfigDict:
        """
        获取最终的配置字典
        
        Returns:
            ConfigDict: 合并后的配置字典
        """
        return ConfigDict(self._config_dict)

    @staticmethod
    def _load_console_dict(parser: Optional[ArgumentParser] = None) -> Dict[str, Any]:
        """
        从命令行参数加载配置
        
        Args:
            parser: 命令行参数解析器，如果为None则创建默认解析器
            
        Returns:
            Dict[str, Any]: 命令行参数字典
        """
        if parser is not None:
            current_parser = parser
        else:
            current_parser = ArgumentParser()
        current_parser.add_argument("-c", "--config_path", type=Path, required=False)
        args = current_parser.parse_args()
        return {k: v for k, v in vars(args).items() if v is not None}

    @staticmethod
    def _load_data_dict(data_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理直接传入的配置字典
        
        Args:
            data_dict: 配置字典
            
        Returns:
            Dict[str, Any]: 处理后的配置字典
        """
        config_dict = {}
        if data_dict is not None:
            config_dict.update(data_dict)
        return config_dict

    def _load_file_dict(self, config_file: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """
        根据文件扩展名加载配置文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            ValueError: 当文件扩展名不支持时
        """
        config_dict = {}
        if config_file is not None:
            file_extension = Path(config_file).suffix.lower()
            if file_extension in ['.yaml', '.yml']:
                config_dict = self._load_yaml_file(config_file)
            elif file_extension == '.py':
                config_dict = self._load_py_file(config_file)
            else:
                raise ValueError(f"不支持的文件扩展名: {file_extension}")
        return config_dict

    def _load_yaml_file(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_file: YAML配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 当文件不存在时
            yaml.YAMLError: 当YAML格式错误时
        """
        config_dict = {}
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
                     [-+]?[0-9][0-9_]*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        
        if config_file is not None:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_file}")
                
            with open(config_path, "r", encoding="utf-8") as fin:
                try:
                    yaml_content = yaml.load(fin.read(), Loader=loader)
                    if yaml_content is not None:
                        config_dict.update(yaml_content)
                except yaml.YAMLError as e:
                    raise yaml.YAMLError(f"YAML文件格式错误: {config_file}, 错误信息: {e}")
        
        # 处理配置继承
        if config_dict.get("_base_") is not None:
            src_dict = config_dict.copy()
            _base_files = config_dict.get('_base_')
            if isinstance(_base_files, str):
                _base_files = [_base_files]
            elif _base_files is None:
                _base_files = []
                
            for base_file in _base_files:
                base_dict = self._load_file_dict(base_file)
                config_dict = self._recur_update(base_dict, config_dict)

            config_dict.pop('_base_')
            config_dict = self._recur_update(config_dict, src_dict)
            
        return config_dict

    def _load_py_file(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """
        加载Python配置文件
        
        Args:
            config_file: Python配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 当文件不存在时
            ImportError: 当模块导入失败时
            ValueError: 当文件名包含非法字符时
        """
        file_path = Path(config_file).absolute()
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
            
        module_name = file_path.stem
        if "." in module_name:
            raise ValueError("配置文件路径中不允许包含点号(.).")
            
        config_dir = str(file_path.parent)
        original_path = sys.path.copy()
        
        try:
            sys.path.insert(0, config_dir)
            module = importlib.import_module(module_name)
            config_dict = {name: value for name, value in module.__dict__.items()
                          if not name.startswith("__")}
        except ImportError as e:
            raise ImportError(f"导入Python配置文件失败: {config_file}, 错误信息: {e}")
        finally:
            sys.path = original_path

        # 处理配置继承
        if config_dict.get('_base_') is not None:
            src_dict = config_dict.copy()
            _base_files = config_dict.get('_base_')
            if isinstance(_base_files, str):
                _base_files = [_base_files]
            elif _base_files is None:
                _base_files = []
                
            for base_file in _base_files:
                base_dict = self._load_file_dict(base_file)
                config_dict = self._recur_update(base_dict, config_dict)

            config_dict.pop('_base_')
            config_dict = self._recur_update(config_dict, src_dict)
            
        return config_dict

    def _merge_config_dict(self) -> Dict[str, Any]:
        """
        合并所有配置来源，优先级从低到高：
        1. 文件配置
        2. 直接传入的字典
        3. 命令行参数
        
        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        config_dict = {}
        config_dict = self._update(config_dict, self._file_dict)
        config_dict = self._update(config_dict, self._data_dict)
        config_dict = self._update(config_dict, self._console_dict)
        return self._convert_to_configdict(config_dict)

    @staticmethod
    def _update(dict1: Optional[Dict[str, Any]], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        简单的字典更新，不进行递归合并
        
        Args:
            dict1: 目标字典
            dict2: 源字典
            
        Returns:
            Dict[str, Any]: 更新后的字典
        """
        if dict1 is None:
            dict1 = {}
        for k, v in dict2.items():
            dict1[k] = v
        return dict1

    def _recur_update(self,dict1: Optional[Dict[str, Any]],dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归合并字典，处理嵌套结构
        
        Args:
            dict1: 目标字典
            dict2: 源字典
            
        Returns:
            Dict[str, Any]: 合并后的字典
        """
        if dict1 is None:
            dict1 = {}
        for k, v in dict2.items():
            if isinstance(v, dict):
                dict1[k] = self._recur_update(
                    dict1.get(k) if k in dict1 else None, v
                )
            else:
                dict1[k] = v
        return dict1
    
    def _convert_to_configdict(self, d: Dict[str, Any]) -> ConfigDict:
        """
        递归将普通字典转换为ConfigDict
        
        Args:
            d: 要转换的字典
            
        Returns:
            ConfigDict: 转换后的ConfigDict
        """
        if not isinstance(d, dict):
            return d
        
        config_dict = ConfigDict()
        for k, v in d.items():
            if isinstance(v, dict):
                config_dict[k] = self._convert_to_configdict(v)
            else:
                config_dict[k] = v
        return config_dict

    @property
    def console_dict(self) -> Dict[str, Any]:
        """获取命令行参数字典"""
        return self._console_dict

    @property
    def data_dict(self) -> Dict[str, Any]:
        """获取直接传入的配置字典"""
        return self._data_dict

    @property
    def file_path(self) -> Optional[Union[str, Path]]:
        """获取配置文件路径"""
        return self._file_path

    @property
    def file_dict(self) -> Dict[str, Any]:
        """获取文件配置字典"""
        return self._file_dict

    @property
    def config_dict(self) -> Dict[str, Any]:
        """获取最终合并的配置字典"""
        return self._config_dict
