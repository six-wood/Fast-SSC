import os
from typing import Optional, Sequence, Union

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import BaseVisBackend, force_init_env

@VISBACKENDS.register_module()
class WandbBackend(BaseVisBackend):
    """Wandb visualization backend class.

    Examples:
        >>> from mmengine.visualization import WandbVisBackend
        >>> import numpy as np
        >>> wandb_vis_backend = WandbVisBackend()
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> wandb_vis_backend.add_image('img', img)
        >>> wandb_vis_backend.add_scaler('mAP', 0.6)
        >>> wandb_vis_backend.add_scalars({'loss': [1, 2, 3],'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> wandb_vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer.
        init_kwargs (dict, optional): wandb initialization
            input parameters.
            See `wandb.init <https://docs.wandb.ai/ref/python/init>`_ for
            details. Defaults to None.
        define_metric_cfg (dict or list[dict], optional):
            When a dict is set, it is a dict of metrics and summary for
            ``wandb.define_metric``.
            The key is metric and the value is summary.
            When a list is set, each dict should be a valid argument of
            the ``define_metric``.
            For example, ``define_metric_cfg={'coco/bbox_mAP': 'max'}``,
            means the maximum value of ``coco/bbox_mAP`` is logged on wandb UI.
            When ``define_metric_cfg=[dict(name='loss',
            step_metric='epoch')]``,
            the "loss" will be plotted against the epoch.
            See `wandb define_metric <https://docs.wandb.ai/ref/python/
            run#define_metric>`_ for details.
            Defaults to None.
        commit (bool, optional) Save the metrics dict to the wandb server
            and increment the step.  If false `wandb.log` just updates the
            current metrics dict with the row argument and metrics won't be
            saved until `wandb.log` is called with `commit=True`.
            Defaults to True.
        log_code_name (str, optional) The name of code artifact.
            By default, the artifact will be named
            source-$PROJECT_ID-$ENTRYPOINT_RELPATH. See
            `wandb log_code <https://docs.wandb.ai/ref/python/run#log_code>`_
            for details. Defaults to None.
            `New in version 0.3.0.`
        watch_kwargs (optional, dict): Agurments for ``wandb.watch``.
            `New in version 0.4.0.`
    """

    def __init__(self,
                 save_dir: str,
                 init_kwargs: Optional[dict] = None,
                 define_metric_cfg: Union[dict, list, None] = None,
                 commit: Optional[bool] = True,
                 log_code_name: Optional[str] = None,
                 watch_kwargs: Optional[dict] = None):
        super().__init__(save_dir)
        self._init_kwargs = init_kwargs
        self._define_metric_cfg = define_metric_cfg
        self._commit = commit
        self._log_code_name = log_code_name
        self._watch_kwargs = watch_kwargs if watch_kwargs is not None else {}

    def _init_env(self):
        """Setup env for wandb."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if self._init_kwargs is None:
            self._init_kwargs = {'dir': self._save_dir}
        else:
            self._init_kwargs.setdefault('dir', self._save_dir)
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        wandb.init(**self._init_kwargs)
        if self._define_metric_cfg is not None:
            if isinstance(self._define_metric_cfg, dict):
                for metric, summary in self._define_metric_cfg.items():
                    wandb.define_metric(metric, summary=summary)
            elif isinstance(self._define_metric_cfg, list):
                for metric_cfg in self._define_metric_cfg:
                    wandb.define_metric(**metric_cfg)
            else:
                raise ValueError('define_metric_cfg should be dict or list')
        self._wandb = wandb

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return wandb object.

        The experiment attribute can get the wandb backend, If you want to
        write other data, such as writing a table, you can directly get the
        wandb backend through experiment.
        """
        return self._wandb

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to wandb.

        Args:
            config (Config): The Config object
        """
        assert isinstance(self._init_kwargs, dict)
        allow_val_change = self._init_kwargs.get('allow_val_change', False)
        self._wandb.config.update(
            config.to_dict(), allow_val_change=allow_val_change)
        self._wandb.run.log_code(name=self._log_code_name)

    @force_init_env
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self._wandb.watch(model, **self._watch_kwargs)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to wandb.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
        """
        image = self._wandb.Image(image)
        self._wandb.log({name: image}, commit=self._commit)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to wandb.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
        """
        self._wandb.log({name: value}, commit=self._commit)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        self._wandb.log(scalar_dict, commit=self._commit)

    def close(self) -> None:
        """close an opened wandb object."""
        if hasattr(self, '_wandb'):
            self._wandb.join()