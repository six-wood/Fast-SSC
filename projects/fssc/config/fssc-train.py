from mmengine.config import read_base
from mmengine.visualization.vis_backend import LocalVisBackend
from projects.fssc.plugin.vis.wandbvis import WandbBackend
from mmdet3d.visualization import Det3DLocalVisualizer

with read_base():
    from .base.lr import *
    from .base.net import *
    from .base.semankitti import *
    from mmdet3d.configs._base_.default_runtime import *

custom_imports = dict(imports=["projects.fssc.plugin"], allow_failed_imports=False)

default_hooks.update(
    dict(
        logger=dict(type=LoggerHook, interval=10),
        checkpoint=dict(type=CheckpointHook, interval=1),
    )
)


vis_backends = [
    dict(type=LocalVisBackend),
    dict(type=WandbBackend, init_kwargs=dict(project="fssc", name="full-model-mmdet3d")),
]

visualizer = dict(type=Det3DLocalVisualizer, vis_backends=vis_backends, name="visualizer")