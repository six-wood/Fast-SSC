from mmengine.config import read_base
from mmengine.visualization.vis_backend import LocalVisBackend
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

# trainval
train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None

test_split.update(dict(ann_file="semantickittiDataset_infos_val.pkl", pipeline=val_pipeline))
test_dataloader.update(dict(dataset=test_split))
test_evaluator = val_evaluator

val_dataloader = None
val_cfg = None
val_evaluator = None

model.update(
    test_cfg=dict(
        save_path="data/semantickitti",
        labels_map_inv=labels_map_inv,
        val=True,
    )
)

vis_backends = [
    dict(type=LocalVisBackend),
]

visualizer = dict(type=Det3DLocalVisualizer, vis_backends=vis_backends, name="visualizer")
