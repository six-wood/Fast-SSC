from mmengine.optim.optimizer.amp_optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import OneCycleLR
from torch.optim.adamw import AdamW

# optimizer
# This schedule is mainly used on Semantickitti dataset in segmentation task

epoch_max = 40
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=(0.01),
        eps=0.000005,
    ),
)

param_scheduler = [
    dict(
        type=OneCycleLR,
        total_steps=epoch_max,
        by_epoch=True,
        eta_max=1e-3,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
        convert_to_iter_based=True,
    )
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=epoch_max, val_interval=1)
val_cfg = dict()
test_cfg = dict()


auto_scale_lr = dict(enable=False, base_batch_size=2)

randomness = dict(seed=10)
