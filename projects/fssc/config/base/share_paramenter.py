# Description: This file contains the shared parameters for the project.
# data parameters
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]
grid_shape = [256, 256, 32]

# model parameters
voxel_channel = 32

# loss parameters
ignore_index = 255
free_index = 0
num_classes = 20

semantic_kitti_class_frequencies = [
    5.41773033e09,
    1.57835390e07,
    1.25136000e05,
    1.18809000e05,
    6.46799000e05,
    8.21951000e05,
    2.62978000e05,
    2.83696000e05,
    2.04750000e05,
    6.16887030e07,
    4.50296100e06,
    4.48836500e07,
    2.26992300e06,
    5.68402180e07,
    1.57196520e07,
    1.58442623e08,
    2.06162300e06,
    3.69705220e07,
    1.15198800e06,
    3.34146000e05,
]

geo_class_weight = [0.446, 0.505]

semantickitti_class_weight = [
    0.446,
    0.603,
    0.852,
    0.856,
    0.747,
    0.734,
    0.801,
    0.796,
    0.818,
    0.557,
    0.653,
    0.568,
    0.683,
    0.560,
    0.603,
    0.530,
    0.688,
    0.574,
    0.716,
    0.786,
]
