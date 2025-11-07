# The new config inherits a base config to highlight the necessary modification
_base_ = 'mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5), mask_head=dict(num_classes=5)))

# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=35, norm_type=2),  # or add grad_clip here
)

# Modify dataset related settings
data_root = '/root/autodl-tmp/hybrid_publaynet_coco/'
metainfo = {
    'classes': ('text', 'title', 'table', 'list', 'figure'),
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# swanlab
custom_imports = dict(  # 引入SwanLab作为日志记录器
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="SwanlabVisBackend",
        init_kwargs={  # swanlab.init 参数
            "project": "MMDetection",  # 项目名称
            "experiment_name": "mask-rcnn",  # 实验名称
            "description": "mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco",  # 实验的描述信息
        },
    ),
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20)