from mmdet.apis import init_detector, inference_detector
from mmdet.models.detectors import BaseDetector
import mmcv

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
img_path='demo/demo.jpg'

img = mmcv.imread(img_path, channel_order='rgb')
result=inference_detector(model,img)
print(result)

from mmdet.registry import VISUALIZERS

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# out_file = './demo/mydemo.jpg'
visualizer.add_datasample('result',
                          img, data_sample=result,
                          draw_gt=False,
                          wait_time=0,
                          pred_score_thr=0.3)

visualizer.show()
