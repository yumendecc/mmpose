# import copy

# x = torch.ones([1, 3, 256, 192], device=device)
# x = torch.ones([1, 3, 224, 224], device=device)
import cv2
import torch
from mmcv import Config

from mmpose.models import build_posenet

cfg_file = 'configs/body/2d_kpt_sview_rgb_img/' \
           'topdown_heatmap/coco/hrformer_small3_coco_256x192.py'
src_file = 'models/hrt_small_coco_256x192.pth'
out_file = src_file.replace('hrt', 'hrformer')

cfg = Config.fromfile(cfg_file)
model = cfg.model
model = build_posenet(model)

out_dict = torch.load(out_file)

# try to load
tmp = model.load_state_dict(out_dict)
print(tmp)

device = torch.device('cuda')
model = model.to(device)
model.eval()
# x = cv2.imread('/home/wzeng/mycodes/mmpose_mine/vis/0_img.png')
x = cv2.imread(
    '/home/wzeng/mycodes/mmpose_mine/tests/data/coco/000000000785.jpg')
x = cv2.resize(x, [256, 192])
x = torch.tensor(x).float().to(device).permute(2, 0, 1)[None, ...]
x = x / 255.0 - 0.5
x = x.expand([2, -1, -1, -1])
y = model.backbone(x)
z = model.keypoint_head(y)
