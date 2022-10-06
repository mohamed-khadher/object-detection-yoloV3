import torch
import cv2
from config import OBJ, test_transforms, ANCHORS
from utils import cells_to_bboxes, non_max_suppression as nms, plot_image
from os import getcwd
from model import YOLOv3
import warnings
warnings.filterwarnings("ignore")


CURRENT_DIR = getcwd() + '\\'
IMAGE_DIR = CURRENT_DIR + 'test_examples/IMG_20221001_091944.jpg'
CHECKPOINT_DIR = CURRENT_DIR + 'checkpoints_models/checkpoint_ep_450.pth.tar'
DEVICE = 'cpu'
#Loading and Initializing MODEL
checkpoint = torch.load(CHECKPOINT_DIR, map_location=DEVICE)
model = YOLOv3(num_classes=4).to(DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#Loading image and prepping it for infering
image = cv2.imread(IMAGE_DIR)
img_shape = image.shape
img = test_transforms(image=image, bboxes = [])
output = model(img['image'].unsqueeze(0).to(DEVICE))
bboxes = [[] for _ in range(1)]

for i in range(3):
    S = output[i].shape[2]
    anchor = torch.tensor([*ANCHORS[i]]).to(DEVICE) * S
    boxes_scale_i = cells_to_bboxes(
        output[i], anchor, S=S, is_preds=True
    )
    for idx, (box) in enumerate(boxes_scale_i):
        bboxes[idx] += box

#OUTPUT from model might yield uncertain bounding boxes
# -> Need to run Non max suppressing to filter useless boxes
anms = nms(bboxes[0], 0.5, .8, 'midpoint')

plot_image(image, anms)
print(f'Image shape is : {img_shape}')
print(anms)