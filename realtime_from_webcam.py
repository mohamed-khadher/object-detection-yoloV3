import cv2
import torch
import warnings
from config import OBJ, test_transforms, ANCHORS
from utils import cells_to_bboxes, non_max_suppression as nms
from os import getcwd
from model import YOLOv3
warnings.filterwarnings("ignore")

CURRENT_DIR = getcwd() + '\\'
IMAGE_DIR = CURRENT_DIR + 'test_examples/IMG_20221001_091944.jpg'
CHECKPOINT_DIR = CURRENT_DIR + 'checkpoints_models/saved_model.pth.tar'
DEVICE = 'cpu'


#Loading and Initializing MODEL
checkpoint = torch.load(CHECKPOINT_DIR, map_location=DEVICE)
model = YOLOv3(num_classes=4).to(DEVICE)
print('Loading Model ...')
model.load_state_dict(checkpoint)
model.eval()
print('Model ready !')
camera = cv2.VideoCapture(0)

def get_image_with_bboxes(frame):
    w, h, _ = frame.shape
    img = test_transforms(image=frame, bboxes = [])
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
    boxes = nms(bboxes[0], 0.7, .8, 'midpoint')
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = int((box[0] - box[2] / 2) * w)
        upper_left_y = int((box[1] - box[3] / 2) * h)
        rect_width = box[2] * w
        rect_height = box[3] * h
        s=OBJ[int(class_pred)]
        image = cv2.rectangle(frame, (upper_left_x, upper_left_y), (int(upper_left_x + rect_width), int(upper_left_y + rect_height)), (255,0,0), 4)
        image = cv2.putText(image, s, (upper_left_x, upper_left_y)
        , cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2, cv2.LINE_AA)
        return image
#Test with a single picture
img = cv2.imread(IMAGE_DIR)
img = get_image_with_bboxes(img)
while True:
    cv2.imshow('test with a single img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# while True:

#     ret, frame = camera.read()
#     img = get_image_with_bboxes(frame)
#     cv2.imshow('RT Balls Detection', frame)
    

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


camera.release()
cv2.destroyAllWindows()