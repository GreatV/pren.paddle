import paddle
import os
from Nets.model import Model
from Utils.utils import *
from Configs.testConf import configs
import cv2
import numpy as np

transform = paddle.vision.transforms.Compose(
    [
        paddle.vision.transforms.ToPILImage(),
        paddle.vision.transforms.Resize((configs.imgH, configs.imgW)),
        paddle.vision.transforms.ToTensor(),
    ]
)


def imread(imgpath):
    img = cv2.imread(imgpath)
    h, w, _ = img.shape
    x = transform(img)
    x.subtract_(y=paddle.to_tensor(0.5)).divide_(y=paddle.to_tensor(0.5))
    x = x.unsqueeze(axis=0)
    is_vert = True if h > w else False
    if is_vert:
        img_clock = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_counter = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        x_clock = transform(img_clock)
        x_counter = transform(img_counter)
        x_clock.subtract_(y=paddle.to_tensor(0.5)).divide_(y=paddle.to_tensor(0.5))
        x_counter.subtract_(y=paddle.to_tensor(0.5)).divide_(y=paddle.to_tensor(0.5))
        x_clock = x_clock.unsqueeze(axis=0)
        x_counter = x_counter.unsqueeze(axis=0)
    else:
        x_clock, x_counter = 0, 0
    return x, x_clock, x_counter, is_vert


class Recognizer(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        with open(configs.alphabet) as f:
            alphabet = f.readline().strip()
        self.converter = strLabelConverter(alphabet)

    def recog(self, imgpath):
        with paddle.no_grad():
            x, x_clock, x_counter, is_vert = imread(imgpath)
            logits = self.model(x)
            if is_vert:
                x_clock = x_clock
                x_counter = x_counter
                logits_clock = self.model(x_clock)
                logits_counter = self.model(x_counter)
                score, pred = logits[0].log_softmax(1).max(1)
                pred = list(pred.cpu().numpy())
                score_clock, pred_clock = logits_clock[0].log_softmax(1).max(1)
                pred_clock = list(pred_clock.cpu().numpy())
                score_counter, pred_counter = logits_counter[0].log_softmax(1).max(1)
                pred_counter = list(pred_counter.cpu().numpy())
                scores = np.ones(3) * -np.inf
                if 1 in pred:
                    score = score[: pred.index(1)]
                    scores[0] = score.mean()
                if 1 in pred_clock:
                    score_clock = score_clock[: pred_clock.index(1)]
                    scores[1] = score_clock.mean()
                if 1 in pred_counter:
                    score_counter = score_counter[: pred_counter.index(1)]
                    scores[2] = score_counter.mean()
                c = scores.argmax()
                if c == 0:
                    pred = pred[: pred.index(1)]
                elif c == 1:
                    pred = pred_clock[: pred_clock.index(1)]
                else:
                    pred = pred_counter[: pred_counter.index(1)]
            else:
                pred = logits[0].argmax(axis=1)
                pred = list(pred.cpu().numpy())
                if 1 in pred:
                    pred = pred[: pred.index(1)]
            pred = self.converter.decode(pred).replace("<unk>", "")
        return pred


if __name__ == "__main__":
    checkpoint = paddle.load(configs.model_path)
    model = Model(checkpoint["model_config"])
    model.set_state_dict(state_dict=checkpoint["state_dict"])
    print("[Info] Load model from {}".format(configs.model_path))
    tester = Recognizer(model)
    imnames = os.listdir("samples")
    paddle.sort(x=imnames), paddle.argsort(x=imnames)
    impaths = [os.path.join("samples", imname) for imname in imnames]
    for impath in impaths:
        pred = tester.recog(impath)
        print("{}: {}".format(impath, pred))
