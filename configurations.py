from models.AgResUnet import *
from utils import iou_loss
from utils import dice_loss


class Configs:
    """
    Configs class, whose objects are constants, parameters of interest. This
    class for better tuning procedures.
    """
    def __init__(self):
        self.DATA = {"image size": 256,
                     "train ratio": 0.8,
                     "seed": 2222,
                     "debug": 0,
                     "subset": (10, 10),
                     "root dir": "/home/mason2/AGVon1080Ti/Images/Adult/Knee",
                     }

        self.MODEL = {"type": AgResUNet,
                      "root channel": 16,
                      "dropout": 0.1,
                      "batch normalization": True,
                      }

        self.TRAINING = {"epochs": 20,
                         "lr": 0.0001,
                         "l2 reg": 0,
                         "momentum": 0,
                         "batch size": 1,
                         "criterion": iou_loss,
                         }

        self.PATHS = {"model name": "AgResUNet+iouloss",
                      "epoch": 2,
                      "predictions folder": "newPredictions",
                      "sample": 3,
                      }

    def __str__(self):
        s1, s2, s3, s4 = str(self.DATA), str(self.MODEL), str(self.TRAINING), \
            str(self.PATHS)
        return s1 + "\n" + s2 + "\n" + s3 + "\n" + s4 + "\n"
