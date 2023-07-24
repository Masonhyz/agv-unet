from models.AgResUnet import *


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
                     "subset": (10, 10)
                     }

        self.MODEL = {"type": ResUNet,
                      "root channel": 16,
                      "dropout": 0,
                      "batch normalization": True
                      }

        self.TRAINING = {"epochs": 20,
                         "lr": 0.0001,
                         "regularization": 0,
                         "momentum": 0,
                         "batch size": 1,
                         "criterion": nn.BCELoss()
                         }

        self.PATHS = {"root dir": "/home/mason2/AGVon1080Ti/Images/Adult/Knee",
                      "progression name": "ProgResUNet",
                      "description name": "DescResUNet",
                      "weights folder": "WeightsResUNet",
                      "epoch": 2,
                      "predictions folder": "newPredictions",
                      "sample": 3,
                      }
