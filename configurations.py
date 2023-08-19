from models import *


class Configs:
    """
    Configs class, whose objects are constants, parameters of interest. This
    class for better tuning procedures.
    """
    def __init__(self):
        self.DATA = {"image size": (256, 256),
                     "train ratio": 0.8,
                     "seed": 2222,
                     "augmentation": False,
                     "debug": 0,
                     "subset": (3, 3),
                     "root dir": "/home/mason2/AGVon1080Ti/Images/Adult/Knee",
                     }

        self.MODEL = {"description": "UNet_1",
                      "type": AgResUNet,
                      "root channel": 16,
                      "dropout": 0.,
                      "batch normalization": True,
                      }

        self.TRAINING = {"epochs": 20,
                         "lr": 0.0001,
                         "ridge reg": 0,
                         "momentum": 0,
                         "batch size": 1,
                         "criterion": nn.BCELoss(),
                         }

        self.ACCESS = {"model name": "AgResUNetBNNoEval()Dropout0.3_0",
                       "model config": self.MODEL,
                       "epoch": 19,
                       "sample": 30,
                       }

    def __str__(self):
        """
        :return: the string of data, model and training configs, excluding the
        access
        """
        s1 = str(self.DATA)
        s2 = str(self.MODEL)
        s3 = str(self.TRAINING)
        return s1 + "\n" + s2 + "\n" + s3 + "\n"
