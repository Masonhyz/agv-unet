from AgResUnet import *


class Configs:

    def __init__(self):
        self.data = {"image size": 256,
                     "train ratio": 0.8,
                     "seed": 2222,
                     "debug": 0,
                     "subset": (10, 10)
                     }

        self.model = {"type": AgResUNet,
                      "root channel": 16,
                      "dropout": 0,
                      "batch normalization": True
                      }

        self.training = {"epochs": 3,
                         "lr": 0.0001,
                         "regularization": 0,
                         "momentum": 0,
                         "batch size": 1,
                         "criterion": nn.BCELoss()
                         }

        self.paths = {"progression name": "newProgression",
                      "description name": "newDescription",
                      "weights folder": "newWeights",
                      "epoch": 2,
                      "predictions folder": "newPredictions",
                      "sample": 3,
                      }
