from configurations import Configs
from utils import *
from models import *
from KRD import KRD


C = Configs()
A = C.ACCESS
dataset = KRD(C.DATA)
trained_unet = A["type"](C.MODEL)
weight_path = 'Weights{}/newTrainedWeight{}.pth'.format(A["model name"], A["epoch"])
trained_unet.load_state_dict(torch.load(weight_path))


def predict():
    generate_mask(trained_unet, dataset[A["sample"]], A["model name"])


# Compare three models
def compare():
    trained1 = UNet(C.MODEL)
    trained1.load_state_dict(torch.load("/home/mason2/AGVon1080Ti/WeightsUNet/newTrainedWeight25.pth"))
    trained2 = AgUNet({"type": AgResUNet,
                      "root channel": 8,
                      "dropout": 0,
                      "batch normalization": True,
                      })
    trained2.load_state_dict(torch.load("/home/mason2/AGVon1080Ti/WeightsAgUNet/newTrainedWeight26.pth"))
    trained3 = AgResUNet(C.MODEL)
    trained3.load_state_dict(torch.load("/home/mason2/AGVon1080Ti/WeightsAgResUNet2/newTrainedWeight16.pth"))
    compare_three_models(trained1, trained2, trained3, dataset[A["sample"]])


def visualize():
    visualize_attention(trained_unet, dataset[A["sample"]])


if __name__ == "__mian__":
    predict()
