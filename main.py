from configurations import Configs
from utils import *
from models import *
from KRD import KRD
import torch.utils.data as da


C = Configs()
A = C.ACCESS

dataset = KRD(C.DATA)
train_size = int(C.DATA["train ratio"] * len(dataset))
val_size = len(dataset) - train_size
torch.manual_seed(C.DATA["seed"])
train_data, val_data = da.random_split(dataset, [train_size, val_size])
val_train_data = da.Subset(dataset=train_data, indices=range(750))
val_loader = da.DataLoader(dataset=val_data, batch_size=C.TRAINING["batch size"])
val_train_loader = da.DataLoader(dataset=val_train_data,
                                 batch_size=C.TRAINING["batch size"])

unet = instantiate_unet(A["model config"])
path = "Logging{}/WeightsEpoch{}.pth".format(A["model name"], A["epoch"])
unet.load_state_dict(torch.load(path))


def predict():
    generate_mask(unet, dataset[A["sample"]], A["model name"])


# def compare():
#     compare_three_models()


def visualize():
    visualize_attention(unet, dataset[A["sample"]], A["model name"])


def validate(metric=iou):
    c, t, v = [], [], []
    for i in range(20):
        with torch.no_grad():
            unet1 = instantiate_unet(A["model config"])
            path1 = "Logging{}/WeightsEpoch{}.pth".format(A["model name"], i)
            unet1.load_state_dict(torch.load(path1))
            unet1.eval()
            disable_running(unet1)
            val_metric = evaluate_model(unet1, val_loader, metric, report=True)
            v.append(val_metric)
            train_metric = evaluate_model(unet1, val_train_loader, metric, report=True)
            t.append(train_metric)
            c.append(0)
    plot_progression(c, v, t, "temp")


def evaluate(metric):
    unet.eval()
    disable_running(unet)
    evaluate_model(unet, val_loader, metric, report=True)


if __name__ == "__main__":
    # compare()
    # validate()
    # evaluate(percentage_one_piece)
    # evaluate(iou)
    # evaluate(batch_hausdorff)
    # evaluate(dice_coefficient)
    # predict()
    visualize()
