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
if C.DATA["debug"]:
    train_data, val_data, val_train_data = \
        da.Subset(dataset=train_data, indices=range(C.DATA["subset"][0])), \
        da.Subset(dataset=val_data, indices=range(C.DATA["subset"][1])),\
        da.Subset(dataset=train_data, indices=range(3))
val_loader = da.DataLoader(dataset=val_data, batch_size=C.TRAINING["batch size"])
val_train_loader = da.DataLoader(dataset=val_train_data,
                                 batch_size=C.TRAINING["batch size"])


unet = instantiate_unet(A["model config"])
path = "Logging{}/WeightsEpoch{}.pth".format(A["model name"], A["epoch"])
unet.load_state_dict(torch.load(path))


def predict():
    generate_mask(unet, dataset[A["sample"]], A["model name"])


def compare():
    trained1 = UNet({"root channel": 16, "batch normalization": False, "dropout": 0})
    path1 = "LoggingUNetDropoutNoBN_0/WeightsEpoch16.pth"
    trained1.load_state_dict(torch.load(path1))
    trained2 = AgUNet({"root channel": 16, "batch normalization": True, "dropout": 0})
    path2 = "LoggingAgUNetNoEval()_0/WeightsEpoch19.pth"
    trained2.load_state_dict(torch.load(path2))
    trained3 = AgResUNet({"root channel": 16, "batch normalization": True, "dropout": 0})
    path3 = "LoggingAgResUNetBNNoEval()Dropout0.3_0/WeightsEpoch39.pth"
    trained3.load_state_dict(torch.load(path3))
    trained4 = ResUNet({"root channel": 16, "batch normalization": True, "dropout": 0})
    path4 = "LoggingResUNetBNNoEval()_0/WeightsEpoch19.pth"
    trained4.load_state_dict(torch.load(path4))
    # compare_four_models(trained1, trained2, trained4, trained3, [dataset[234], dataset[7], dataset[76], dataset[765]])
    compare_four_models(trained1, trained2, trained4, trained3, [dataset[234], dataset[765], dataset[65], dataset[2991]])


def visualize():
    trained2 = AgUNet(
        {"root channel": 16, "batch normalization": True, "dropout": 0})
    path2 = "LoggingAgUNetNoEval()_0/WeightsEpoch19.pth"
    trained2.load_state_dict(torch.load(path2))
    trained3 = AgResUNet(
        {"root channel": 16, "batch normalization": True, "dropout": 0})
    path3 = "LoggingAgResUNetBNNoEval()Dropout0.3_0/WeightsEpoch39.pth"
    trained3.load_state_dict(torch.load(path3))
    compare_attention(trained2, trained3, dataset[A["sample"]])


def validate(metric=iou):
    c, t, v = [], [], []
    for i in tqdm(range(40), desc="Batch"):
        if i % 2 != 0:
            with torch.no_grad():
                unet1 = instantiate_unet(A["model config"])
                path1 = "Logging{}/WeightsEpoch{}.pth".format(A["model name"], i)
                unet1.load_state_dict(torch.load(path1))
                unet1.eval()
                disable_running(unet1)
                val_metric = evaluate_model(unet1, val_loader, metric, report=True)
                v.append(val_metric.item())
                train_metric = evaluate_model(unet1, val_train_loader, metric, report=True)
                t.append(train_metric.item())
                cost = 0
                for __, image, mask in tqdm(val_train_loader, desc="Eval"):
                    criterion = nn.BCELoss()
                    cost += criterion(unet1(image), mask).item()
                c.append(cost)
    plot_progression_formal(c, v, t, "AgResUNet")


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
