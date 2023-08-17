from configurations import Configs
from utils import *
from models import *
from KRD import KRD
import torch.utils.data as da


C = Configs()
A = C.ACCESS
dataset = KRD(C.DATA)
trained_unet = UNet({"root channel": 16, "batch normalization": True, "dropout": 0})
weight_path = 'Weights{}/newTrainedWeight{}.pth'.format("UNet", 24)
trained_unet.load_state_dict(torch.load(weight_path))


def predict(name):
    # trained_unet.eval()
    generate_mask(trained_unet, dataset.get(name), "UNet")
    # generate_mask(trained_unet, dataset.get('11_0.png'), "AgUNet")
    # generate_mask(trained_unet, dataset.get('749_88.png'), "AgUNet")
    # generate_mask(trained_unet, dataset.get('889_46.png'), "AgUNet")
    # generate_mask(trained_unet, dataset.get('762_23.png'), "AgUNet")




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


def validate():
    train_size = int(C.DATA["train ratio"] * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(C.DATA["seed"])
    train_data, val_data = da.random_split(dataset, [train_size, val_size])
    val_train_data = da.Subset(dataset=train_data, indices=range(750))
    if C.DATA["debug"]:
        train_data, val_data, val_train_data = \
            da.Subset(dataset=train_data, indices=range(C.DATA["subset"][0])), \
                da.Subset(dataset=val_data, indices=range(C.DATA["subset"][1])), \
                da.Subset(dataset=train_data, indices=range(10))
    val_loader = da.DataLoader(dataset=val_data, batch_size=C.TRAINING["batch size"])
    val_train_loader = da.DataLoader(dataset=val_train_data,
                                     batch_size=C.TRAINING["batch size"])
    c, t, v = [], [], []
    for i in tqdm(range(24, 25)):
        unet = UNet({"root channel": 16, "batch normalization": True, "dropout": 0})
        path = "WeightsUNet/newTrainedWeight{}.pth".format(i)
        unet.load_state_dict(torch.load(path))
        with torch.no_grad():
            for m in unet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    # m.running_mean = None
                    # m.running_var = None
            unet.eval()
            val_iou = evaluate_model(unet, val_loader)
            v.append(val_iou)
            train_iou = evaluate_model(unet, val_train_loader)
            t.append(train_iou)
            c.append(0)
    plot_progression(c, v, t, "temp2")


def calculate_iou(name):
    trained_unet.eval()
    for m in trained_unet.modules():
        if isinstance(m, nn.BatchNorm2d):
            print(m.running_var, m.running_mean)
    p = trained_unet(dataset.get(name)[1].unsqueeze(0))
    predictions = (p > 0.5).float()
    return iou(predictions.detach(), dataset.get(name)[2].detach().unsqueeze(0))


if __name__ == "__main__":
    # validate()
    print(calculate_iou('1168_20.png'))
    # predict('1168_20.png')
