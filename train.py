import torch as torch
from tqdm import tqdm
from utils import *
from KRD import *
from UNet import *
from AgUNet import *
from AamUNet import *
from RAUnet import *


if torch.cuda.is_available():
    print("CUDA is available. Device:",
          torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")

data_config = {
    "image size": 256,
    "train ratio": 0.8,
    "seed": 2222,
    "debug": 0,
    "subset": (10, 10)
}

model_config = {
    "channel": 8,
    "dropout": 0.3,
    "batch normalization": False
}

training_config = {
    "epochs": 30,
    "lr": 0.0001,
    "regularization": 0,
    "momentum": 0,
    "batch size": 1,
    "criterion": nn.BCELoss()
}


# Instantiating my dataset
dataset = KRD('/home/mason2/AGVon1080Ti/Images/Adult/Knee', data_config)
train_size = int(data_config["train ratio"] * len(dataset))
val_size = len(dataset) - train_size
torch.manual_seed(data_config["seed"])
train_data, val_data = da.random_split(dataset, [train_size, val_size])
if data_config["debug"]:
    train_data, val_data = da.Subset(dataset=train_data, indices=range(data_config["subset"][0])), \
        da.Subset(dataset=val_data, indices=range(data_config["subset"][1]))


# Defining training procedures
def train_unet(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = config["criterion"]
    cost_list = []
    # dice_list = []
    val_iou_list = []
    train_iou_list = []
    train_loader = da.DataLoader(dataset=train_data, batch_size=config["batch size"])
    val_loader = da.DataLoader(dataset=val_data, batch_size=config["batch size"])
    for _ in tqdm(range(config["epochs"]), desc="Epoch"):
        # Model Training
        model.train()
        cost = 0
        for __, x, y in tqdm(train_loader, desc="Batch", leave=False):
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            cost += loss.data
        cost_list.append(cost)

        # Model evaluation
        with torch.no_grad():
            model.eval()
            for __, images, targets in tqdm(val_loader, desc="Eval V", leave=False):
                p = model(images)
                predictions = (p > 0.5).float()
                # dice = dice_coefficient(predictions.detach(), targets.detach())
                val_iouu = iou(predictions.detach(), targets.detach())
            # dice_list.append(dice)
            val_iou_list.append(val_iouu)
            for __, images, targets in tqdm(train_loader, desc="Eval T", leave=False):
                p = model(images)
                predictions = (p > 0.5).float()
                tra_iou = iou(predictions.detach(), targets.detach())
            train_iou_list.append(tra_iou)

        # Model state saving
        torch.save(model.state_dict(), '/home/mason2/AGVon1080Ti/newTrainedWeights/newTrainedWeight{}.pth'.format(_))
    return cost_list, val_iou_list, train_iou_list


if __name__ == "__main__":
    # Instantiating my_unet
    my_unet = AamUNet(model_config)

    # Train my_unet
    cost, val_iou, train_iou = train_unet(my_unet, training_config)
    plot_progression(cost, val_iou, train_iou)

    # n = 4
    # # Show prediction with a trained UNet
    # trained_unet = AamUNet(model_config)
    # trained_unet.load_state_dict(torch.load('newTrainedWeights/newTrainedWeight13.pth'))
    # predict('/home/mason/MiDATA/AGV/newPreds',
    #         trained_unet, val_data[n])
