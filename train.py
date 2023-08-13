from KRD import KRD
from configurations import Configs
from tqdm import tqdm
import torch.utils.data as da
from utils import *


if torch.cuda.is_available():
    print("CUDA is available. Device:",
          torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")


C = Configs()
# Instantiating my dataset
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
        da.Subset(dataset=train_data, indices=range(10))


# Defining training procedures
T = C.TRAINING


def train_unet():
    """
    train the unet models configured in configurations
    """
    print("UNet training with configurations: \n{}".format(C))
    my_unet = C.MODEL["type"](C.MODEL)

    # Setting up logging folder
    weights_dir, num = 'Logging{}_'.format(C.MODEL["description"]), 0
    while os.path.isdir(weights_dir + str(num)):
        num += 1
    weights_dir = weights_dir + str(num)
    os.mkdir(weights_dir)
    desc_path = weights_dir + "/Desc" + C.MODEL["description"]
    write_description(my_unet, desc_path)

    # Setting up optimizers and data loaders
    optimizer = torch.optim.Adam(my_unet.parameters(), lr=T["lr"])
    train_loader = da.DataLoader(dataset=train_data, batch_size=T["batch size"])
    val_loader = da.DataLoader(dataset=val_data, batch_size=T["batch size"])
    val_train_loader = da.DataLoader(dataset=val_train_data, batch_size=T["batch size"])

    # Start training!
    cost_list = []
    val_iou_list = []
    train_iou_list = []
    for _ in tqdm(range(T["epochs"]), desc="Epoch"):
        # Set to training mode
        my_unet.train()
        cost = 0
        for __, x, y in tqdm(train_loader, desc="Batch", leave=False):
            optimizer.zero_grad()
            z = my_unet(x)
            loss = T["criterion"](z, y)
            loss.backward()
            optimizer.step()
            cost += loss.data
        cost_list.append(cost)

        # Model evaluation on both training (partial) and validation set
        with torch.no_grad():
            my_unet.eval()
            val_iou = my_unet.evaluate(val_loader, iou)
            val_iou_list.append(val_iou)
            train_iou = my_unet.evaluate(val_train_loader, iou)
            train_iou_list.append(train_iou)

        # Model state saving
        torch.save(my_unet.state_dict(), weights_dir + '/WeightsEpoch{}.pth'.format(_))
        # Reporting epoch
        print("Loss: {}; Val acc: {}".format(cost, val_iou, train_iou))

    # Plot training progress
    plot_path = weights_dir + "/Prog" + C.MODEL["description"]
    plot_progression(cost_list, val_iou_list, train_iou_list, plot_path)


if __name__ == "__main__":
    train_unet()

