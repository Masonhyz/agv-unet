from utils import *
from KRD import *
from AgResUnet import *
from AamUNet import *
from configurations import Configs


if torch.cuda.is_available():
    print("CUDA is available. Device:",
          torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")


C = Configs()
# Instantiating my dataset
dataset = KRD('/home/mason2/AGVon1080Ti/Images/Adult/Knee', C.data)
train_size = int(C.data["train ratio"] * len(dataset))
val_size = len(dataset) - train_size
torch.manual_seed(C.data["seed"])
train_data, val_data = da.random_split(dataset, [train_size, val_size])
if C.data["debug"]:
    train_data, val_data = da.Subset(dataset=train_data, indices=range(C.data["subset"][0])), \
        da.Subset(dataset=val_data, indices=range(C.data["subset"][1]))


# Defining training procedures
def train_unet(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = config["criterion"]
    cost_list = []
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
            val_iou_list.append(model.evaluate(val_loader, iou))
            train_iou_list.append(model.evaluate(train_loader, iou))

        # Model state saving
        torch.save(model.state_dict(), '/home/mason2/AGVon1080Ti/{}/newTrainedWeight{}.pth'.format(C.paths["weights folder"], _))
    return cost_list, val_iou_list, train_iou_list


if __name__ == "__main__":
    # Instantiating my_unet
    my_unet = C.model["type"](C.model)
    # Writing model description
    write_description(my_unet, C.paths["description name"])

    # Train my_unet
    cost, val_iou, train_iou = train_unet(my_unet, C.training)
    plot_progression(cost, val_iou, train_iou, C.paths["progression name"])

    # # Load a pretrained weight to a model and show a prediction with it
    # trained_unet = C.model["type"](C.model)
    # weight_path = '{}/newTrainedWeight{}.pth'.format(C.paths["weights folder"], C.paths["epoch"])
    # trained_unet.load_state_dict(torch.load(weight_path))
    # predict(trained_unet, val_data[C.paths["sample"]], C.paths["predictions folder"])
