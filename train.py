from utils import *
from KRD import *
from configurations import Configs
from tqdm import tqdm


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
if C.DATA["debug"]:
    train_data, val_data = da.Subset(dataset=train_data, indices=range(C.DATA["subset"][0])), \
        da.Subset(dataset=val_data, indices=range(C.DATA["subset"][1]))
val_train_data = da.Subset(dataset=train_data, indices=range(750))


# Defining training procedures
def train_unet(model, config):
    """
    train the unet models given a models and a training configuration
    :param model: an unet models of class UNet
    :param config: training configurations, dict, an attribute of a Configs
    instance
    :return: the cost list, validation IoU list, and training IoU list
    """
    # Setting up optimizers and data loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = config["criterion"]
    train_loader = da.DataLoader(dataset=train_data, batch_size=config["batch size"])
    val_loader = da.DataLoader(dataset=val_data, batch_size=config["batch size"])
    val_train_loader = da.DataLoader(dataset=val_train_data, batch_size=config["batch size"])

    weights_dir = 'Weights{}'.format(C.PATHS["model name"])
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    cost_list = []
    val_iou_list = []
    train_iou_list = []
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
            train_iou_list.append(model.evaluate(val_train_loader, iou))

        # Model state saving
        torch.save(model.state_dict(), weights_dir + '/newTrainedWeight{}.pth'.format(_))
    return cost_list, val_iou_list, train_iou_list


if __name__ == "__main__":
    # Instantiating my_unet
    my_unet = C.MODEL["type"](C.MODEL)
    # Writing models description
    print("UNet training with configurations: \n{}".format(C))
    write_description(my_unet, C.PATHS["model name"])

    # Train my_unet
    cost_list, val_iou, train_iou = train_unet(my_unet, C.TRAINING)
    plot_progression(cost_list, val_iou, train_iou, C.PATHS["model name"])

    # # Load a pretrained weight to a models and show a prediction with it
    # trained_unet = C.models["type"](C.models)
    # weight_path = '{}/newTrainedWeight{}.pth'.format(C.paths["weights folder"], C.paths["epoch"])
    # trained_unet.load_state_dict(torch.load(weight_path))
    # predict(trained_unet, val_data[C.paths["sample"]], C.paths["predictions folder"])
