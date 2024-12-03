from accelerate import Accelerator
from datasets import load_dataset
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_prefix", type=str,
                        default="~/workspace/")
    parser.add_argument("--dataset_name", type=str,
                        default="invariant/datasets/data/split_celeba", help="waterbirds or celeba")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--path_ckpt", type=str,
                        default="invariant/ckpts/celeba_human/classifier")
    parser.add_argument("--only_test", type=str2bool, default=False)
    parser.add_argument("--only_train_valid", type=str2bool, default=False)
    parser.add_argument("--label_name", type=str, default='Male')
    parser.add_argument("--use_test_to_train", type=str2bool, default=False)
    args = parser.parse_args()
    if 'AMLT_BLOB_DIR' in os.environ:
        # update the path prefix
        args.path_prefix = os.environ['AMLT_BLOB_DIR']
    args.path_ckpt = os.path.join(args.path_prefix, args.path_ckpt,
                                  f'{args.model_name}-{args.batch_size}-{args.learning_rate:.4f}')
    return args


args = parse_args()


def process_train_for_one(single_data):
    image = single_data["image"].convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])  # normalization
    ])
    image = transform(image)
    label = torch.tensor(single_data["place"])
    print(f'image type: {type(image)}, label type: {type(label)}')
    return {"image": image, "label": label}


def process_val_for_one(single_data):
    image = single_data["image"].convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    label = torch.tensor(single_data["place"])
    return {"image": image, "label": label}


def process_train(data):
    images = [image.convert("RGB") for image in data["image"]]
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])  # normalization
    ])
    data["image"] = [transforms_train(image) for image in images]
    # print(data["dataset_name"][0])
    if data["dataset_name"][0] == "waterbird": # place # y
        birds = ['a baird sparrow', 'a bay breasted warbler', 'a black capped vireo','a blue grosbeak',        'a boat tailed grackle',  'a bronzed cowbird',  'a california gull',      'an american redstart',
        'a baltimore oriole',  'a belted kingfisher' ,       'a black tern' ,                  'a blue headed vireo' ,   'a bobolink' ,            'a brown pelican' ,   'a canada warbler' ,      'an anna hummingbird',
        'a bank swallow', 'a black and white warbler',  'a black throated blue warbler', 'a blue jay', 'a brandt cormorant',     'a brown thrasher',   'an acadian flycatcher', 'a barn swallow', 'a black billed cuckoo', 
        'a black throated sparrow', 'a blue winged warbler', 'a brewer blackbird', 'a cactus wren', 'an american goldfinch']
        birds_type = {birds[i]: i for i in range(len(birds))}
        birds_type_label = [birds_type[i] if i in birds_type else len(birds) for i in data['text']]
        data["BirdName"] = birds_type_label
        data["label"] = torch.tensor(data[args.label_name])
    elif data["dataset_name"][0] == "celeba":
        data["Blond_Hair"] = [
            1 if str(i) == "1" else 0 for i in data["Blond_Hair"]]
        data["Male"] = [1 if str(i) == "1" else 0 for i in data["Male"]]
        data['Hair_Color_Blond_Black'] = [
            1 if str(i) == "1" else 0 for i in data['Blond_Hair']]
        data["label"] = torch.tensor(data[args.label_name])
    elif data["dataset_name"][0] == "fairness":
        race_label = {'White': 0, 'Latino_Hispanic': 0, 'Middle Eastern': 0, 'East Asian':1, 'Southeast Asian':1, 'Indian':2, 'Black':3,}
        data["Race"] = [race_label[i] for i in data['race']]
        data["Gender"] = [1 if i == 'Male' else 0 for i in data['gender']]
        data["label"] = torch.tensor(data[args.label_name])
    else:
        raise ValueError("dataset_name must be waterbird or celeba")
    return data


def process_val(data):
    images = [image.convert("RGB") for image in data["image"]]
    transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data["image"] = [transforms_val(image) for image in images]
    if data["dataset_name"][0] == "waterbird":

        birds = ['a baird sparrow', 'a bay breasted warbler', 'a black capped vireo','a blue grosbeak',        'a boat tailed grackle',  'a bronzed cowbird',  'a california gull',      'an american redstart',
        'a baltimore oriole',  'a belted kingfisher' ,       'a black tern' ,                  'a blue headed vireo' ,   'a bobolink' ,            'a brown pelican' ,   'a canada warbler' ,      'an anna hummingbird',
        'a bank swallow', 'a black and white warbler',  'a black throated blue warbler', 'a blue jay', 'a brandt cormorant',     'a brown thrasher',   'an acadian flycatcher', 'a barn swallow', 'a black billed cuckoo', 
        'a black throated sparrow', 'a blue winged warbler', 'a brewer blackbird', 'a cactus wren', 'an american goldfinch']
        birds_type = {birds[i]: i for i in range(len(birds))}
        birds_type_label = [birds_type[i] if i in birds_type else len(birds) for i in data['text']]
        data["BirdName"] = birds_type_label

        data["label"] = torch.tensor(data[args.label_name])
    elif data["dataset_name"][0] == "celeba":
        data["Blond_Hair"] = [
            1 if str(i) == "1" else 0 for i in data["Blond_Hair"]]
        data["Male"] = [1 if str(i) == "1" else 0 for i in data["Male"]]
        data["Hair_Color_Blond_Black"] = [
            1 if str(i) == "1" else 0 for i in data["Blond_Hair"]]
        data["label"] = torch.tensor(data[args.label_name])
    elif data["dataset_name"][0] == "fairness":
        race_label = {'White': 0, 'Latino_Hispanic': 0, 'Middle Eastern': 0, 'East Asian':1, 'Southeast Asian':1, 'Indian':2, 'Black':3,}
        data["Race"] = [race_label[i] for i in data['race']]

        data["Gender"] = [1 if i == 'Male' else 0 for i in data['gender']]
        data["label"] = torch.tensor(data[args.label_name])
    else:
        raise ValueError("dataset_name must be waterbird or celeba")
    return data


def train(model, train_loader, criterion, optimizer, accelerator, args):
    max_steps = len(train_loader)
    progress_bar = tqdm(
        range(0, max_steps),
        desc="train",
        leave=True,)
    model.train()
    for i, batch in enumerate(train_loader):

        x, y = batch["image"], batch["label"]
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        accelerator.backward(loss)
        optimizer.step()
        progress_bar.update(1)
        logs = {"step_loss": loss.detach().item(
        ), "lr": optimizer.param_groups[0]["lr"]}
        progress_bar.set_postfix(**logs)
    progress_bar.refresh()
    progress_bar.close()


def valid(model, valid_loader, criterion, args, epoch=0, prefix="valid"):
    max_steps = len(valid_loader)
    progress_bar = tqdm(
        range(0, max_steps),
        desc=prefix,
        leave=True,)
    model.eval()
    valid_loss = 0
    valid_acc = 0
    ys = []
    y_preds = []
    for i, batch in enumerate(valid_loader):
        x, y = batch["image"], batch["label"]
        y_pred = model(x)
        loss = criterion(y_pred, y)
        valid_loss += loss.item()
        valid_acc += (y_pred.argmax(1) == y).float().mean().item()
        ys.append(y.detach().cpu())
        y_preds.append(y_pred.argmax(1).detach().cpu())
        progress_bar.update(1)
    progress_bar.refresh()
    progress_bar.close()
    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader)
    ys = torch.cat(ys)
    y_preds = torch.cat(y_preds)
    print(f'ys.shape: {ys.shape}, y_preds.shape: {y_preds.shape}')
    print(f"ys: {ys}, y_preds: {y_preds}")
    accuracy, precision, recall, f1, auc = calculate_metrics(ys, y_preds)
    metrics = {"epoch": epoch,
               "valid_loss": valid_loss, "valid_acc": valid_acc,
               "accuracy": accuracy, "precision": precision,
               "recall": recall, "f1": f1, "auc": auc}
    print(f"{prefix}: Epoch {epoch}, loss {valid_loss}, valid acc {valid_acc}, precision {precision}, recall {recall}, f1 {f1}, auc {auc}")
    return metrics


def calculate_metrics(ys, y_preds):
    ys = ys.clone().detach()  # turn the ground truth into a PyTorch tensor
    y_preds = y_preds.clone().detach() 

    # calculate accuracy
    accuracy = accuracy_score(ys.cpu().numpy(), y_preds.cpu().numpy())

    # calculate precision, recall, and F1
    if args.num_classes > 2:
        precision = precision_score(ys.cpu().numpy(),
                                    y_preds.cpu().numpy(), average='macro')
        recall = recall_score(ys.cpu().numpy(),
                            y_preds.cpu().numpy(), average='macro')
        f1 = f1_score(ys.cpu().numpy(), y_preds.cpu().numpy(), average='macro')
    else:
        precision = precision_score(ys.cpu().numpy(), y_preds.cpu().numpy())
        recall = recall_score(ys.cpu().numpy(), y_preds.cpu().numpy())
        f1 = f1_score(ys.cpu().numpy(), y_preds.cpu().numpy())

    # calculate AUC, please note that AUC is only applicable to binary classification problems
    if len(torch.unique(ys)) == 2:
        auc = roc_auc_score(ys.cpu().numpy(), y_preds.cpu().numpy())
    else:
        auc = None
    return accuracy, precision, recall, f1, auc


args = parse_args()
args.dataset_name = os.path.join(args.path_prefix, args.dataset_name)
args.path_ckpt = os.path.join(args.path_prefix, args.path_ckpt)
if not os.path.exists(args.path_ckpt):
    os.makedirs(args.path_ckpt, exist_ok=True)
print(args)
dataset = load_dataset(args.dataset_name)
print('loading dataset success')
# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(type(dataset))
print(type(dataset["train"]))

if args.use_test_to_train:
    dataset_train = dataset['train'].with_transform(process_train)
    dataset_valid = dataset["validation"].with_transform(process_val)
    dataset_test = dataset["validation"].with_transform(process_val)

else:
    dataset_train = dataset['train'].with_transform(process_train)
    dataset_valid = dataset["validation"].with_transform(process_val)
    if "test" in dataset:
        dataset_test = dataset["test"].with_transform(process_val)
    else:
        dataset_test = dataset_valid

print(dataset.keys())
print(dataset_train.column_names)

train_loader = DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(
    dataset_valid, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=False)


if args.model_name == "resnet18":
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
elif args.model_name == "resnet50":
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
elif args.model_name == "resnet101":
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
accelerator = Accelerator()
train_loader, valid_loader, test_loader, model, criterion, optimizer = accelerator.prepare(
    train_loader, valid_loader, test_loader, model, criterion, optimizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'model device: {next(model.parameters()).device}')
print(f'data device: {next(iter(train_loader))["image"].device}')
print(f'accelerator device: {accelerator.device}')


if args.only_test:
    print("only test, load model and test")
    model.load_state_dict(torch.load(
        os.path.join(args.path_ckpt, "model.pth")))
    model.eval()
    print("load model success")
    print("start test")
    metrics_test = valid(model, test_loader, criterion, args, 0, "test")
    print(f"test: {metrics_test}")
    print("save test metrics")
    with open(os.path.join(args.path_ckpt, "metrics_test.json"), "w") as f:
        import json
        json.dump(metrics_test, f)
    exit(0)


max_metric = 0
max_metrics = {}
for epoch in range(args.num_epoch):
    train(model, train_loader, criterion, optimizer, accelerator, args)
    metrics = valid(model, valid_loader, criterion, args, epoch, "valid")
    if metrics["f1"] > max_metric:
        max_metric = metrics["f1"]
        max_metrics = metrics
        torch.save(model.state_dict(), os.path.join(
            args.path_ckpt, "model.pth"))
        print(f"save model at epoch {epoch}, F1 {metrics['f1']}")
        metrics_all = {"valid": metrics}
        with open(os.path.join(args.path_ckpt, "metrics_all.json"), "w") as f:
            import json
            json.dump(metrics_all, f)

        if not args.only_train_valid:
            metrics_test = valid(model, test_loader,
                                criterion, args, epoch, "test")
            merrics_all = {"valid": metrics, "test": metrics_test}
            with open(os.path.join(args.path_ckpt, "merrics_all.json"), "w") as f:
                import json
                json.dump(merrics_all, f)
print(f'max_metrics: {max_metrics}')
