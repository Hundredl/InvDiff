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
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial


def load_classifier_model(model_name, path_ckpt, device, num_class):
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_class)

    ckpt = torch.load(os.path.join(path_ckpt, "model.pth"))
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    return model


def load_pic_dataset(image_path, need_transform=True):
    import torchvision
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if need_transform else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(image_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataset, dataloader


def get_clip_score(image_path,):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset, dataloader = load_pic_dataset(image_path, need_transform=False)
    clip_score_fn = partial(
        clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    scores_all = []
    for data in tqdm(dataloader):
        images = data[0].to(device)
        images = (images * 255).type(torch.uint8)
        targets = data[1]
        prompts = [dataset.classes[i] for i in targets]
        score = clip_score_fn(images, prompts).detach().cpu().numpy()
        scores_all.append(score.item())
    mean_score = np.mean(scores_all)
    std_score = np.std(scores_all)
    return {
        'clip_scores': scores_all,
        'clip_score': mean_score,
        'std_score': std_score,
        'clip_score_mean_std': f"{mean_score:.4f}({std_score:.4f})"
    }


def get_clip_score_using_classifier(image_path, classifier_path, num_class, prompt_for_class):

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset, dataloader = load_pic_dataset(image_path, need_transform=True)
    classifier = load_classifier_model(
        "resnet18", classifier_path, device, num_class)

    prompts_all = []
    for data in tqdm(dataloader):
        images = data[0].to(device)
        pred = classifier(images.to(device))
        pred_labels = torch.argmax(pred, dim=1).detach().cpu().numpy()
        prompts_all.append([prompt_for_class[i] for i in pred_labels])

    dataset_no_transform, dataloader_no_transform = load_pic_dataset(
        image_path, need_transform=False)
    clip_score_fn = partial(
        clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    scores_all = []
    for index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = data[0].to(device)
        images = (images * 255).type(torch.uint8)
        prompts = prompts_all[index]
        score = clip_score_fn(images, prompts).detach().cpu().numpy()
        scores_all.append(score.item())
    mean_score = np.mean(scores_all)
    std_score = np.std(scores_all)
    return {
        'clip_scores': scores_all,
        'clip_score': mean_score,
        'std_score': std_score,
        'clip_score_mean_std': f"{mean_score:.4f}({std_score:.4f})"
    }

