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


def load_pic_dataset(image_path, need_transform=True, batch_size=8):
    import torchvision
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if need_transform else None
    dataset = datasets.ImageFolder(image_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader


def calculate_debias_score(K, prompts_count, prompt_indexs, pred_results):
    # calculate a matrix, unique prompts * K,
    matrix = np.zeros((prompts_count, K))
    generate_nums = len(prompt_indexs) / prompts_count
    for (prompt_index, pred_result) in zip(prompt_indexs, pred_results):
        prompt_index = int(prompt_index)
        matrix[prompt_index][pred_result] += 1
    print(f'matrix: {matrix}')
    bias_p = []
    for count in matrix:
        bias_cur = 0
        for i in range(K):
            for j in range(i+1, K):
                bias_cur += np.abs(count[i] - count[j])
        print(f'sum(count): {sum(count)}')
        bias_p.append(bias_cur / sum(count))

    bias_p = np.array(bias_p) / (K * (K-1) / 2)
    # bias_p /= generate_nums
    return bias_p


def get_bias_score(image_path, classifier_path, num_class=2):
    print(f'get_bias_score: {image_path}, {classifier_path}, {num_class}')
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset, dataloader = load_pic_dataset(image_path, need_transform=True)
    model = load_classifier_model(
        "resnet18", classifier_path, device, num_class)
    pred_image_labels = []
    for data in dataloader:
        image = data[0].detach().cpu()
        pred = model(image.to(device))
        pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
        pred_image_labels.extend(pred_label)

    imgs_class = np.array(dataset.imgs)[:, 1]

    # calculate score
    prompts_count = len(dataset.classes)
    prompt_indexs = np.array(dataset.imgs)[:, 1]
    pred_results = pred_image_labels
    bias_p = calculate_debias_score(
        num_class, prompts_count, prompt_indexs, pred_results)
    mean_score = bias_p.mean()
    std_score = bias_p.std()
    return {
        'bias_scores': list(bias_p),
        'bias_score': mean_score,
        'std_score': std_score,
        'bias_score_mean_std': f"{mean_score:.4f}({std_score:.4f})"
    }


def get_bias_score_2classifier(image_path, classifier_paths, num_classes=[2, 2]):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device: {device}')
    dataset, dataloader = load_pic_dataset(image_path, need_transform=True)
    labels_all = []
    for classifier_path, num_class in zip(classifier_paths, num_classes):
        print(f'load classifier: {classifier_path}')
        model = load_classifier_model(
            "resnet18", classifier_path, device, num_class)
        pred_image_labels = []
        for data in dataloader:
            image = data[0].detach().cpu()
            pred = model(image.to(device))
            pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
            pred_image_labels.extend(pred_label)
            # print(pred_label)
        labels_all.append(pred_image_labels)
        print(f'sum pred_image_labels: {sum(pred_image_labels)}')

    # show labels_all
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(labels_all)

    # 统计 0_0, 0_1, 1_0, 1_1 的数量
    df = df.T
    print(df.shape)
    df.columns = ['classifier1', 'classifier2']
    df['classifier1'] = df['classifier1'].astype(int)
    df['classifier2'] = df['classifier2'].astype(int)
    df['sub_group'] = df['classifier1'].astype(
        str) + '_' + df['classifier2'].astype(str)
    df['sub_group'] = df['sub_group'].astype(str)
    print(df['sub_group'].value_counts())

    # calculate score
    prompt_count = num_classes[0]
    prompt_indexs = df['classifier1'].values
    pred_results = df['classifier2'].values
    print(f'sum prompt_indexs: {sum(prompt_indexs)}')
    print(f'sum pred_results: {sum(pred_results)}')
    print(f'prompt_indexs: {prompt_indexs}')
    print(f'pred_results: {pred_results}')
    bias_p = calculate_debias_score(
        num_classes[-1], prompt_count, prompt_indexs, pred_results)
    mean_score = bias_p.mean()
    std_score = bias_p.std()
    return {
        'bias_scores': list(bias_p),
        'bias_score': mean_score,
        'std_score': std_score,
        'bias_score_mean_std': f"{mean_score:.4f}({std_score:.4f})"
    }


def get_labels_by_classifier(image_path, classifier_path, num_class=2, batch_size=256):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device: {device}')
    dataset, dataloader = load_pic_dataset(image_path, need_transform=True, batch_size=batch_size)

    print(f'load classifier: {classifier_path}')
    model = load_classifier_model(
        "resnet18", classifier_path, device, num_class)
    pred_image_labels = []
    from tqdm import tqdm
    for data in tqdm(dataloader, total=len(dataloader)):
        image = data[0].detach().cpu()
        pred = model(image.to(device))
        pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
        pred_image_labels.extend(pred_label)
        # break
    # return file names and labels
    return np.array(dataset.imgs)[:, 0], pred_image_labels

if __name__ == "__main__":
    classifier_paths = [
        '~/workspace/blob/drive/invariant/ckpts/celeba/classifier_blond/resnet18-64-0.0001',
        '~/workspace/blob/drive/invariant/ckpts/celeba/classifier_more/resnet18-64-0.0001',
    ]
    image_path = '~/workspace/invariant/baseline/'
    num_classes = [2, 2]
    res = get_bias_score_2classifier(image_path, classifier_paths, num_classes)
    print(res)
