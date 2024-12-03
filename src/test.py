
import os
import sys
from datasets import load_dataset
import numpy as np
import lpips
from torchvision import models
import torch
from typing import Union, Tuple

import huggingface_hub

from model.epsilon_with_delta import EpsilonWithDelta
from utils.score.bias_score import get_bias_score
from utils.score.clip_score import get_clip_score
def load_pipe(checkpoint_num, model_path, delta=False, delta_ratio=0.5, delta_init0=True):
    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel

    if checkpoint_num == 0:
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        print(f'Loaded diffusion pipeline from CompVis/stable-diffusion-v1-4')
    else:
        if delta:
            pretrained_unet = UNet2DConditionModel.from_pretrained(model_path + f"/checkpoint-{checkpoint_num}/unet_pretrain_unet", torch_dtype=torch.float16)
            delta = UNet2DConditionModel.from_pretrained(model_path + f"/checkpoint-{checkpoint_num}/unet_delta", torch_dtype=torch.float16)
            unet = EpsilonWithDelta(pretrain_unet=pretrained_unet, delta=delta,delta_ratio=delta_ratio, delta_init0=delta_init0,**pretrained_unet.config)
            print(f'Loaded EpsilonWithDelta from {model_path + f"/checkpoint-{checkpoint_num}/unet"}')
        else:
            unet = UNet2DConditionModel.from_pretrained(model_path + f"/checkpoint-{checkpoint_num}/unet", torch_dtype=torch.float16)
            print(f'Loaded UNet from {model_path + f"/checkpoint-{checkpoint_num}/unet"}')
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.float16, safety_checker=None)
        # disable the savechecker
        print(f'Loaded diffusion pipeline from CompVis/stable-diffusion-v1-4')
    pipe.to("cuda")
    print('Moved to cuda')
    return pipe

def sample_result(pipe, dataset, result_path, sample_num, text_num, save=False):
    import random
    import os


    texts = [sample['text'] for sample in dataset['validation']]
    unique_texts = list(set(texts))
    unique_texts.sort()
    unique_texts = []
    for i in range(8):
        unique_texts += texts[i::8]
    sample_texts = unique_texts[0:text_num]
    print(f'total {len(texts)} samples, {len(unique_texts)} unique texts, {len(sample_texts)} sampled texts')
    
    if os.path.exists(result_path):
        os.system(f"rm -rf {result_path}")
    os.makedirs(result_path, exist_ok=True)
    for index, text in enumerate(sample_texts):
        os.makedirs(os.path.join(result_path, text), exist_ok=True)
        for i in range(sample_num):

            image = None
            for j in range(3):
                try:
                    image = pipe(prompt=text).images[0]
                    break
                except Exception as e:
                    print(f"Error: {e}, resample {j+1}, resample {i+1}/{sample_num}")

            if save:
                image.save(f"{result_path}/{text}/{i}_image.png")
        print(f"Finished sampling for {text}, {index}/{text_num}.")
    print("Finished sampling for all texts, save mode: ", save)

def sample_result_batch(pipe, dataset, result_path, sample_num, text_num, batch_size=16 ,save=False, diversity_prompt=False):
    print(f'sample_result_batch: diversity_prompt: {diversity_prompt}')
    import random
    import os

    texts = [sample['text'] for sample in dataset['validation']]
    unique_texts = list(set(texts))
    unique_texts.sort()
    if text_num > len(unique_texts):
        unique_texts = []
        print(f'text_num {text_num} is larger than unique_texts {len(unique_texts)}, use all unique_texts')
        for i in range(8):
            unique_texts += texts[i::8]
    else:
        unique_texts = unique_texts
    sample_texts = unique_texts[0:text_num]

    if diversity_prompt:
        sample_texts = [f'{prompt}, please generate diverse outputs' for prompt in sample_texts]
    
    print(f'total {len(texts)} samples, {len(unique_texts)} unique texts, {len(sample_texts)} sampled texts')
    print(f'prompt: {sample_texts}')

    if os.path.exists(result_path):
        os.system(f"rm -rf {result_path}")
    os.makedirs(result_path, exist_ok=True)

    for index, text in enumerate(sample_texts):
        if save:
            os.makedirs(os.path.join(result_path, text), exist_ok=True)
    # make a list of texts
    text_list = []
    for t in sample_texts:
        text_list += [t] * sample_num
    # there will be incomplete batches, incomplete batches also need to be sampled
    text_num = len(text_list)
    sample_count = 0
    for i in range(0, text_num, batch_size):
        print(f"Sampling for batch {i//batch_size}/{text_num//batch_size}")
        text_batch = text_list[i:i+batch_size]
        images = pipe(prompt=text_batch).images
        sample_count += len(images)
        for j, text in enumerate(text_batch):
            if save:
                images[j].save(f"{result_path}/{text}/{i+j}_image.png")
    print(f"Finished sampling for all texts, save mode: {save}, total samples: {sample_count}")



def sample_result_batch_given_text(pipe, text_list, result_path, batch_size=16 ,save=False):
    import random
    import os

    # there will be incomplete batches, incomplete batches also need to be sampled
    unique_texts = list(set(text_list))
    for index, text in enumerate(unique_texts):
        os.makedirs(os.path.join(result_path, text), exist_ok=True)
    text_num = len(text_list)
    sample_count = 0
    for i in range(0, text_num, batch_size):
        print(f"Sampling for batch {i//batch_size}/{text_num//batch_size}")
        text_batch = text_list[i:i+batch_size]
        images = pipe(prompt=text_batch).images
        sample_count += len(images)
        for j, text in enumerate(text_batch):
            images[j].save(f"{result_path}/{text}/{text}_{i+j}_image.png")
    print(f"Finished sampling for all texts, save mode: {save}, total samples: {sample_count}")

# load images
def get_real_images(path, num_images=10,random=True,normalize=False):
    # read image files and return [n,3,255,255], torch
    import os
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    import numpy as np

     # find all images
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]
    # randomly select num_images
    if random:
        files = np.random.choice(files, num_images)
    files = files[:num_images]
    # read images
    
    images = []
    for i in files:
        image = Image.open(os.path.join(path, i))
        image = transforms.Resize((255, 255))(image)
        image = transforms.ToTensor()(image)
        images.append(image)
    images = torch.stack(images)
    print(f"Read {len(files)} images from {path}, shape: {images.shape}")
    return images

def get_real_images_filelist(path, num_images=10, random=True, with_path=False):
    # read image files and return [n,3,255,255], torch
    import os
    import numpy as np

    # find all images, subfolders are not considered
    files = []
    for root, dirs, fs in os.walk(path):
        for f in fs:
            if f.endswith(".jpg") or f.endswith(".png"):
                if with_path:
                    files.append(os.path.join(root, f))
    
    # randomly select num_images
    print(f"Found {len(files)} images in {path}, choose {num_images} images")
    if random:
        files = np.random.choice(files, num_images)
    else:
        print(f'Choose first {num_images} images')
        files = files[:min(num_images, len(files))]
    return files

def get_real_images_from_filelist(filelist):
    # read image files and return [n,3,255,255], torch
    from PIL import Image
    import torch
    import torchvision.transforms as transforms

    images = []
    for i in filelist:
        image = Image.open(i)
        image = transforms.Resize((255, 255))(image)
        image = transforms.ToTensor()(image)
        images.append(image)
    images = torch.stack(images)
    print(f"Read {len(filelist)} images, shape: {images.shape}")
    return images


# scores LPIPS
def get_lpips_score_inset(images_filelist, loss_fn_vgg=None):
    import torch
    import lpips
    images = []
    for f in images_filelist:
        print(f)
        image = lpips.im2tensor(lpips.load_image(f))
        images.append(image)
    images = torch.stack(images)

    losses = []
    for i in range(images.size(0)):
        for j in range(images.size(0)):
            loss = loss_fn_vgg(images[i], images[j])
            losses.append(loss)
    losses = torch.stack(losses)
    return losses.mean().item()


def get_lpips_score(result_path, sample_num=10, net='vgg'):
    """Evaluate the result using LPIPS

    Args:
        result_path (str): The path of the result. result_path/{text}/{index}_image.png
        sample_num (int, optional): The number of samples for each text. Defaults to 10.
        net (str, optional): The network used in LPIPS. Defaults to 'vgg'. 'alex' is also available.
    """
    lpips_scores = []
    dirs = os.listdir(result_path)
    loss = lpips.LPIPS(net=net)
    for d in dirs:
        # images = get_real_images(os.path.join(result_path, d), num_images=sample_num, random=False, normalize=True)
        file_list = get_real_images_filelist(os.path.join(result_path, d), num_images=sample_num, random=False, with_path=True)
        lpips_score = get_lpips_score_inset(file_list, loss)
        lpips_scores.append(lpips_score)
    mean_score = np.mean(lpips_scores)
    std_score = np.std(lpips_scores)
    print(f"LPIPS score: {mean_score:.4f}({std_score:.4f})")
    return {"lpips_scores": lpips_scores, "lpips_score": mean_score, "lpips_score_std": std_score, "lpips_score_mean_std": f"{mean_score:.4f}({std_score:.4f})"}


# score FID
def get_fid_score_inset(real_images, fake_images):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    score = fid.compute()
    print(f"FID: {score.item()}")
    return score.item()

def get_fid_score(original_path, result_path, sample_num=10):
    """Evaluate the result using FID

    Args:
        original_path (str): The path of the original images
        result_path (str): The path of the result. result_path/{text}/{index}_image.png
        sample_num (int, optional): The number of samples for each text. Defaults to 10.
    """

    filelist_real = get_real_images_filelist(original_path, num_images=sample_num, random=True, with_path=True)
    filelist_fake = get_real_images_filelist(result_path, num_images=sample_num, random=True, with_path=True)   

    real_images = get_real_images_from_filelist(filelist_real)
    fake_images = get_real_images_from_filelist(filelist_fake)

    fid_score = get_fid_score_inset(real_images, fake_images)
    print(f"FID score: {fid_score:.4f}")
    return {"fid_score": fid_score}


# score RECALL
def get_recall_score_inset(real_images, fake_images):
    from utils.score.improved_prd import IPR
    ipr = IPR(32, k = 2, num_samples=10)
    ipr.compute_manifold_ref(real_images)
    metric = ipr.precision_and_recall(fake_images)
    print(f"Recall: {metric.recall:.4f}, Precision: {metric.precision:.4f}")
    return metric.recall, metric.precision

def get_recall_score(original_path, result_path, sample_num=10):
    """Evaluate the result using Recall

    Args:
        original_path (str): The path of the original images
        result_path (str): The path of the result. result_path/{text}/{index}_image.png
        sample_num (int, optional): The number of samples for each text. Defaults to 10.
    """
    
    filelist_real = get_real_images_filelist(original_path, num_images=sample_num, random=False, with_path=True)
    filelist_fake = get_real_images_filelist(result_path, num_images=sample_num, random=False, with_path=True)   
    tip(filelist_fake)
    real_images = get_real_images_from_filelist(filelist_real)
    fake_images = get_real_images_from_filelist(filelist_fake)

    recall_score, precision_score = get_recall_score_inset(real_images, fake_images)
    print(f"Recall score: {recall_score:.4f}")
    return {"recall_score": recall_score, "precision_score": precision_score}

# others
def trans_image_to_embedding(images):
    # input: [n,3,255,255]
    # output: [n,2048]
    import torch
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F

    # inception-v3
    inception_model = torchvision.models.inception_v3(pretrained=True)
    inception_model.eval()
    inception_model.fc = nn.Identity()
    inception_model.AuxLogits.fc = nn.Identity()

    # forward
    with torch.no_grad():
        embeddings = []
        for i in range(images.size(0)):
            image = images[i].unsqueeze(0)
            embedding = inception_model(image)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
    return embeddings


def arg_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    import argparse
    parser = argparse.ArgumentParser(description='Sample and evaluate the result of a model')
    parser.add_argument('--model_base', type=str, default='CompVis/stable-diffusion-v1-4', help='The base model name')
    parser.add_argument('--model_checkpoint_num', type=int, default=0, help='The checkpoint number of the model')
    parser.add_argument('--model_name', type=str, default='diff-waterbird-eiil', help='The model name')
    parser.add_argument('--path_ckpt', type=str, default='invariant/ckpts', help='The checkpoint path')
    parser.add_argument('--path_dataset', type=str, default='invariant/datasets/data/split_waterbirds', help='The dataset path')
    parser.add_argument('--path_result', type=str, default='invariant/results', help='The result path')
    parser.add_argument('--path_result_root', type=str, default='invariant/results', help='The result path')
    parser.add_argument('--path_prefix', type=str, default='~', help='The prefix of the path')

    # sample arguments
    parser.add_argument('--sample_num', type=int, default=5, help='The number of samples for each text')
    parser.add_argument('--text_num', type=int, default=10, help='The number of texts to sample')
    parser.add_argument('--save_sample', type=str2bool, default=True, help='Save the sample result')
    
    # model
    parser.add_argument('--delta', type=str2bool, default=False, help='The delta of the model')
    parser.add_argument(
        "--lambda_value",
        type=float,
        default=1,
        help="The invariant loss weight.",
    )
    parser.add_argument(
        "--delta_ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--delta_init0",
        type=str2bool,
        default=True,
        help="Whether to initialize delta to 0.",
    )
    # score basemodel
    parser.add_argument('--lpips_base', type=str, default='vgg', help='The base model of LPIPS, vgg or alex')

    # task
    parser.add_argument('--task_sample', type=str2bool, default=False, help='Sample the result')
    parser.add_argument('--task_sample_given_text', type=str2bool, default=False, help='Sample the result')
    parser.add_argument('--task_lpips', type=str2bool, default=False, help='Evaluate the result using LPIPS')
    parser.add_argument('--task_fid', type=str2bool, default=False, help='Evaluate the result using FID')
    parser.add_argument('--task_recall', type=str2bool, default=False, help='Evaluate the result using Recall')
    parser.add_argument('--task_bias_score', type=str2bool, default=False)
    parser.add_argument('--task_clip_score', type=str2bool, default=False)
    parser.add_argument('--task_diverse_outputs_prompt', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=8)

    # bias score
    parser.add_argument('--path_classifier', type=str, default='invariant/ckpts/waterbird/classifier/resnet18-64-0.0001')
    parser.add_argument('--num_class_classifier', type=int, default=2)

    # given text
    parser.add_argument('--path_text_list', type=str, default='invariant/datasets/texts/waterbird_train.txt')
    parser.add_argument('--path_output_given_text', type=str, default='invariant/results/given_text')
    
    args = parser.parse_args()
    return args



def concate_images_for_samples(results_path, save_path):
    # read images and get 4*4 concated image
    import os
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    import torch
    from PIL import Image

    dirs = os.listdir(results_path)
    for d in dirs:
        images = get_real_images(os.path.join(results_path, d), num_images=16, random=False, normalize=False)
        grid = make_grid(images, nrow=4)
        grid = transforms.ToPILImage()(grid)
        grid.save(os.path.join(save_path, f"{d}.png"))
    print(f"Concated images saved to {save_path}")


def main():


    args = arg_parser()
    if 'AMLT_BLOB_DIR' in os.environ:
        # update the path prefix
        args.path_prefix = os.environ['AMLT_BLOB_DIR']
    print(args.path_prefix)
    args.path_ckpt = os.path.join(args.path_prefix, args.path_ckpt)
    args.path_dataset = os.path.join(args.path_prefix, args.path_dataset)
    args.path_result_root = os.path.join(args.path_prefix, args.path_result_root)
    args.path_result = os.path.join(args.path_result_root, f"{args.model_name}-{args.model_checkpoint_num}")
    args.path_classifier = os.path.join(args.path_prefix, args.path_classifier)
    args.path_text_list = os.path.join(args.path_prefix, args.path_text_list)
    args.path_output_given_text = os.path.join(args.path_prefix, args.path_output_given_text)

    print(args)
    all_metric_path = os.path.join(args.path_result_root, 'all_results')
    cur_metric_path = os.path.join(all_metric_path, f"{args.model_name}-{args.model_checkpoint_num}")
    args.cur_metric_path = cur_metric_path
    
    if not os.path.exists(args.path_result_root):
        os.makedirs(args.path_result_root, exist_ok=True)
    if args.task_sample_given_text:
        tip('Task: Sample given text')
        # read text list
        with open(args.path_text_list, 'r') as f:
            text_list = f.readlines()
            text_list = [t.strip() for t in text_list]
        tip('Loading model')
        pipe = load_pipe(args.model_checkpoint_num, os.path.join(args.path_ckpt, args.model_name), delta=args.delta, delta_ratio=args.delta_ratio, delta_init0=args.delta_init0)
        tip('Sampling')
    if args.task_sample:
        tip('Task: Sample')
        tip('Loading dataset')


        dataset = load_dataset(f"{args.path_dataset}")

        print(f"Dataset loaded, total {len(dataset['validation'])} samples, from {args.path_dataset}")
        tip('Loading model')
        pipe = load_pipe(args.model_checkpoint_num, os.path.join(args.path_ckpt, args.model_name), delta=args.delta, delta_ratio=args.delta_ratio, delta_init0=args.delta_init0)
        tip('Sampling')
        if args.task_diverse_outputs_prompt:
            print('Diverse outputs')
            sample_result_batch(pipe, dataset, args.path_result, args.sample_num, args.text_num, save=args.save_sample, batch_size=args.batch_size, diversity_prompt=True)
        else:
            sample_result_batch(pipe, dataset, args.path_result, args.sample_num, args.text_num, save=args.save_sample, batch_size=args.batch_size, diversity_prompt=False)
        if not os.path.exists(cur_metric_path):
            os.makedirs(cur_metric_path, exist_ok=True)
        concate_images_for_samples(args.path_result, args.cur_metric_path)
    if args.task_clip_score:
        tip('Task: clip_score')
        score_clip = get_clip_score(args.path_result)
        with open(os.path.join(cur_metric_path, 'clip.txt'), 'w') as f:
            for k, v in score_clip.items():
                f.write(f"{k}: {v}\n")
        # json
        import json
        with open(os.path.join(cur_metric_path, 'clip.json'), 'w') as f:
            json.dump(score_clip, f)
    if args.task_bias_score:
        tip('Task: task_bias_score')
        score_bias = get_bias_score(args.path_result, args.path_classifier, args.num_class_classifier)
        with open(os.path.join(cur_metric_path, 'bias.txt'), 'w') as f:
            for k, v in score_bias.items():
                f.write(f"{k}: {v}\n")
        # json
        import json
        with open(os.path.join(cur_metric_path, 'bias.json'), 'w') as f:
            json.dump(score_bias, f)

    if args.task_fid:
        tip('Task: FID')
        original_path = args.path_dataset
        score_fid = get_fid_score(original_path, args.path_result, args.sample_num* args.text_num)
        with open(os.path.join(cur_metric_path, 'fid.txt'), 'w') as f:
            for k, v in score_fid.items():
                f.write(f"{k}: {v}\n")
        # json
        import json
        with open(os.path.join(cur_metric_path, 'fid.json'), 'w') as f:
            json.dump(score_fid, f)
    if args.task_recall:
        tip('Task: Recall')
        original_path = args.path_dataset
        score_recall = get_recall_score(original_path, args.path_result, args.sample_num* args.text_num)
        with open(os.path.join(cur_metric_path, 'recall.txt'), 'w') as f:
            for k, v in score_recall.items():
                f.write(f"{k}: {v}\n")
        # json
        import json
        with open(os.path.join(cur_metric_path, 'recall.json'), 'w') as f:
            json.dump(score_recall, f)
    

    if args.task_lpips:

        score_lpips = {'lpips_scores':[-1, -1, -1, -1, -1, -1, -1],
                    'lpips_score':-1,
                    'lpips_score_std': -1,
                    'lpips_score_mean_std': '-1(0)'
                    }
        with open(os.path.join(cur_metric_path, 'lpips.txt'), 'w') as f:
            for k, v in score_lpips.items():
                f.write(f"{k}: {v}\n")
        # json
        import json
        with open(os.path.join(cur_metric_path, 'lpips.json'), 'w') as f:
            json.dump(score_lpips, f)

        tip('Task: LPIPS')
        tip('Calculating LPIPS')
        score_lpips = get_lpips_score(args.path_result, args.sample_num, args.lpips_base)
        with open(os.path.join(cur_metric_path, 'lpips.txt'), 'w') as f:
            for k, v in score_lpips.items():
                f.write(f"{k}: {v}\n")
        # json
        import json
        with open(os.path.join(cur_metric_path, 'lpips.json'), 'w') as f:
            json.dump(score_lpips, f)

def tip(str):
    print(f'-----------------\n {str} \n-----------------')

if __name__ == "__main__":
    main()