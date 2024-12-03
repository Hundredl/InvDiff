import os
import zipfile
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub import HfApi
from huggingface_hub import login
# login_token = 'hf_XXXXXXXXXXXXXXXXXXXXX'
# login(login_token)
repo_id = 'Invdiff/Invdiff-Data'
# api = HfApi(endpoint='https://hf-mirror.com', token=login_token)
# api = HfApi(token=login_token)
api = HfApi()

classifier_template = 'invariant/ckpts/{dataset}/classifier.zip'
groupers_template = 'invariant/ckpts/{dataset}/groupers.zip'
models_template = 'invariant/ckpts/{dataset}/models.zip'

datasets = ['celeba', 'fairness', 'waterbird']
for dataset in datasets:
    # download classifier
    classifier_path = classifier_template.format(dataset=dataset)
    local_dir = './'
    hf_hub_download(repo_id=repo_id, filename=classifier_path, local_dir=local_dir,
                    revision='main',)
    
    # unzip the file
    unzip_dir = '/'.join(classifier_path.split('/')[:-1])
    with zipfile.ZipFile(classifier_path, 'r') as zip_ref:
        print(f'unzipping {classifier_path} to {unzip_dir}')
        zip_ref.extractall(unzip_dir)
        
    # download groupers
    groupers_path = groupers_template.format(dataset=dataset)
    local_dir = './'
    hf_hub_download(repo_id=repo_id, filename=groupers_path, local_dir=local_dir,
                    revision='main')
    # unzip the file
    unzip_dir = '/'.join(groupers_path.split('/')[:-1])
    with zipfile.ZipFile(groupers_path, 'r') as zip_ref:
        print(f'unzipping {groupers_path} to {unzip_dir}')
        zip_ref.extractall(unzip_dir)
    
    # download models
    models_path = models_template.format(dataset=dataset)
    local_dir = './'
    hf_hub_download(repo_id=repo_id, filename=models_path, local_dir=local_dir,
                    revision='main')
    # unzip the file
    unzip_dir = '/'.join(models_path.split('/')[:-1])
    with zipfile.ZipFile(models_path, 'r') as zip_ref:
        print(f'unzipping {models_path} to {unzip_dir}')
        zip_ref.extractall(unzip_dir)