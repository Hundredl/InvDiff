import os
import zipfile
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub import HfApi
from huggingface_hub import login
# login_token = 'hf_XXXXXXXXXXXXXXXX'
# login(login_token)
# api = HfApi(token=login_token)
repo_id = 'Invdiff/Invdiff-Data'
# api = HfApi(endpoint='https://hf-mirror.com', token=login_token)
api = HfApi()

# upload classifier and groupers and models
classifier_template = 'invariant/ckpts/{dataset}/classifier.zip'
groupers_template = 'invariant/ckpts/{dataset}/groupers.zip'
models_template = 'invariant/ckpts/{dataset}/models.zip'

datasets = ['celeba', 'fairness', 'waterbird']
for dataset in datasets:
    # upload classifier
    classifier_path = classifier_template.format(dataset=dataset)
    api.upload_file(repo_id=repo_id, path_or_fileobj=classifier_path, path_in_repo=classifier_path,
                    use_auth_token=True)
    # upload groupers
    groupers_path = groupers_template.format(dataset=dataset)
    api.upload_file(repo_id=repo_id, path_or_fileobj=groupers_path, path_in_repo=groupers_path,
                    use_auth_token=True)
    # upload models
    models_path = models_template.format(dataset=dataset)
    api.upload_file(repo_id=repo_id, path_or_fileobj=models_path, path_in_repo=models_path,
                    use_auth_token=True)

    
    
    