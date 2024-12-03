from huggingface_hub import HfApi, hf_hub_download, HfFileSystem
from huggingface_hub import login
repo_id = 'usernamexxxxxxxxx/InvDiff'
# login_token = 'hf_XXXXXXXXXXXXXXX'
# login(login_token)
api = HfApi()


from huggingface_hub import hf_hub_download
# download the file to the local directory
# dir_local = './'
# files_remote = [
#     # 'datasets/waterbird/split_waterbirds_index.zip',
#     # 'datasets/waterbird/split_waterbirds_index_unbias.zip',
#     # 'datasets/fairness/split_fairness_32111123.zip',
#     # 'datasets/fairness/split_fairness_11111111.zip',
#     # 'datasets/celeba/split_celeba_black_ratio_1_1_1_1.zip',
#     # 'datasets/celeba/split_celeba_black_ratio_1_2_2_1.zip',
#     # 'datasets/celeba/split_celeba_black_ratio_1_2_2_2.zip',
#     # 'datasets/celeba/split_celeba_black_ratio_1_4_4_1.zip',
#     # 'datasets/celeba/split_celeba_black_ratio_1_4_4_4.zip',
#     # 'datasets/celeba/split_celeba_black_ratio_1_8_8_1.zip',
#     # 'datasets/celeba/split_celeba_black_ratio_1_8_8_8.zip',
# ]
# for file_remote in files_remote:
#     dir_remote = 'main'
#     hf_hub_download(repo_id=repo_id,
#                  filename=file_remote, revision='main', use_auth_token=True, local_dir=dir_local)
#     # unzip the file
#     import zipfile
#     with zipfile.ZipFile(dir_local + file_remote, 'r') as zip_ref:
#         print(f'unzipping {file_remote}')
#         zip_ref.extractall(dir_local)




# download the floder
from huggingface_hub import Repository
local_folder = "./test"
specific_folder_paths = [
    # 'ckpts/waterbird/diff-waterbird/checkpoint-10000',
    'ckpts/waterbird/eiil-delta-lambda1-delta0.2-small-deltaparam2-True-t_g4_w0-bs64-sc1e-04-ac1-lrcosine-wu1000/checkpoint-10000',
    'ckpts/celeba/split_celeba_black_ratio_1_2_2_1-noeiil-nodelta-lambda0-delta0-small-deltaparam0-False-4-t_g2_w3-bs64-sc1e-04-ac1-lrcosine-wu1000/checkpoint-10000',
    'ckpts/celeba/split_celeba_black_ratio_1_2_2_1-eiil-delta-lambda1-delta0.8-pretrain-deltaparam0-True-4-t_g8_w0-bs64-sc1e-04-ac1-lrcosine-wu1000/checkpoint-10000',
    'ckpts/fairness/split_fairness_32111123-noeiil-nodelta-lambda0-delta0-small-deltaparam0-False-4-t_g2_w3-bs64-sc1e-04-ac1-lrcosine-wu1000/checkpoint-10000',
    'ckpts/fairness/split_fairness_32111123-eiil-delta-lambda1-delta0.8-pretrain-deltaparam0-True-8-t_g8_w0-bs64-sc1e-04-ac1-lrcosine-wu1000/checkpoint-10000',
    # 'ckpts/waterbird//checkpoint-10000',
]

for specific_folder_path in specific_folder_paths:
    fs = HfFileSystem()
    # walk and download all files in the folder
    for root, dirs, files in fs.walk(f'{repo_id}/{specific_folder_path}'):
        print(root, dirs, files)
        for file in files:
            print(root, file)
            file_root = root.split(f'{repo_id}/')[-1]
            file_remote = f'{file_root}/{file}'
            file_local = f'{local_folder}/{file_remote}'
            print(file_remote, file_local)
            hf_hub_download(repo_id=repo_id, filename=file_remote, local_dir=local_folder)

