# 데이터셋 샘플링 모듈. 서버에 올릴만한 크기로 줄인다. 줄이는 과정에서 클래스별 데이터의 개수를 균등하게 만든다. 

# 제작일 : 2022.01.09
# 제작자 : 김민규(minkyu4506@gmail.com)

import os
import shutil
from tqdm import tqdm
import natsort

# root_path = 데이터셋이 있는 장소, new_root_path = 샘플링할 데이터셋을 저장할 장소, num_sampling_per_class = 클래스별로 샘플링할 데이터의 개수
def dataset_selection(root_path, new_root_path, num_sampling_per_class) : 
    folder_list = os.listdir(root_path)

    edited_folder_list = []

    # '.DS_Store' 제거
    for i in range(len(folder_list)) :
        if folder_list[i] != '.DS_Store' :
            edited_folder_list.append(folder_list[i])

    # 폴더 생성
    for folder in tqdm(edited_folder_list, desc = "data sampling", mininterval = 0.01) :
        new_folder_path = new_root_path + '/' + folder
        os.makedirs(new_folder_path)
    
    # 샘플한 파일 복사
    for folder in tqdm(edited_folder_list, desc = "data sampling", mininterval = 0.01) :
        folder_path = root_path + '/' + folder
        file_list = os.listdir(folder_path)
        file_list = natsort.natsorted(file_list)


        for i in range(0, num_sampling_per_class) :
            file_path = folder_path + '/' + file_list[i]
            copied_path = new_root_path + '/' + folder + '/' + file_list[i]
            shutil.copy(file_path, copied_path)