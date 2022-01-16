# 학습과 테스트에 사용할 데이터셋을 만드는데 쓰이는 메소드들이 저장된 모듈

# 제작일 : 2022.01.09
# 제작자 : 김민규(minkyu4506@gmail.com)

from tqdm import tqdm
from PIL import Image
import json
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Scalp_Health_Dataset(Dataset) :
    def __init__(self, image_path_list, vals_list) : # 용량을 고려해 이미지는 경로만 받는걸로
        self.image_path_list = image_path_list
        self.vals_list = vals_list
    def __len__(self) : 
        return len(self.image_path_list)

    def __getitem__(self, index) : # 한 개의 데이터 가져오는 함수
        # 224 X 224로 전처리
        to_tensor = transforms.ToTensor()
        img = to_tensor(Image.open(self.image_path_list[index]).convert('RGB'))
        img.resize_(3, 224, 224)
        img = torch.divide(img, 255.0) # 텐서로 변경 후 이미지 리사이징하고 각 채널을 0~1 사이의 값으로 만들어버림
        
        label = self.vals_list[index]
        
        return img, label


# dataset_path : root_path + '/Train'이나 root_path + '/Test'을 받음
# (root_path : 데이터셋이 저장된 경로) 
def make_dataset(dataset_path, category) :
    
    
    image_group_folder_path = dataset_path + '/Image'
    label_group_folder_path = dataset_path + '/Label'
    
    ori_label_folder_list = os.listdir(label_group_folder_path) # '[라벨]피지과다_3.중증' 등 폴더명 알기
    
    label_folder_list = []
    
    for i in range(len(ori_label_folder_list)) :
        if ori_label_folder_list[i] != '.DS_Store' : # '.DS_Store'가 생성되었을 수 있으니 폴더 목록에서 제외
            label_folder_list.append(ori_label_folder_list[i])
    
    image_path_list = []
    vals_list = []
    class_str_list = []
    
    desc_str = category + "_make_dataset"
    
    for i in tqdm(range(len(label_folder_list)), desc = desc_str) :
                  
        label_folder_path = label_group_folder_path + "/" + label_folder_list[i]
        
        # label_folder_list에서 '라벨'을 '원천'으로 만 바꿔도 image파일들이 들어있는 폴더명으로 만들 수 있다
        image_folder_path = image_group_folder_path + "/" + label_folder_list[i].replace('라벨', '원천')
        
        json_list = os.listdir(label_folder_path) # json파일 목록 담기

        for j in range(len(json_list)) : 
            json_file_path = label_folder_path + '/' + json_list[j]

            with open(json_file_path, "r", encoding="utf8") as f: 
                contents = f.read() # string 타입 
                json_content = json.loads(contents) # 딕셔너리로 저장

            image_file_name = json_content['image_file_name'] # 라벨 데이터에 이미지 파일의 이름이 들어있다
            
            image_file_path = image_folder_path + "/" + image_file_name

            # val1 : 미세각질, val2 : 피지과다, val3 : 모낭사이홍반, val4 : 모낭홍반/농포, val5 : 비듬, val6 : 탈모
            # 모든 val은 0,1,2,3 중 하나의 값을 가지고 있다
            vals_true = []
            vals_true.append(int(json_content['value_1']))
            vals_true.append(int(json_content['value_2']))
            vals_true.append(int(json_content['value_3']))
            vals_true.append(int(json_content['value_4']))
            vals_true.append(int(json_content['value_5']))
            vals_true.append(int(json_content['value_6']))

            vals_true = torch.Tensor(vals_true).type(torch.float32)

            image_path_list.append(image_file_path)
            vals_list.append(vals_true/3.0)
            
    return image_path_list, vals_list
    # image_path_list : 파일 경로가 저장된 리스트
    # vals_list : val1 ~ val6이 들어있는 Tensor 리스트
    
# 하나의 모발 이미지가 여러 증상을 가진 경우가 있다.
# 하나의 이미지가 [A증상 중증, B증상 경증] 등 여러 증상에 대한 중증도를 나타내게끔 라벨 데이터를 만들어주는 기능도 한다
# 즉, 데이터 전처리
def make_unique_dataset(image_path_list, vals_list) :  
    unique_image_path_list = []
    unique_vals_list = []
    
    for i in tqdm(range(len(image_path_list)), desc = "make unique dataset" ) : 
        file_name = image_path_list[i].split('/')[-1] # 이미지 파일 이름
        
        # 만들고 있던 unique list의 안에 같은 파일이름을 가진게 없는지 확인
        is_sameFilename_here = False
        for j in range(len(unique_image_path_list)) :
            if unique_image_path_list[j].split('/')[-1] == file_name : # 중복된 파일이 있으면
                is_sameFilename_here = True # 중복 처리
                
        # 동일한 파일이 없으면 unique리스트에 추가
        if is_sameFilename_here == False :
            unique_image_path_list.append(image_path_list[i])
            unique_vals_list.append(vals_list[i])
    
    return unique_image_path_list, unique_vals_list


def get_dataset(root_path) : 
    
    Train_image_path_list, Train_vals_list = make_dataset(root_path + '/Train', "Train")
    Train_image_path_list, Train_vals_list = make_unique_dataset(Train_image_path_list, Train_vals_list)
    
    
    Test_image_path_list, Test_vals_list = make_dataset(root_path + '/Test', "Test")
    Test_image_path_list, Test_vals_list = make_unique_dataset(Test_image_path_list, Test_vals_list)
    
    len_train_ds = int(len(Train_image_path_list) * 0.8) # 학습에 사용할 데이터 개수
    
    Valid_image_path_list = Train_image_path_list[len_train_ds:]
    Train_image_path_list = Train_image_path_list[:len_train_ds]
    
    Valid_vals_list = Train_vals_list[len_train_ds:]
    Train_vals_list = Train_vals_list[:len_train_ds]
    
    Train_Scalp_Health_Dataset = Scalp_Health_Dataset(Train_image_path_list, Train_vals_list)
    Valid_Scalp_Health_Dataset = Scalp_Health_Dataset(Valid_image_path_list, Valid_vals_list)
    Test_Scalp_Health_Dataset = Scalp_Health_Dataset(Test_image_path_list, Test_vals_list)

    return Train_Scalp_Health_Dataset, Valid_Scalp_Health_Dataset, Test_Scalp_Health_Dataset
    
    