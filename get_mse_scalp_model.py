# 모델의 MSE를 구할 때 사용하는 메소드들을 모아놓은 파이썬 파일

# 제작일 : 2022.01.20
# 제작자 : 김민규(minkyu4506@gmail.com)

from tqdm import tqdm
from PIL import Image
import json
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def make_model() :
    
    model = models.densenet161(pretrained = True, memory_efficient = True)

    # ImageNet으로 학습시킨 CNN은 수정하지 못하게끔
    for param in model.features.parameters():
        param.requires_grad = False

    # 모발 분류를 위한 linear model 생성 후 교체
    new_classifier = nn.Sequential(
            nn.Linear(in_features=2208, out_features=512, bias=True),
            nn.BatchNorm1d(num_features = 512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.BatchNorm1d(num_features = 256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.BatchNorm1d(num_features = 256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.BatchNorm1d(num_features = 128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.BatchNorm1d(num_features = 64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.BatchNorm1d(num_features = 32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=6, bias=False)
        )

    for m in new_classifier.modules():
        if isinstance(m, nn.Linear) :
            nn.init.kaiming_uniform_(m.weight)
            
    model.classifier = new_classifier # 교체
    
    return model

def test_model(dataset_path, device) :
    
    model = make_model(device)

    # 학습된 가중치들 불러오기
    PATH_model = dataset_path + '/model_parameters.pt'
        
    model.load_state_dict(torch.load(PATH_model, map_location=torch.device('cpu')))

    image_group_folder_path = dataset_path + '/Image'
    label_group_folder_path = dataset_path + '/Label'
    
    ori_label_folder_list = os.listdir(label_group_folder_path) # '[라벨]피지과다_3.중증' 등 폴더명 알기
    
    label_folder_list = []

    for i in range(len(ori_label_folder_list)) :
        if ori_label_folder_list[i] != '.DS_Store' : # '.DS_Store'가 생성되었을 수 있으니 폴더 목록에서 제외
            label_folder_list.append(ori_label_folder_list[i])
    
    loss = nn.MSELoss()

    model.eval()
    
    for folder_name in tqdm(label_folder_list, desc = "get MSE") : 
        
        class_name = folder_name[7:] # or folder_name[4:]. 실행하기 전에 테스트 해볼 것. '모낭사이홍반_0.양호'과 같은 문자열이 나오는게 정상

        image_folder_path = image_group_folder_path + "/" + '[원천]' + class_name 
        label_folder_path = label_group_folder_path + '/' + folder_name

        json_list = os.listdir(label_folder_path) # json파일 목록 담기

        total_mse_per_class = 0.0
        for j in range(len(json_list)) : 
            json_file_path = label_folder_path + '/' + json_list[j]

            # 파일 오픈
            with open(json_file_path, "r", encoding="utf8") as f: 
                contents = f.read() # string 타입 
                json_content = json.loads(contents) # 딕셔너리로 저장

            image_file_name = json_content['image_file_name'] # 라벨 데이터에 이미지 파일의 이름이 들어있다
            image_file_path = image_folder_path + "/" + image_file_name
            to_tensor = transforms.ToTensor()
            img = to_tensor(Image.open(image_file_path).convert('RGB'))
            img.resize_(3, 224, 224)
            img = torch.divide(img, 255.0) # 텐서로 변경 후 이미지 리사이징하고 각 채널을 0~1 사이의 값으로 만들어버림
            img = torch.unsqueeze(img, 0)

            # val1 : 미세각질, val2 : 피지과다, val3 : 모낭사이홍반, val4 : 모낭홍반/농포, val5 : 비듬, val6 : 탈모
            # 모든 val은 0,1,2,3 중 하나의 값을 가지고 있다
            vals_true = []
            vals_true.append(int(json_content['value_1']))
            vals_true.append(int(json_content['value_2']))
            vals_true.append(int(json_content['value_3']))
            vals_true.append(int(json_content['value_4']))
            vals_true.append(int(json_content['value_5']))
            vals_true.append(int(json_content['value_6']))

            y_true = torch.Tensor(vals_true).type(torch.float32) # true

            y_pred = torch.squeeze(model(img))

            total_mse_per_class +=loss(y_pred, y_true).item()
        
        avr_mse_per_class = total_mse_per_class / len(json_list)

        print("\"" + class_name + "\"의 MSE : ", avr_mse_per_class) 


def main(DATASET_PATH) :

    test_model(DATASET_PATH)

if __name__ == "__main__":
    DATASET_PATH = sys.argv # 터미널에서 실행할 때 같이 입력한 이미지 경로
    main(DATASET_PATH)
