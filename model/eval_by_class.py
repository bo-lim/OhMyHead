from tqdm import tqdm
from PIL import Image
import json
import os
import sys
from scalp_model import *
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def test_model(dataset_path) :
    
    model = make_model()

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

    total_mse_val_1 = 0.0
    total_mse_val_2 = 0.0
    total_mse_val_3 = 0.0
    total_mse_val_4 = 0.0
    total_mse_val_5 = 0.0
    total_mse_val_6 = 0.0

    data_count = 0
    
    for folder_name in tqdm(label_folder_list, desc = "get MSE") : 
        
        class_name = folder_name[7:] # or folder_name[4:]. 실행하기 전에 테스트 해볼 것. '모낭사이홍반_0.양호'과 같은 문자열이 나오는게 정상

        image_folder_path = image_group_folder_path + "/" + '[원천]' + class_name 
        label_folder_path = label_group_folder_path + '/' + folder_name

        json_list = os.listdir(label_folder_path) # json파일 목록 담기

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

            total_mse_val_1 += loss(y_pred[0], y_true[0]).item()
            total_mse_val_2 += loss(y_pred[1], y_true[1]).item()
            total_mse_val_3 += loss(y_pred[2], y_true[2]).item()
            total_mse_val_4 += loss(y_pred[3], y_true[3]).item()
            total_mse_val_5 += loss(y_pred[4], y_true[4]).item()
            total_mse_val_6 += loss(y_pred[5], y_true[5]).item()

            data_count+=1

    total_mse_val_1 /= data_count
    total_mse_val_2 /= data_count
    total_mse_val_3 /= data_count
    total_mse_val_4 /= data_count
    total_mse_val_5 /= data_count
    total_mse_val_6 /= data_count

    print("미세각질의 MSE : ", total_mse_val_1)
    print("피지과다의 MSE : ", total_mse_val_2)
    print("모낭사이홍반의 MSE : ", total_mse_val_3)
    print("모낭홍반농포의 MSE : ", total_mse_val_4)
    print("비듬의 MSE : ", total_mse_val_5)
    print("탈모의 MSE : ", total_mse_val_6)

