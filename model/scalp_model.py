# 모델의 생성, 학습, 가중치 불러오기, 테스트를 하는 메소드를 모아놓은 모듈

# 제작일 : 2022.01.09
# 제작자 : 김민규(minkyu4506@gmail.com)

import scalp_dataset

from tqdm import tqdm
from PIL import Image
import copy

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# 모델 학습에 사용됨
def train(model, criterion, optimizer, epochs, train_data_loader, valid_data_loader, device, desc_str) :
    # 학습 -> 성능 측정
    
    checkpoint_model = 0
    minimun_loss = 0
    
    train_loss = torch.Tensor([-1]).to(device)
    avr_valid_loss = torch.Tensor([-1]).to(device)
    
    pbar_str = desc_str + ", [train_loss = " + str(round(train_loss.item(),4)) + ", avr_val_loss = " + str(round(avr_valid_loss.item(),4)) + "]"
    pbar = tqdm(range(epochs), desc = desc_str, mininterval=0.01)
    
    for epoch in pbar:  # loop over the dataset multiple times
            
        for inputs, labels in train_data_loader:
            
            # get the inputs
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            
            train_loss.backward()
            optimizer.step()
            
            # print statistics
            pbar_str = desc_str + ", [*train_loss* = " + str(round(train_loss.item(),4)) + ", avr_val_loss = " + str(round(avr_valid_loss.item(),4)) + "]"
            pbar.set_description(pbar_str)
            
            # 체크포인트
            # train loss가 줄어든 모델을 저장
            if checkpoint_model == 0 :
                minimun_loss = train_loss
                checkpoint_model = copy.deepcopy(model)

            elif torch.lt(train_loss, minimun_loss) == True : 
                minimun_loss = train_loss
                checkpoint_model = copy.deepcopy(model)
            
        # 에포크당 검증손실의 평균 계산
        valid_count = 0
        total_valid_loss = torch.Tensor([0]).to(device)
        for inputs, labels in valid_data_loader : 
            
            # get the inputs
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward + backward + optimize
            outputs = model(inputs)
            valid_loss = criterion(outputs, labels) # 검증 손실 
            
            total_valid_loss = torch.add(total_valid_loss, valid_loss)
            valid_count += 1.0
            
        avr_valid_loss = torch.div(total_valid_loss, torch.Tensor([valid_count]).to(device))
        
        pbar_str = desc_str + ", [train_loss = " + str(round(train_loss.item(),4)) + ", *avr_val_loss* = " + str(round(avr_valid_loss.item(),4)) + "]"
        pbar.set_description(pbar_str)

    # 체크포인트에 저장했던걸 최종 모델로
    model = copy.deepcopy(checkpoint_model)
    
    return model

# 모델을 생성
def make_model(device) :

    model = models.densenet161(pretrained = True, memory_efficient = True).to(device)

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
            ).to(device)

    for m in new_classifier.modules():
        if isinstance(m, nn.Linear) :
            nn.init.kaiming_uniform_(m.weight)
            
    model.classifier = new_classifier # 교체
    
    return model

# 학습된 가중치를 불러옴
# device : 'cuda' or 'cpu'
# PATH의 예 : '/home/ubuntu/CUAI_2021/Advanced_Minkyu_Kim/Scalp_model_parameters/'
def load_model_trained(device, PATH) :

    model = make_model(device)

    # 학습된 가중치들 불러오기
    PATH_model = PATH + 'model_parameter.pt'
    
    model.load_state_dict(torch.load(PATH_model))
    
    return model

# 모델을 학습시킴
def train_model(dataset_root_path, model, device) :

    Train_Scalp_Health_Dataset, Valid_Scalp_Health_Dataset,_ = scalp_dataset.get_dataset(dataset_root_path)

    # 학습에 사용
    BATCH_SIZE = 256

    Train_data_loader_Scalp_Health_Dataset = torch.utils.data.DataLoader(dataset=Train_Scalp_Health_Dataset, # 사용할 데이터셋
                                            batch_size=BATCH_SIZE, # 미니배치 크기
                                            shuffle=True, # 에포크마다 데이터셋 셔플할건가? 
                                            drop_last=True) # 마지막 배치가 BATCH_SIZE보다 작을 수 있다. 나머지가 항상 0일 수는 없지 않는가. 이 때 마지막 배치는 사용하지 않으려면 drop_last = True를, 사용할거면 drop_last = False를 입력한다

    Valid_data_loader_Scalp_Health_Dataset = torch.utils.data.DataLoader(dataset=Valid_Scalp_Health_Dataset, # 사용할 데이터셋
                                            batch_size=BATCH_SIZE, # 미니배치 크기
                                            shuffle=True, # 에포크마다 데이터셋 셔플할건가? 
                                            drop_last=True) # 마지막 배치가 BATCH_SIZE보다 작을 수 있다. 나머지가 항상 0일 수는 없지 않는가. 이 때 마지막 배치는 사용하지 않으려면 drop_last = True를, 사용할거면 drop_last = False를 입력한다

    # 모델 학습

    EPOCHS = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-4)
    loss = nn.MSELoss() # 6종류의 출력값을 예측하는 선형회귀 모델이라 MSE사용

    model = train(model, loss, optimizer, EPOCHS, Train_data_loader_Scalp_Health_Dataset, Valid_data_loader_Scalp_Health_Dataset, device, "training")
    
    return model

# 모델을 테스트
# img_path : 이미지 파일의 경로
def test_model(img_path, model, device) :

    state_str_list = ["미세각질", "피지과다", "모낭사이홍반", "모낭홍반농포", "비듬", "탈모", "양호"]
    
    # 입력받은 이미지 경로에서 가져온 이미지를 전처리
    to_tensor = transforms.ToTensor()
    img = to_tensor(Image.open(img_path).convert('RGB'))
    img.resize_(3, 224, 224)
    img = torch.divide(img, 255.0) # 텐서로 변경 후 이미지 리사이징하고 각 채널을 0~1 사이의 값으로 만들어버림
    img = torch.unsqueeze(img, 0).to(device)
    
    # 테스트 모드로 전환
    model.eval()
    
    output = model(img) # val1 ~ val6을 반환
    
    # 결과 출력
    print_str = "검사 결과, "
    for i in range(len(state_str_list)) : 
        print_str += state_str_list[i] + "이(가) %.4f 만큼, " % output[i].item()

    print_str = print_str[:-2]
    
    print_str += " 있습니다."

    print(print_str)
    
    return output # 사용할 수도 있으니 반환