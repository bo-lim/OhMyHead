# 모델의 생성, 학습, 가중치 불러오기, 테스트를 하는 메소드를 모아놓은 모듈

# 제작일 : 2022.01.09
# 제작자 : 김민규(minkyu4506@gmail.com)

import dataset 

from tqdm import tqdm
from PIL import Image
import copy

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# 모델 학습에 사용됨
def train(model, criterion, optimizer, epochs, data_loader, device, desc_str) :
    # 학습 -> 성능 측정
    
    pred_loss = -1.0
    checkpoint_model = 0
    
    pbar = tqdm(range(epochs), desc = desc_str, mininterval=0.01)
    
    for epoch in pbar:  # loop over the dataset multiple times
            
        for inputs, labels in data_loader:
            
            # get the inputs
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            pbar_str = "training CNN" + ", [loss = %.4f]" % loss.item()
            pbar.set_description(pbar_str)
            
            # 체크포인트
            # 배치 단위로 학습시킬 때마다 체크포인트 저장 여부 확인
            if pred_loss == -1.0 :
                pred_loss = loss
                checkpoint_model = copy.deepcopy(model)
            elif torch.lt(loss, pred_loss) == True : # loss를 더 줄였으면
                pred_loss = loss
                checkpoint_model = copy.deepcopy(model)
    
    # 체크포인트에 저장했던걸 최종 모델로
    model = copy.deepcopy(checkpoint_model)
    
    return model

# 모델을 생성
def make_model(device) :

    model_CNN = models.densenet161(pretrained = True, memory_efficient = True).to(device)

    # ImageNet으로 학습시킨 CNN은 수정하지 못하게끔
    for param in model_CNN.features.parameters():
        param.requires_grad = False
    
    # 모발 분류를 위한 linear model 생성 후 교체
    new_classifier = nn.Sequential(
                nn.Linear(in_features=2208, out_features=1024, bias=True),
                nn.LeakyReLU(),
                nn.Linear(in_features=1024, out_features=512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(in_features=512, out_features=256, bias=True),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=128, bias=True),
                nn.LeakyReLU(),
                nn.Linear(in_features=128, out_features=64, bias=True),
                nn.LeakyReLU(),
                nn.Linear(in_features=64, out_features=32, bias=True),
                nn.LeakyReLU(),
                nn.Linear(in_features=32, out_features=6, bias=False),
                nn.Sigmoid() # 0~1사이 값으로 만들어버림
            ).to(device)

    for m in new_classifier.modules():
        if isinstance(m, nn.Linear) :
            nn.init.kaiming_uniform_(m.weight)
            
    model_CNN.classifier = new_classifier # 교체

    model_Diagnoser = nn.Sequential(
            nn.Linear(in_features=6, out_features=64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64, bias=True), 
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=32, bias=True), 
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=7, bias=False), 
            nn.Sigmoid()
        ).to(device)
    
    return model_CNN, model_Diagnoser

# 학습된 가중치를 불러옴
# device : 'cuda' or 'cpu'
# PATH의 예 : '/home/ubuntu/CUAI_2021/Advanced_Minkyu_Kim/Scalp_model_parameters/'
def load_model_trained(device, PATH) : 
    model_CNN, model_Diagnoser = make_model(device)

    # 학습된 가중치들 불러오기
    PATH_model_CNN = PATH + 'model_CNN_parameter.pt'
    PATH_model_Diagnoser = PATH + 'model_Diagnoser_parameter.pt'
    
    model_CNN.load_state_dict(torch.load(PATH_model_CNN))
    model_Diagnoser.load_state_dict(torch.load(PATH_model_Diagnoser))

    return model_CNN, model_Diagnoser

# 모델을 학습시킴
def train_model(dataset_root_path, model_CNN, model_Diagnoser, device) :

    Scalp_Health_Dataset, Scalp_classifier_Dataset = dataset.get_dataset(dataset_root_path)

    # 학습에 사용
    BATCH_SIZE = 256

    data_loader_Scalp_Health_Dataset = torch.utils.data.DataLoader(dataset=Scalp_Health_Dataset, # 사용할 데이터셋
                                            batch_size=BATCH_SIZE, # 미니배치 크기
                                            shuffle=True, # 에포크마다 데이터셋 셔플할건가? 
                                            drop_last=True) # 마지막 배치가 BATCH_SIZE보다 작을 수 있다. 나머지가 항상 0일 수는 없지 않는가. 이 때 마지막 배치는 사용하지 않으려면 drop_last = True를, 사용할거면 drop_last = False를 입력한다

    data_loader_Scalp_classifier_Dataset = torch.utils.data.DataLoader(dataset=Scalp_classifier_Dataset, # 사용할 데이터셋
                                            batch_size=BATCH_SIZE, # 미니배치 크기
                                            shuffle=True, # 에포크마다 데이터셋 셔플할건가? 
                                            drop_last=True) 

    # 모델 학습

    EPOCHS = 100
    # CNN 학습에 필요한 것들
    optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=0.001, weight_decay = 1e-4)
    loss_CNN = nn.MSELoss() # 6종류의 출력값을 예측하는 선형회귀 모델이라 MSE사용

    # Diagnoser 학습에 필요한 것들
    loss_diagnoser = nn.MSELoss()
    optimizer_diagnoser = torch.optim.Adam(model_Diagnoser.parameters(), lr=0.001, weight_decay = 1e-4)

    model_CNN = train(model_CNN, loss_CNN, optimizer_CNN, EPOCHS, data_loader_Scalp_Health_Dataset, device, "training CNN")
    model_Diagnoser = train(model_Diagnoser, loss_diagnoser, optimizer_diagnoser, EPOCHS, data_loader_Scalp_classifier_Dataset, device, "training Diagnoser")

    return model_CNN, model_Diagnoser

# 모델을 테스트
# img_path : 이미지 파일의 경로
def test_model(img_path, model_CNN, model_Diagnoser, device) :

    state_str_list = ["미세각질", "피지과다", "모낭사이홍반", "모낭홍반농포", "비듬", "탈모", "양호"]
    
    # 입력받은 이미지 경로에서 가져온 이미지를 전처리
    to_tensor = transforms.ToTensor()
    img = to_tensor(Image.open(img_path).convert('RGB'))
    img.resize_(3, 224, 224)
    img = torch.divide(img, 255.0) # 텐서로 변경 후 이미지 리사이징하고 각 채널을 0~1 사이의 값으로 만들어버림
    img = torch.unsqueeze(img, 0).to(device)
    
    # 전처리
    model_CNN.eval()
    model_Diagnoser.eval()
    
    output = model_CNN(img)
    output_state_per_class = model_Diagnoser(output)
    
    # 결과 출력
    print_str = "검사 결과, "
    for i in range(len(state_str_list)) : 
        print_str += state_str_list[i] + "이(가) %.4f 만큼, " % output_state_per_class[i].item()

    print_str = print_str[:-2]
    
    print_str += " 있습니다."

    print(print_str)
    
    return output_state_per_class