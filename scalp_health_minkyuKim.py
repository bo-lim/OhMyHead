# 모델을 테스트해보는 코드 

# 제작일 : 2022.01.09
# 제작자 : 김민규(minkyu4506@gmail.com)

import model.scalp_model as scalp_model
import torch
import os
import sys

def main(IMAGE_PATH) :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PATH = os.getcwd() + "/Scalp_model_parameters"

    model = scalp_model.load_model_trained(device, PATH)

    scalp_model.test_model(IMAGE_PATH, model, device)

if __name__ == "__main__":
    IMAGE_PATH = sys.argv # 터미널에서 실행할 때 같이 입력한 이미지 경로
    main(IMAGE_PATH)
