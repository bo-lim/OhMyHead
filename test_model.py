import model.scalp_model as scalp_model
from model.eval_by_class import test_model
import torch
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--do_eval', type=bool)
    args = parser.parse_args()

    IMAGE_PATH = args.img_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PATH = os.getcwd() + "/Scalp_model_parameters/"

    model = scalp_model.load_model_trained(device, PATH)
    scalp_model.test_model(IMAGE_PATH, model, device)

    if args.do_eval:
        test_model(IMAGE_PATH)
