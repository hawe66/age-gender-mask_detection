import argparse
import multiprocessing
import os
from importlib import import_module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import Counter
from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir_m, model_dir_g, model_dir_a, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18

    model_m = load_model(model_dir_m, 3, device).to(device)
    model_g = load_model(model_dir_g, 2, device).to(device)
    model_a = load_model(model_dir_a, 3, device).to(device)
    model_m.eval()
    model_g.eval()
    model_a.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds_ = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred_m = model_m(images)
            pred_g = model_g(images)
            pred_a = model_a(images)
            pred_m = pred_m.argmax(dim=-1)
            pred_g = pred_g.argmax(dim=-1)
            pred_a = pred_a.argmax(dim=-1)
            pred = MaskBaseDataset.encode_multi_class(pred_m, pred_g, pred_a)
            preds_.extend(pred.cpu().numpy())
    
    preds = []
    preds.append(preds_)
    # Test-Time-Augmentation !!!!
    for i in range(5):
        dataset = TestDataset(img_paths, args.resize, augment=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print(f"Calculating augmented ver {i+1} inference results..")
        preds_ = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred_m = model_m(images)
                pred_g = model_g(images)
                pred_a = model_a(images)
                pred_m = pred_m.argmax(dim=-1)
                pred_g = pred_g.argmax(dim=-1)
                pred_a = pred_a.argmax(dim=-1)
                pred = MaskBaseDataset.encode_multi_class(pred_m, pred_g, pred_a)
                preds_.extend(pred.cpu().numpy())

    # Voting   
    pred_ = np.array(preds).T
    pred = []
    for p_ in pred_:
        counts = Counter(p_)
        pred.append(counts.most_common(1)[0][0])

    info['ans'] = pred
    save_path = os.path.join(output_dir, f'output_split.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=(320, 256), help='resize size for image when you trained (default: (320, 256))')
    parser.add_argument('--model', type=str, default='MyModel', help='model type (default: MyModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir_m', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp_m'))
    parser.add_argument('--model_dir_g', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp_g'))
    parser.add_argument('--model_dir_a', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp_a'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir_m = args.model_dir_m
    model_dir_g = args.model_dir_g
    model_dir_a = args.model_dir_a
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir_m, model_dir_g, model_dir_a, output_dir, args)
