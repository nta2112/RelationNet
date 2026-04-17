#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21 | Updated for modern PyTorch + Colab compatibility
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",    type=int,   default=64)
parser.add_argument("-r","--relation_dim",   type=int,   default=8)
parser.add_argument("-w","--class_num",      type=int,   default=5)
parser.add_argument("-s","--sample_num_per_class", type=int, default=5)
parser.add_argument("-b","--batch_num_per_class",  type=int, default=10)
parser.add_argument("-e","--episode",        type=int,   default=500000)
parser.add_argument("-t","--test_episode",   type=int,   default=600)
parser.add_argument("-l","--learning_rate",  type=float, default=0.001)
parser.add_argument("-g","--gpu",            type=int,   default=0)
parser.add_argument("-u","--hidden_unit",    type=int,   default=10)
# ── Configurable paths ──────────────────────────────────────────────────────
parser.add_argument("--model_dir", type=str, default="./models",
                    help="Directory to save/load model checkpoints")
parser.add_argument("--log_dir",   type=str, default="./logs",
                    help="Directory to save training logs (CSV)")
# ── JSON split options ───────────────────────────────────────────────────────
parser.add_argument("--split_file",   type=str, default=None,
                    help="(Format 1) Path to split JSON: {\"train\": [...], \"val\": [...]}")
parser.add_argument("--train_json",   type=str, default=None,
                    help="(Format 2) Path to train JSON with image_names/image_labels")
parser.add_argument("--test_json",    type=str, default=None,
                    help="(Format 2) Path to val/test JSON with image_names/image_labels")
parser.add_argument("--data_root",    type=str, default=None,
                    help="Root directory of images (required when using JSON split)")
parser.add_argument("--train_key",    type=str, default="train",
                    help="Key for meta-train split inside split_file (default: 'train')")
parser.add_argument("--test_key",     type=str, default="val",
                    help="Key for meta-test  split inside split_file (default: 'val')")
# ── Backbone options ─────────────────────────────────────────────────────────
parser.add_argument("--backbone",     type=str, default="conv4",
                    choices=["conv4", "resnet50"],
                    help="Feature extractor backbone: 'conv4' (default) or 'resnet50'")
parser.add_argument("--image_size",   type=int, default=None,
                    help="Input image size (default: 84 for conv4, 224 for resnet50)")
parser.add_argument("--pretrained",   action="store_true", default=True,
                    help="Use ImageNet pretrained weights for resnet50 (default: True)")
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM          = args.feature_dim
RELATION_DIM         = args.relation_dim
CLASS_NUM            = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS  = args.batch_num_per_class
EPISODE              = args.episode
TEST_EPISODE         = args.test_episode
LEARNING_RATE        = args.learning_rate
GPU                  = args.gpu
HIDDEN_UNIT          = args.hidden_unit
MODEL_DIR            = args.model_dir
LOG_DIR              = args.log_dir
SPLIT_FILE           = args.split_file
TRAIN_JSON           = args.train_json
TEST_JSON            = args.test_json
DATA_ROOT            = args.data_root
TRAIN_KEY            = args.train_key
TEST_KEY             = args.test_key
BACKBONE             = args.backbone
USE_PRETRAINED       = args.pretrained
IMAGE_SIZE           = args.image_size if args.image_size is not None else (
                           224 if args.backbone == "resnet50" else 84)


def mean_confidence_interval(data, confidence=0.95):
    a  = 1.0 * np.array(data)
    n  = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h  = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


class CNNEncoder(nn.Module):
    """
    Backbone gốc: CNN 4 lớp (Conv-64).
    Input:  (B, 3, 84, 84)
    Output: (B, 64, H, W)
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    @property
    def out_channels(self):
        return 64


class ResNet50Encoder(nn.Module):
    """
    Backbone ResNet50 với trọng số pretrained từ ImageNet.
    Loại bỏ lớp Global Average Pooling và Fully Connected cuối cùng.
    Input:  (B, 3, 224, 224)  Output: (B, 2048, H, W)
    """
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)

    @property
    def out_channels(self):
        return 2048


class RelationNetwork(nn.Module):
    """
    Mạng so sánh Adaptive – dùng AdaptiveAvgPool2d, hoạt động với mọi spatial size.
    in_channels : feat_channels * 2 (128 cho Conv4, 4096 cho ResNet50)
    """
    def __init__(self, in_channels, hidden_size):
        super(RelationNetwork, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(256, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(64, hidden_size)
        self.fc2  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.gap(out).view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return torch.sigmoid(self.fc2(out))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    # ── Create output directories ────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    log_path = os.path.join(LOG_DIR,
                            f"miniimagenet_{SAMPLE_NUM_PER_CLASS}shot_{CLASS_NUM}way_train_log.csv")
    log_file = open(log_path, "w")
    log_file.write("episode,loss,test_accuracy,h\n")
    print(f"[INFO] Models will be saved to : {MODEL_DIR}")
    print(f"[INFO] Logs  will be saved to  : {LOG_DIR}")

    # ── Step 1: init data folders ────────────────────────────────────────
    print("init data folders")

    if SPLIT_FILE is not None:
        if DATA_ROOT is None:
            raise ValueError("--data_root bắt buộc khi dùng --split_file")
        print(f"[INFO] Đọc split từ JSON (Format 1): {SPLIT_FILE}")
        metatrain_folders, metatest_folders = tg.mini_imagenet_folders_from_split_json(
            SPLIT_FILE, DATA_ROOT, train_key=TRAIN_KEY, test_key=TEST_KEY)
        print(f"       meta-train: {len(metatrain_folders)} classes, "
              f"meta-test: {len(metatest_folders)} classes")

    elif TRAIN_JSON is not None and TEST_JSON is not None:
        if DATA_ROOT is None:
            raise ValueError("--data_root bắt buộc khi dùng --train_json / --test_json")
        print(f"[INFO] Đọc split từ JSON (Format 2): {TRAIN_JSON} / {TEST_JSON}")
        metatrain_folders, metatest_folders = tg.mini_imagenet_folders_from_image_json(
            TRAIN_JSON, TEST_JSON, DATA_ROOT)
        print(f"       meta-train: {len(metatrain_folders)} classes, "
              f"meta-test: {len(metatest_folders)} classes")

    else:
        print("[INFO] Dùng cách đọc gốc (quét thư mục)")
        metatrain_folders, metatest_folders = tg.mini_imagenet_folders()

    _use_image_list = (TRAIN_JSON is not None)
    TaskClass = tg.MiniImagenetTaskFromImageList if _use_image_list else tg.MiniImagenetTask

    # ── Step 2: init neural networks ────────────────────────────────────
    print("init neural networks")
    print(f"[INFO] Backbone   : {BACKBONE}")
    print(f"[INFO] Image size : {IMAGE_SIZE}")

    if BACKBONE == "resnet50":
        feature_encoder = ResNet50Encoder(pretrained=USE_PRETRAINED)
        feat_channels   = feature_encoder.out_channels
        print(f"[INFO] ResNet50 pretrained={USE_PRETRAINED}, out_channels={feat_channels}")
    else:
        feature_encoder = CNNEncoder()
        feat_channels   = feature_encoder.out_channels
        feature_encoder.apply(weights_init)

    relation_network = RelationNetwork(feat_channels * 2, RELATION_DIM)
    relation_network.apply(weights_init)

    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    feature_encoder  = feature_encoder.to(device)
    relation_network = relation_network.to(device)

    feature_encoder_optim      = torch.optim.Adam(feature_encoder.parameters(),  lr=LEARNING_RATE)
    feature_encoder_scheduler  = StepLR(feature_encoder_optim,  step_size=100000, gamma=0.5)
    relation_network_optim     = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    # ── Resume from checkpoint if available ─────────────────────────────
    enc_ckpt = os.path.join(MODEL_DIR,
        f"miniimagenet_{BACKBONE}_feature_encoder_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl")
    net_ckpt = os.path.join(MODEL_DIR,
        f"miniimagenet_{BACKBONE}_relation_network_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl")

    if os.path.exists(enc_ckpt):
        feature_encoder.load_state_dict(
            torch.load(enc_ckpt, map_location=device))
        print("load feature encoder success")
    if os.path.exists(net_ckpt):
        relation_network.load_state_dict(
            torch.load(net_ckpt, map_location=device))
        print("load relation network success")

    # ── Step 3: training loop ────────────────────────────────────────────
    print("Training...")
    last_accuracy = 0.0
    mse = nn.MSELoss().to(device)

    for episode in range(EPISODE):
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        task = TaskClass(
            metatrain_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(
            task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False,
            image_size=IMAGE_SIZE)
        batch_dataloader  = tg.get_mini_imagenet_data_loader(
            task, num_per_class=BATCH_NUM_PER_CLASS, split="test",  shuffle=True,
            image_size=IMAGE_SIZE)

        samples, sample_labels = next(iter(sample_dataloader))
        batches, batch_labels  = next(iter(batch_dataloader))

        # calculate features
        sample_features = feature_encoder(samples.to(device))
        _, C, fH, fW = sample_features.shape   # kích thước thực tế
        sample_features = sample_features.view(
            CLASS_NUM, SAMPLE_NUM_PER_CLASS, C, fH, fW)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        batch_features  = feature_encoder(batches.to(device))

        sample_features_ext = sample_features.unsqueeze(0).repeat(
            BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext  = batch_features.unsqueeze(0).repeat(
            CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext  = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat(
            (sample_features_ext, batch_features_ext), 2).view(
            -1, C * 2, fH, fW)
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

        one_hot_labels = torch.zeros(
            BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM
        ).scatter_(1, batch_labels.view(-1, 1), 1).to(device)

        loss = mse(relations, one_hot_labels)

        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),  0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode + 1) % 100 == 0:
            print(f"episode: {episode+1}  loss: {loss.item():.6f}")

        # ── Periodic evaluation ──────────────────────────────────────────
        if episode % 5000 == 0:
            print("Testing...")
            accuracies = []
            feature_encoder.eval()
            relation_network.eval()
            with torch.no_grad():
                for _ in range(TEST_EPISODE):
                    total_rewards = 0
                    task = TaskClass(
                        metatest_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, 15)
                    sample_dataloader = tg.get_mini_imagenet_data_loader(
                        task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False,
                        image_size=IMAGE_SIZE)
                    test_dataloader   = tg.get_mini_imagenet_data_loader(
                        task, num_per_class=5, split="test", shuffle=False,
                        image_size=IMAGE_SIZE)

                    sample_images, _ = next(iter(sample_dataloader))
                    for test_images, test_labels in test_dataloader:
                        batch_size = test_labels.shape[0]
                        sample_features = feature_encoder(sample_images.to(device))
                        _, C, fH, fW    = sample_features.shape
                        sample_features = sample_features.view(
                            CLASS_NUM, SAMPLE_NUM_PER_CLASS, C, fH, fW)
                        sample_features = torch.sum(sample_features, 1).squeeze(1)
                        test_features   = feature_encoder(test_images.to(device))

                        sample_features_ext = sample_features.unsqueeze(0).repeat(
                            batch_size, 1, 1, 1, 1)
                        test_features_ext   = test_features.unsqueeze(0).repeat(
                            CLASS_NUM, 1, 1, 1, 1)
                        test_features_ext   = torch.transpose(test_features_ext, 0, 1)

                        relation_pairs = torch.cat(
                            (sample_features_ext, test_features_ext), 2).view(
                            -1, C * 2, fH, fW)
                        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

                        _, predict_labels = torch.max(relations, 1)
                        rewards = [
                            1 if predict_labels[j] == test_labels[j] else 0
                            for j in range(batch_size)
                        ]
                        total_rewards += np.sum(rewards)

                    accuracies.append(total_rewards / float(CLASS_NUM * 15))

            feature_encoder.train()
            relation_network.train()

            test_accuracy, h = mean_confidence_interval(accuracies)
            print(f"test accuracy: {test_accuracy:.4f}  h: {h:.4f}")

            log_file.write(f"{episode},{loss.item():.6f},{test_accuracy:.4f},{h:.4f}\n")
            log_file.flush()

            if test_accuracy > last_accuracy:
                torch.save(feature_encoder.state_dict(),  enc_ckpt)
                torch.save(relation_network.state_dict(), net_ckpt)
                print(f"save networks for episode: {episode}")
                last_accuracy = test_accuracy

    log_file.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
