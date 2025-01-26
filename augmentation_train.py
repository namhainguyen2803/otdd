import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
from models.resnet import ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.method5 import compute_pairwise_distance
from trainer import *
import os
import random
import argparse
import logging

OTDD_MAXSIZE_IMAGENET = None
OTDD_MAXSIZE_CIFAR10 = None
sOTDD_MAXSIZE_IMAGENET = None
sOTDD_MAXSIZE_CIFAR10 = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Arguments for Augmentation')
    parser.add_argument('--parent_dir', type=str, default="saved_augmentation_2", help='Parent directory')
    parser.add_argument('--seed', type=int, default=1, help='Seed')

    args = parser.parse_args()
    parent_dir = f"{args.parent_dir}/aug_{str(args.seed)}"
    os.makedirs(parent_dir, exist_ok=True)
    log_file = f"{parent_dir}/log_seed_{args.seed}.log"

    # Set up logging configuration
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 'w' to write a new file each run, 'a' to append
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Log to console as well
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Using CUDA: {DEVICE}")
    logger.info(f"Parent directory: {parent_dir}")

    def create_data():
        brightness = random.uniform(0.1, 0.9)
        contrast = random.uniform(0.1, 0.9)
        saturation = random.uniform(0.1, 0.9)
        hue = random.uniform(0, 0.5)
        logger.info(f"Random: brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue}")

        datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
        imagenet = load_imagenet(datadir=datadir_tiny_imagenet, 
                                 resize=32, 
                                 tiny=True, 
                                 augmentations=True, 
                                 brightness=brightness, 
                                 contrast=contrast, 
                                 saturation=saturation, 
                                 maxsize=None,
                                 hue=hue)
        imagenet_trainset = imagenet[1]["train"]
        imagenet_testset = imagenet[1]["test"]
        
        datadir_cifar10 = "data/CIFAR10"
        cifar10 = load_torchvision_data("CIFAR10",
                                        valid_size=0, 
                                        download=False, 
                                        maxsize=None, 
                                        datadir=datadir_cifar10)

        def save_data(data_set, saved_tensor_path):
            list_images = [img for img, label in data_set]
            list_labels = [label for img, label in data_set]
            tensor_images = torch.stack(list_images)
            tensor_labels = torch.tensor(list_labels)
            torch.save((tensor_images, tensor_labels), saved_tensor_path)
            logger.info(f"Saved data to {saved_tensor_path} with {len(list_images)} samples.")

        save_data(data_set=imagenet_trainset, saved_tensor_path=f"{parent_dir}/transformed_train_imagenet.pt")
        save_data(data_set=imagenet_testset, saved_tensor_path=f"{parent_dir}/transformed_test_imagenet.pt")
        save_data(data_set=cifar10[1]["train"], saved_tensor_path=f"{parent_dir}/transformed_train_cifar10.pt")
        save_data(data_set=cifar10[1]["test"], saved_tensor_path=f"{parent_dir}/transformed_test_cifar10.pt")

        return {
            "cifar10": {
                "trainloader": cifar10[0]["train"], 
                "testloader": cifar10[0]["test"]
            },
            "imagenet": {
                "trainloader": imagenet[0]["train"], 
                "testloader": imagenet[0]["test"]
            }
        }

    def transfer_learning(train_imagenet_loader, test_imagenet_loader, train_cifar10_loader, test_cifar10_loader, num_epochs_pretrain=300, num_epochs_adapt=30, device=DEVICE):
        logger.info("Starting training on ImageNet...")
        imagenet_feature_extractor = ResNet50().to(device)
        imagenet_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 200).to(device)
        feature_extractor_optimizer = optim.SGD(imagenet_feature_extractor.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        classifier_optimizer = optim.SGD(imagenet_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        for epoch in range(1, num_epochs_pretrain + 1):
            train_loss = train(feature_extractor=imagenet_feature_extractor,
                               classifier=imagenet_classifier,
                               device=device,
                               train_loader=train_imagenet_loader,
                               epoch=epoch,
                               criterion=nn.CrossEntropyLoss(),
                               ft_extractor_optimizer=feature_extractor_optimizer,
                               classifier_optimizer=classifier_optimizer)
            logger.info(f"ImageNet Epoch {epoch}: Train Loss = {train_loss}")

        frozen_module(imagenet_feature_extractor)
        imagenet_acc_no_adapt = test_func(imagenet_feature_extractor, imagenet_classifier, device, test_imagenet_loader)
        logger.info(f"ImageNet Test Accuracy: {imagenet_acc_no_adapt}")

        logger.info("Starting transfer learning on CIFAR10...")
        cifar10_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 10).to(device)
        cifar10_classifier_optimizer = optim.SGD(cifar10_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        for epoch in range(1, num_epochs_adapt + 1):
            train_loss = train(feature_extractor=imagenet_feature_extractor,
                               classifier=cifar10_classifier,
                               device=device,
                               train_loader=train_cifar10_loader,
                               epoch=epoch,
                               criterion=nn.CrossEntropyLoss(),
                               ft_extractor_optimizer=None,
                               classifier_optimizer=cifar10_classifier_optimizer)
            logger.info(f"CIFAR10 Epoch {epoch}: Train Loss = {train_loss}")

        cifar10_acc_adapt = test_func(imagenet_feature_extractor, cifar10_classifier, device, test_cifar10_loader)
        logger.info(f"CIFAR10 Test Accuracy after Adaptation: {cifar10_acc_adapt}")

    logger.info("Creating data...")
    DATA_DICT = create_data()
    logger.info("Data creation completed.")

    imagenet_trainloader = DATA_DICT["imagenet"]["trainloader"]
    cifar10_trainloader = DATA_DICT["cifar10"]["trainloader"]
    imagenet_testloader = DATA_DICT["imagenet"]["testloader"]
    cifar10_testloader = DATA_DICT["cifar10"]["testloader"]

    transfer_learning(train_imagenet_loader=imagenet_trainloader, 
                      test_imagenet_loader=imagenet_testloader, 
                      train_cifar10_loader=cifar10_trainloader, 
                      test_cifar10_loader=cifar10_testloader)

if __name__ == "__main__":
    main()
