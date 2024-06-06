import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from utils.dataset import SSIRGBDataset
from torch.utils.data import DataLoader
from fas_models.fusion_models.resnet_fusion import SMN_ResNet18_AttentionFusion_ as Net
from utils.model_wrapper import ModelWrapper, APCER, BPCER, BinaryAUC


def build_dataset(batch_size=16, image_size=128, crop_scale=(0.5, 0.8), protocol=1):
    trainset = SSIRGBDataset(
        ssi_root_dir='./data/datasets/ssi',
        rgb_root_dir='./data/datasets/rgb',
        label_path='./data/datasets/train_p{:d}.txt'.format(protocol),
        image_size=image_size,
        crop_scale=crop_scale,
        training=True,
        cache_images=True,
        positive_rep_rate=4,
        inter_class_mixup=True,
        intra_class_mixup=True,
    )

    valset = SSIRGBDataset(
        ssi_root_dir='./data/datasets/ssi',
        rgb_root_dir='./data/datasets/rgb',
        label_path='./data/datasets/val_p{:d}.txt'.format(protocol),
        image_size=image_size,
        crop_scale=crop_scale,
        training=False,
        cache_images=True,
        using_existing_cache=trainset.bbox_dict,
    )

    testset = SSIRGBDataset(
        ssi_root_dir='./data/datasets/ssi',
        rgb_root_dir='./data/datasets/rgb',
        label_path='./data/datasets/test_p{:d}.txt'.format(protocol),
        image_size=image_size,
        crop_scale=crop_scale,
        training=False,
        cache_images=True,
        using_existing_cache=trainset.bbox_dict,
    )

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(Model=Net, name='smn-resnet-s-fusion', verbose=1, batch_size=60, do_validation=True, protocol=1,
          data_parallel=False, **model_kwargs):
    batch_size = batch_size
    epochs = 30
    model_name = name

    trainloaer, valloader, testloader = build_dataset(batch_size=batch_size, protocol=protocol)
    if not do_validation:
        valloader = None
    model = Model(**model_kwargs)
    device = torch.device('cuda')

    if data_parallel:
        model = torch.nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    loss_fcn = torch.nn.functional.binary_cross_entropy
    apcer = APCER(device=torch.device('cpu'))
    bpcer = BPCER(device=torch.device('cpu'))
    auc = BinaryAUC(device=torch.device('cpu'))

    trainer = ModelWrapper(model, device)
    trainer.compile(
        optimizer=optim,
        loss_fcn=loss_fcn,
        metrics=[apcer, bpcer, auc],
        schedule=schedule,
    )

    trainer.train(
        train_data=trainloaer,
        val_data=valloader,
        epochs=epochs,
        save_path='./checkpoints/{:s}.pth'.format(model_name),
        verbose=verbose,
    )
    rtv = trainer.evaluate(testloader, verbose=verbose)

    if data_parallel:
        torch.save(model.module.state_dict(), './checkpoints/{:s}.pth'.format(model_name))
    else:
        torch.save(model.state_dict(), './checkpoints/{:s}.pth'.format(model_name))

    return rtv

if __name__ == '__main__':
    train()
