import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from utils.dataset_rec import HSIDataset
from torch.utils.data import DataLoader
from fas_models.ssi_models.smn_unet import SMN_Unet_M as Net
from utils.model_wrapper import ModelWrapper, SAM, PSNR, SSIM


def build_dataset(batch_size=8, image_size=128):
    trainset = HSIDataset(
        label_path='./data/datasets/hsi_measured/train.txt',
        root_dir='./data/datasets/hsi_measured',
        image_size=image_size,
        repetition=20,
        training=True,
        mixup=True,
    )

    valset = HSIDataset(
        label_path='./data/datasets/hsi_measured/val.txt',
        root_dir='./data/datasets/hsi_measured',
        image_size=image_size,
        repetition=20,
        training=False,
        mixup=False,
    )

    testset = HSIDataset(
        label_path='./data/datasets/hsi_measured/test.txt',
        root_dir='./data/datasets/hsi_measured',
        image_size=image_size,
        repetition=20,
        training=False,
        mixup=False,
    )

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(Model=Net, name='smn-unet', verbose=1, batch_size=32, data_parallel=False, do_validation=True, **model_kwargs):
    batch_size = batch_size
    epochs = 30
    model_name = name

    trainloader, valloader, testloader = build_dataset(batch_size=batch_size)
    if not do_validation:
        valloader = None
    model = Model(**model_kwargs)
    device = torch.device('cuda')

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)
    # schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=30, T_mult=1)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    loss_fcn = torch.nn.functional.mse_loss
    psnr = PSNR(device=torch.device('cpu'))
    sam = SAM(device=torch.device('cpu'))
    ssim = SSIM(device=torch.device('cpu'))

    trainer = ModelWrapper(model, device)
    trainer.compile(
        optimizer=optim,
        loss_fcn=loss_fcn,
        metrics=[psnr, sam, ssim],
        schedule=schedule,
    )

    trainer.train(
        train_data=trainloader,
        val_data=valloader,
        epochs=epochs,
        save_path='./checkpoints/{:s}.pth'.format(model_name),
        verbose=verbose,
    )
    rtv = trainer.evaluate(testloader, verbose=verbose)

    model.eval()
    trainer.evaluate(testloader, verbose=verbose)

    if data_parallel:
        torch.save(model.module.state_dict(), './checkpoints/{:s}.pth'.format(model_name))
    else:
        torch.save(model.state_dict(), './checkpoints/{:s}.pth'.format(model_name))

    return rtv


if __name__ == '__main__':
    train()