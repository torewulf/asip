import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import utils
from unet import UNet
from dataloader import ASIPDS


def train(model, loss_fn, optimizer, scheduler, trainloader, valloader, epochs, device, save_as=None):    
    if save_as:
        if not os.path.isdir(save_as):
            os.mkdir(save_as)
    
    model = model.to(device)
    session_log = pd.DataFrame(columns=['epoch', 'lr', 'train loss', 'val loss'])
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            inputs, targets = batch
            
            S1, _, DST, AMSR = inputs
            preds = model(S1.to(device), DST.to(device), AMSR.to(device))
            loss = loss_fn(preds, targets.squeeze(dim=1).to(device))

            loss.backward()
            optimizer.step()
            
            training_loss += loss.data.item()*targets.size(0)
            del loss, preds
        training_loss /= len(trainloader.dataset)
            
        valid_loss = 0.0        
        model.eval()
        for batch in valloader:
            inputs, targets = batch

            S1, _, DST, AMSR = inputs
            preds = model(S1.to(device), DST.to(device), AMSR.to(device))
            loss = loss_fn(preds, targets.squeeze(dim=1).to(device))

            valid_loss += loss.data.item()*targets.size(0)
            del loss, preds
        valid_loss /= len(valloader.dataset)

        if save_as:
            session_log.loc[epoch] = [epoch,
                                      optimizer.param_groups[0]['lr'],
                                      training_loss,
                                      valid_loss]
            session_log.to_csv(os.path.join(save_as, 'session_log.csv'), index=False)
            torch.save(model.state_dict(), os.path.join(save_as, save_as + '.pt'))
        
        scheduler.step(training_loss)


if __name__ == "__main__":

    SESSION = 'unet_demo'
    NET = UNet(in_channels=2, out_channels=11)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 60
    BATCH_SIZE = 32
    LR = 1e-3

    patches_folder = '/data/users/twu/ds-2/patches'
    scenes = utils.get_scene_paths(patches_folder)
    test_scenes = scenes[:20]
    train_scenes, val_scenes = train_test_split(scenes[20:], train_size=0.9, random_state=42)
    
    train_ds = ASIPDS(scenes=train_scenes, crop_size=600, downsample=True) 
    val_ds = ASIPDS(scenes=val_files, crop_size=600, downsample=True) 
    trainloader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(NET.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=False, min_lr=0.00001)

    train(model=NET,
          loss_fn=loss_fn,
          optimizer=optimizer,
          scheduler=scheduler,
          trainloader=trainloader,
          valloader=valloader,
          epochs=EPOCHS,
          device=DEVICE,
          save_as=SESSION)
