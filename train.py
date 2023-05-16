from model import *

def main():
    
    config =  json.load(open('./nsmc_model_config.json'))

    model = NSMCClassifier(config)
    dm = NSMCDataModule('./')
    dm.setup('fit')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath= './' ,
        filename='{epoch:02d}-{val_acc:.3f}',
        verbose=True,
        save_last=False,
        mode='max',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        mode='max'
    )

    trainer = pl.Trainer(max_epochs=config['n_epochs'], callbacks = [checkpoint_callback, early_stopping])
    trainer.fit(model, dm)

    torch.save(model.state_dict(), './nsmc_clf.pth')

if __name__ == '__main__':
    main()