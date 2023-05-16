from model import *

def main():
    config =  json.load(open('./nsmc_model_config.json'))

    model = NSMCClassifier(config)
    model.load_state_dict(torch.load(f = './nsmc_clf.pth'))
    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    
    nsmc_dm = NSMCDataModule('./')

    nsmc_dm.setup('test')

    trainer = pl.Trainer()
    trainer.test(model, nsmc_dm)

if __name__ == '__main__':
    main()