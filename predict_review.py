from model import *

def predcit_review(text):
    config =  json.load(open('./nsmc_model_config.json'))

    model = NSMCClassifier(config)
    model.load_state_dict(torch.load(f = './nsmc_clf.pth'))
    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    m = Mecab()

    text = ' '.join(m.morphs(text))
    output = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length= 80,
        return_attention_mask = True,
        return_token_type_ids=False
    )

    input_ids = output['input_ids']
    attention_mask = output['attention_mask']

    loss, output = model(input_ids, attention_mask)

    prediction = torch.argmax(output, dim=1).numpy().tolist()
    if prediction[0] == 0:
        prediction = '부정'
    else:
        prediction = '긍정'

    return prediction

