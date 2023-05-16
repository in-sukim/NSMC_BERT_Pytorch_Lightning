from konlpy.tag import Mecab
from tqdm import tqdm
import re

def preprocess(df):
  mecab = Mecab()
  mecab_texts = []
  for i in tqdm(df['document']):
    text = re.sub('[^a-zA-Z가-힣\s]', '', i)
    clean_text = ' '.join(mecab.morphs(text))
    mecab_texts.append(str(clean_text))

  df['document'] = mecab_texts
  return df