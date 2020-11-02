import pandas as pd
from tqdm import tqdm
import os
import re
import json
import numpy as np
from konlpy.tag import Okt

'''
 데이터 전처리
'''
FILTERS = "([~.,!?\"':;)(])"
CHANGE_FILTER = re.compile(FILTERS) # 미리 Complie
PAD, PAD_INDEX = "<PAD>", 0 # 패딩 토큰
STD, STD_INDEX = "<SOS>", 1 # 시작 토큰
END, END_INDEX = "<END>", 2 # 종료 토큰
UNK, UNK_INDEX = "<UNK>", 3 # 사전에 없음
MARKER = [PAD,STD,END,UNK]
MAX_SEQUNECE = 25


# Data reading
def load_data(path):
    df = pd.read_csv(path,header=0)
    question, answer = list(df['Q']),list(df['A'])
    return question, answer

# Tokenizing
def data_tokenizer(data):
    words = []
    for sentence in data:
        # 미리 컴파일한 특수문자를 제거하는 코드
        sentence = re.sub(CHANGE_FILTER,"",sentence)
        for word in sentence.split():
            words.append(word) 
    # 공백 기준으로 단어를 나눠서 Return
    return [word for word in words if word]

# 형태소 분리 
def prepro_like_morphlized(data):
    morph_analyzer= Okt()
    results = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ','')))
        results.append(morphlized_seq)
    return results

# 단어 사전 만드는 함수
def load_vocabulary(path, vocab_path):
    vocabulary_list = []

    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            df = pd.read_csv(path,encoding='utf-8')
            question, answer = list(df['Q']),list(df['A'])
            data = []
            data.extend(question)
            data.extend(answer)
            # Tokenizing 
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER

