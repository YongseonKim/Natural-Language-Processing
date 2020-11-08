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

# 단어 사전을 불러오는 함수
def load_vocabulary(path, vocab_path):
    vocabulary_list = []
    # vocab path가 없고 -- 단어 사전파일이 없고
    if not os.path.exists(vocab_path):
        # Raw데이터를 불러와서 사전을 만든다.
        # if (os.path.exists(path)):
        df = pd.read_csv(path,encoding='utf-8')
        question, answer = list(df['Q']),list(df['A'])
        data = []
        data.extend(question)
        data.extend(answer)
        # Tokenizing 
        words = data_tokenizer(data)
        words = list(set(words))
        words[:0] = MARKER # 사전에 정의한 토큰을 단어 리스트 앞에 추가
            # print(vocab_path)
        # print(words)
        with open(vocab_path, 'w', encoding = 'utf-8') as vocabulary_file:
            for word in words:
                # print(word)
                vocabulary_file.write(word + '\n')

    
        
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            # print(line)
            vocabulary_list.append(line.strip())
    # print(vocabulary_list) 
    word2idx, idx2word = make_vocabulary(vocabulary_list)
    
    return word2idx, idx2word, len(word2idx)

 
def make_vocabulary(vocabulary_list):
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}

    return word2idx, idx2word

# 인코더와 디코더 부분 처리하기
def enc_processing(value, dictionary):
    sequences_input_index = []
    sequences_length = []

    for sequence in value :
        sequence = re.sub(CHANGE_FILTER,"",sequence)
        sequence_index = []
        
        for word in sequence.split(): # 공백 기준으로 word를 구분
            if dictionary.get(word) is not None : # 사전에 있으면
                sequence_index.extend([dictionary[word]]) # index 값 쓰고
            else:
                sequence_index.extend([dictionary[UNK]])
        # 길이 제한
        if len(sequence_index) > MAX_SEQUNECE:
            sequence_index = sequence_index[:MAX_SEQUNECE]

        sequences_length.append(len(sequence_index)) # 이 문장의 길이 저장
        # Padding 추가
        # "안녕"  → "안녕,<PAD>,<PAD>,<PAD>,<PAD>"
        
        sequence_index += (MAX_SEQUNECE - len(sequences_input_index))*[dictionary[PAD]]
        
        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index), sequences_length

# Decoder input

def dec_output_processing(value, dictionary):
    sequences_output_index = []
    sequences_length = []

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER,"",sequence)
        sequence_index = []
        # 앞부분에 시작을 알리는 토큰 넣기
        sequence_index = [dictionary[STD]]+[dictionary[word] for word in sequence.split()]

        if len(sequence_index) > MAX_SEQUNECE:
            sequence_index = sequence_index[:MAX_SEQUNECE]

        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUNECE - len(sequence_index))*[dictionary[PAD]]

        sequences_output_index.append(sequence_index)
    return np.asarray(sequences_output_index), sequence_index

# 디코더 Target 값 전처리
def dec_target_processing(value,dictionary):
    sequences_target_index = []
    for sequence in value :
        sequence = re.sub(CHANGE_FILTER,"", sequence)
        sequence_index = [dictionary[word] for word in sequence.split() ]
        if len(sequence_index)>= MAX_SEQUNECE:
            # 이부분이 Decoder 입력값 전처리와 다른점
            sequence_index = sequence_index[:MAX_SEQUNECE-1] + [dictionary[END]] #마지막에 END xhzms
        else :
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUNECE - len(sequence_index))*[dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)

if __name__ == "__main__":
    PATH = 'data_in/ChatBotData.csv'
    VOCAB_PATH = 'data_in/vocabulary.txt'
    # 데이터 부르기
    inputs, outputs = load_data(PATH)
    # 단어 사전 부르기
    # 토크나이저를 사용하여 처리하도록 변경하기
    char2idx, idx2char, vocab_size = load_vocabulary(PATH,VOCAB_PATH)
    # print(char2idx)

    # encoder/decoder input /target
    index_inputs, input_seq_len = enc_processing(inputs, char2idx)
    index_outputs, output_seq_len = dec_output_processing(outputs, char2idx)
    index_targets =  dec_target_processing(outputs, char2idx)

    data_configs = {}
    data_configs['char2idx'] =char2idx
    data_configs['idx2char'] = idx2char
    data_configs['vocab_size'] = vocab_size
    data_configs['pad_symbol'] = PAD
    data_configs['std_symbol'] = STD
    data_configs['end_symbol'] = END
    data_configs['unk_symbol'] = UNK

    DATA_IN_PATH = './data_in/'
    np.save(open(DATA_IN_PATH+'train_inputs.npy','wb'), index_inputs)
    np.save(open(DATA_IN_PATH+'train_outputs.npy','wb'), index_inputs)
    np.save(open(DATA_IN_PATH+'train_targets.npy','wb'), index_inputs)

    json.dump(data_configs, open(DATA_IN_PATH+'data_configs.json','w'))

index_outputs.shape
index_targets.shape


