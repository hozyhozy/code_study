#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
import string
import re

from sklearn.model_selection import cross_val_score, train_test_split

from konlpy.tag import Okt
# from konlpy.tag import Mecab 
from kss import split_sentences   
#from pykospacing import spacing
from gensim.models import Word2Vec

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import Input,Conv2D, MaxPool2D, MaxPooling1D
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import Embedding, Dense,GRU,Flatten, LSTM,Conv1D, GlobalMaxPooling1D, Dropout, GlobalAveragePooling1D

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from kerastuner.tuners import RandomSearch # 랜덤서치를 합니다


# In[ ]:


data = pd.read_excel('준지도예시.xlsx')
data=data[['review', 'target']]
data


# # 데이터 줄이기

# In[ ]:


def reduce_memory_usage(df):
    
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024**2 
    
    return df


# In[4]:


data=reduce_memory_usage(data)


# # train, test data 정의

# In[8]:


train=data[:200]
test=data[200:330]

train['mark'] = 'train'
test['target'] = 0 
test['mark']='test'


# In[9]:


df=pd.concat([train,test])
df


# # 전처리+토큰화
# - 띄어쓰기 필요.. 안돌아가..요..
# 

# In[10]:


# 전처리 후 혹시나 있을 짧은 (분석에는 불필요한) 행은 삭제합니다
# 문단단위면 없을 것으로 생각되긴 합니다 (문장단위면 안녕!, 어서와! 이런 쓸데없는 문장들이 있습니다)

LENGTH = 10 # 길이 지정해줍니다

count = 0
for i in df['review']:
    if len(i) < LENGTH:
        count+=1
        print(i)
print(count)
print('짧은 글 비율:',np.round(count/len(df),4))


# In[11]:


ix = df['review'].str.len() < LENGTH # 짧은 리뷰 표시 후 제거
df = df.loc[~ix]
df.info()


# In[12]:


okt = Okt() 

def preprocess_okt(text):
    #     text = spacing(text) # 띄어쓰기 보정 위에서 했으면 필요없습니다
    pos_words = okt.pos(text, stem=True)
    words = [word for word, tag in pos_words if tag in ['Noun', 'Adjective', 'Verb', 'KoreanParticle', 'VerbPrefix'] ]
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','것','수','있다','않다','이다','이었다', '안','억','이건','정도','개']
    stopped_words = [w for w in words if not w in stopwords]
    return ', '.join(stopped_words)


# In[13]:


df['review_token']=df['review'].apply(preprocess_okt)
df['review_token']= [[_] for _ in df.review_token]

#빈 값 확인
empty_index = [index for index, sentence in zip(df.index,df['review_token']) if len(sentence) < 1]
print(empty_index) 

df


# # 정수 인코딩
# - https://wikidocs.net/69141

# In[14]:


total_token= []
for _ in df.review_token:
    total_token.extend(_)
total_token


# In[15]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(total_token) 
word2idx = tokenizer.word_index
idx2word = {value : key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(total_token)
print(encoded[:2])


# In[16]:


vocab_size = len(word2idx) + 1 
print('단어 집합의 크기 :', vocab_size)


# # 네거티브 샘플링을 통한 데이터 셋 구성

# - https://wikidocs.net/69141

# In[17]:


from tensorflow.keras.preprocessing.sequence import skipgrams
# 네거티브 샘플링
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]


# In[18]:


# 첫번째 샘플인 skip_grams[0] 내 skipgrams로 형성된 데이터셋 확인
# 0은 네거티브, 1은 주변 단어의 관계를 가지는 경우
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          idx2word[pairs[i][0]], pairs[i][0], 
          idx2word[pairs[i][1]], pairs[i][1], 
          labels[i]))


# In[19]:


print('전체 샘플 수 :',len(skip_grams))


# # Skip-Gram with Negative Sampling(SGNS) 구현하기

# - https://wikidocs.net/69141

# In[20]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG


# In[21]:


embedding_dim = 100

# 중심 단어를 위한 임베딩 테이블
w_inputs = Input(shape=(1, ), dtype='int32')
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

# 주변 단어를 위한 임베딩 테이블
c_inputs = Input(shape=(1, ), dtype='int32')
context_embedding  = Embedding(vocab_size, embedding_dim)(c_inputs)

dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation('sigmoid')(dot_product)

model = Model(inputs=[w_inputs, c_inputs], outputs=output)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')

for epoch in range(1, 10):
    loss = 0
    for _, elem in enumerate(skip_grams):
        first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [first_elem, second_elem]
        Y = labels
        loss += model.train_on_batch(X,Y)  
    print('Epoch :',epoch, 'Loss :',loss)


# In[22]:


import gensim

f = open('vectors.txt' ,'w')
f.write('{} {}\n'.format(vocab_size-1, embedding_dim))
vectors = model.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()

# 모델 로드
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
w2v.most_similar(positive=['냉장고'])


# # 훈련에 들어갈 train, test 셋 만들기

# In[23]:


#삭제후 index 리셋 (삭제되는 글이 있으면 뒤에 코드에서 에러)
df = df.reset_index().drop('index',axis=1)

### 필요 없는 것 삭제
train_=df[df.mark=='train']
test_=df[df.mark=='test']
target=pd.DataFrame({'target':train_.target})

train_ = train_.drop(['review_token','mark','target'], axis =1)
test_ = test_.drop(['review_token','mark', 'target'], axis = 1)
train_ = train_.reset_index(drop=True)
test_ = test_.reset_index(drop=True)


def token(x):
    total_token= []
    for n, _ in enumerate(x.review):
        total_token.append( _)
    return total_token

#target = np.array(target.values).ravel() # ravel을 하지 말아야하는건가.. 
train = token(train_)
test= token(test_)


# In[24]:


print(len(train), len(target), len(test))


# In[25]:


train[0]


# # 모델에 들어갈 데이터 정수화

# In[26]:


tokenizer = Tokenizer()             
tokenizer.fit_on_texts(train)
tokenizer 


# In[27]:


print("len(tokenizer) :",len(list(tokenizer.word_index)))


# In[28]:


# 코드 출처  4) 정수인코딩 부분
# https://wikidocs.net/44249

# 희귀 단어를 삭제할 지 여부를 결정하는 부분입니다 
# (모델 학습에 불필요하다고 생각된 단어들을 삭제한다는 의미입니다)
# threshold에 따라 어느정도에서 짜를 지 선택할 수 있고
# 그에 따라 토크나이저에서의 VOCAB_SIZE가 정해집니다

threshold = 1
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 tokenizer.word_counts.items() 통해 key(단어)와 value(빈도수)로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
VOCAB_SIZE = total_cnt - rare_cnt + 2
print('최종 단어 집합의 크기 :',VOCAB_SIZE)


# In[29]:


# token화
tokenizer = Tokenizer(VOCAB_SIZE, oov_token = 'OOV') # 자주나오는 상위 vocab_size만 학습, 학습되지 않은 단어는 1로 처리
tokenizer.fit_on_texts(train)
word_index = tokenizer.word_index
print(len(word_index))


# In[30]:


# OOV 설정했기 때문에 VOCAB_SIZE 초과하는 단어들은 OOV 1 정수로 인코딩 처리됩니다.
#Training set token with text_to_sequences.
x_train_tokens = tokenizer.texts_to_sequences(train)
test_tokens = tokenizer.texts_to_sequences(test)

# pos tagging 된거 길이 = 토큰화 된거 길이 는 같아야합니다
print(len(train), len(x_train_tokens))
print(len(test), len(test_tokens))

x_train_tokens[50]


# # 제일 긴 문장의 길이로 padding 맞추기
# - 적정한 길이로 문장 자르기

# In[31]:


# 코드 출처  6) 제로 패딩
# https://wikidocs.net/44249

print('문장의 최대 길이 :',max(len(l) for l in x_train_tokens))
print('문장의 평균 길이 :',sum(map(len, x_train_tokens))/len(x_train_tokens))
plt.hist([len(s) for s in x_train_tokens], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


# In[32]:


# 삭제되는 비중을 알려주는 함수
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))


# In[33]:


max_len = 500  # 이 부분에 숫자를 바꾸며 적정 길이를 선택합니다
below_threshold_len(max_len, x_train_tokens)
count = 0
cutted = 0

for i in range(len(x_train_tokens)):
    s = x_train_tokens[i]
    count += 1
    if len(s) > max_len:
        cutted += 1
        
print('Total number of samples:', count)
print('길이 짤리는 문장 개수, 비율:', cutted, (cutted/count)*100)


# In[34]:


MAX_LEN = max_len # 정해준 길이에 맞춰 제로패딩해줍니다

#패딩
x_train_pad = pad_sequences(x_train_tokens, maxlen=MAX_LEN)
test_pad = pad_sequences(test_tokens, maxlen=MAX_LEN)
#Zero is added before the values given in the padding operation.
print("x_train_pad.shape :",x_train_pad.shape)
print("test_pad.shape :",test_pad.shape)
print('='*92)
print("x_train_tokens :",x_train_tokens[0])
print("x_train_pad :",x_train_pad[0])


# In[35]:


# 토큰을 다시 문자로 --> 쓸일 없지만.. 혹시나 나중에 쓰일까봐 넣어놓음
# idx = tokenizer.word_index
# inverse_map = dict(zip(idx.values(), idx.keys()))

# #토큰을 다시 텍스트로!
# def tokens_to_string(tokens):
#     words = [inverse_map[token] for token in tokens if token!=0]
#     text = ' '.join(words)
#     return text

# tokens_to_string(x_train_tokens[50])


# # 이제 Word2vec을 불러와서 모델 데이터에 적용

# In[36]:


#Word2Vec 파일 불러오기
word2vec = {}
with open('vectors.txt', encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec


# In[37]:


num_words = len(list(tokenizer.word_index))

#임베딩 매트릭스(word2vec이랑 맞추기)
embedding_matrix = np.random.uniform(-1, 1, (VOCAB_SIZE, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
embedding_matrix.shape


# In[38]:


embedding_matrix[3]


# # 드디어 모델 적용!!
# ## 준지도 학습 
# - 1) LSTM을 돌려서 예측값을 추출
# - 2) CNN 돌려서 예측값을 추출
# - 3) LSTM vs. CNN predict에서 일치하지 않은 것들을 GRU의 Test Set으로 보내고 일치하는 것은 train셋으로 넣는다
# ** tuner Search 기능에 validation set을 지정해주는 기능이 존재함

# # Model 1: LSTM
# ## project_name 꼭꼭 새로 정해줘야함!!

# In[39]:


print(len(x_train_pad), len(target))


# In[40]:


target = target.astype(np.float32)
target['target'].value_counts()


# In[41]:


# tuner version
# 밑에 설정된 step의 크기, 노드 학습rate dropout rate 의 min,max범위 등은 직접 변경하셔도 됩니다
EMBEDDING_DIM=100
def build_lstm(hp):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_LEN, trainable=False))
    model.add(LSTM(units=hp.Int('units1', min_value=4, max_value=128,step=16)))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=256,step=32), activation='relu'))
    model.add(Dropout(rate = hp.Float('drop_out_rate', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-1, sampling='log')),
                  loss = 'binary_crossentropy',
                  metrics=['acc'])
    return model


# In[42]:


tuner = RandomSearch(
    build_lstm,
    objective='val_acc',
    max_trials = 15,  #  total number of trials 총 15번 랜덤서치를 합니다
    executions_per_trial=2, # the number of models that should be built and fit for each trial
    project_name ='lstm5') # 현재 주피터파일이 있는 위치에 lstm1이라는 폴더가 생성됩니다. 

# ***모델 다시 돌릴때 주의사항
# 모델을 다시 돌릴땐 project_name을 바꾸던지 현재 위치(폴더)를 바꿔야 돌아갑니다
# 아니면 이전에 만들어진 파일을 삭제하고 다시 돌리면 됩니다
# 폴더 안에는 test해본 모델들의 정보가 기록됩니다


# In[43]:


tuner.search_space_summary() # 랜덤서치의 범위를 요약해서 보여줍니다


# In[44]:


# 모델 input으로 정수인코딩 된 train 데이터와 라벨이 들어갑니다
# epoch, validation_split, patience 등 하이퍼파라미터들을 원하시는대로 조정합니다 

tuner.search(x_train_pad, target, epochs=15, validation_split=0.25, 
             callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3)]) 


# In[45]:


tuner.results_summary() 

# Score는 val_acc 를 의미합니다
# 탑 10개 모델의 정보를 출력해줍니다


# In[46]:


# 가장 성능이 좋았던 모델을 저장해줍니다 
lstm_best_model = tuner.get_best_models(num_models=1)[0]


# In[ ]:





# # Model 2: CNN

# In[47]:


def build_cnn(hp):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_LEN, trainable=False))
    model.add(Dropout(rate = hp.Float('drop_out_rate1', 0.1, 0.5, step=0.1)))
    
    filters_ = hp.Int('filters', 16, 128, step=32)
    kernel_size_ = hp.Choice('kernerl_size',values=[2,3,4,5])
    model.add(Conv1D(filters = filters_, kernel_size = kernel_size_,
                     strides = 1,padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=hp.Int('units', min_value=16, max_value=256,step=32), activation='relu'))
    model.add(Dropout(rate = hp.Float('drop_out_rate2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-1, sampling='log')),
                  loss = 'binary_crossentropy',
                  metrics=['acc'])
    return model


# In[48]:


tuner = RandomSearch(
    build_cnn,
    objective='val_acc',
    max_trials = 15,  #  total number of trials
    executions_per_trial=2, # the number of models that should be built and fit for each trial
    project_name ='cnn6')


# In[49]:


tuner.search_space_summary()


# In[50]:


tuner.search(x_train_pad,target, epochs=15, validation_split=0.25, 
             callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3)]) 


# In[51]:


tuner.results_summary()


# In[52]:


cnn_best_model = tuner.get_best_models(num_models=1)[0]


# # LSTM & CNN 1차 합의

# # Model1. Threshold 정하기
# - Train 데이터로

# In[53]:


# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
from sklearn.metrics import f1_score
model1_ = lstm_best_model.predict(x_train_pad) # model1 결과

scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4,0.81,0.01):
    print(f'{threshold:.02f}, ',end='')
    preds = (model1_.reshape((-1))>threshold).astype('int')
    m = f1_score(target.to_numpy().reshape((-1)), preds, average='macro')   
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score:
        best_score = m
        best_threshold = threshold


# In[54]:


import matplotlib.pyplot as plt

# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()


# # Model2. Threshold 정하기

# In[55]:


# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
model2_ = cnn_best_model.predict(x_train_pad) # model2 결과

scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4,0.81,0.01):
    print(f'{threshold:.02f}, ',end='')
    preds = (model2_.reshape((-1))>threshold).astype('int')
    m = f1_score(target.to_numpy().reshape((-1)), preds, average='macro')   
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score:
        best_score = m
        best_threshold = threshold


# In[56]:


import matplotlib.pyplot as plt

# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()


# # 위 그래프에서 나온 threshold 기준으로 to_binary함수 변경

# In[60]:


# 확률을 이산변수로
def model_1_to_binary(result):
    for i in range(len(result)):
        if result[i]>= 0.47: # threshold 조정이 필요하다면 여기서 조정하면 됩니다
            result[i] = 1
        else:
            result[i] = 0
    return result


def model_2_to_binary(result):
    for i in range(len(result)):
        if result[i]>= 0.4: # threshold 조정이 필요하다면 여기서 조정하면 됩니다
            result[i] = 1
        else:
            result[i] = 0
    return result


# 합의 여부를 출력
def agreement(df):
    x = df[0]
    y = df[1]
    if x == y:
        return True
    else:
        return False


# In[61]:


#Test 데이터 넣어서 예측
model1_result = lstm_best_model.predict(test_pad) # model1 결과
model2_result = cnn_best_model.predict(test_pad) # model2 결과

model1_result = model_1_to_binary(model1_result) # 임계값을 기준으로 0,1로 변환
model2_result = model_2_to_binary(model2_result) # 임계값을 기준으로 0,1로 변환

#model1, model2 결과를 한 dataframe에 저장
result1 = pd.DataFrame(model1_result, columns=['model1']) 
result1['model2'] = model2_result
#result1.index = X_test_pad_index

# aggrement 함수로 합의 여부 출력
result1['accepted'] = result1.apply(agreement,axis=1) # 합의 여부 출력
result1['accepted'].value_counts() # True는 합의된 개수를 의미합니다

success = result1[result1['accepted']==True].index  # 합의된 행의 인덱스
fail = result1[result1['accepted']==False].index

print('Model_3에 Train으로 편입될 합의된 인덱스: ',len(success))
print('Model_3에 Test로 편입될 합의되지 않은 인덱스: ',len(fail))


# In[65]:


success


# # 최종 확정을 위한 새로운 데이터 생성

# In[66]:


# 합의된 결과를 기반으로 test 데이터에서 해당하는 행을 추출
# success는 train에, fail은 test에 속하는 인덱스

# 최종 Train 데이터셋
test_agreement = test_['review'][success]
test_agreement=pd.DataFrame({'review': test_agreement})
final_train = pd.concat([train_, test_agreement], axis=0, ignore_index=True)
final_train=final_train.review
print(len(test_agreement),len(final_train))

#최종 Test 데이터셋
final_test= test_['review'][fail]
print(len(final_test))

# 최종 train target
y_agree = result1['model1'][success]
y_agree=pd.DataFrame({'target': y_agree})
final_target = pd.concat([target, y_agree], axis=0, ignore_index=True) 
print(len(final_target), len(y_agree))
print('전체 데이터:', f'{len(final_test)+len(final_train)}')


# # 다시 정수화+패딩

# In[67]:


tokenizer = Tokenizer()             
tokenizer.fit_on_texts(final_train)
print("len(tokenizer) :",len(list(tokenizer.word_index)))

threshold = 1
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 tokenizer.word_counts.items() 통해 key(단어)와 value(빈도수)로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
VOCAB_SIZE = total_cnt - rare_cnt + 2
print('최종 단어 집합의 크기 :',VOCAB_SIZE)

# token화
tokenizer = Tokenizer(VOCAB_SIZE, oov_token = 'OOV') # 자주나오는 상위 vocab_size만 학습, 학습되지 않은 단어는 1로 처리
tokenizer.fit_on_texts(final_train)
word_index = tokenizer.word_index
print(len(word_index))

# OOV 설정했기 때문에 VOCAB_SIZE 초과하는 단어들은 OOV 1 정수로 인코딩 처리됩니다.
#Training set token with text_to_sequences.
x_train_tokens = tokenizer.texts_to_sequences(final_train)
test_tokens = tokenizer.texts_to_sequences(final_test)

# pos tagging 된거 길이 = 토큰화 된거 길이 는 같아야합니다
print(len(train), len(x_train_tokens))
print(len(test), len(test_tokens))

x_train_tokens[20]


# In[68]:


print('문장의 최대 길이 :',max(len(l) for l in final_train))
print('문장의 평균 길이 :',sum(map(len, final_train))/len(x_train_tokens))
plt.hist([len(s) for s in x_train_tokens], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


# In[69]:


# 삭제되는 비중을 알려주는 함수
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
    
max_len = 500  # 이 부분에 숫자를 바꾸며 적정 길이를 선택합니다
below_threshold_len(max_len, x_train_tokens)
count = 0
cutted = 0

for i in range(len(x_train_tokens)):
    s = x_train_tokens[i]
    count += 1
    if len(s) > max_len:
        cutted += 1
        
print('Total number of samples:', count)
print('길이 짤리는 문장 개수, 비율:', cutted, (cutted/count)*100)

# 패딩
MAX_LEN = max_len # 정해준 길이에 맞춰 제로패딩해줍니다

#패딩
x_train_pad = pad_sequences(x_train_tokens, maxlen=MAX_LEN)
test_pad = pad_sequences(test_tokens, maxlen=MAX_LEN)
#Zero is added before the values given in the padding operation.
print("x_train_pad.shape :",x_train_pad.shape)
print("test_pad.shape :",test_pad.shape)
print('='*92)
print("x_train_tokens :",x_train_tokens[0])
print("x_train_pad :",x_train_pad[0])


# In[70]:


num_words = len(list(tokenizer.word_index))

#임베딩 매트릭스(word2vec이랑 맞추기)
embedding_matrix = np.random.uniform(-1, 1, (VOCAB_SIZE, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
print(embedding_matrix.shape)
embedding_matrix[3]


# In[ ]:





# # 마지막 모델: GRU

# In[71]:


def build_gru(hp):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_LEN, trainable=False))
    model.add(GRU(units=hp.Int('units1', min_value=4, max_value=128,step=16)))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=256,step=32), activation='relu'))
    model.add(Dropout(rate = hp.Float('drop_out_rate', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-1, sampling='log')),
                  loss = 'binary_crossentropy',
                  metrics=['acc'])
    return model


# In[72]:


tuner = RandomSearch(
    build_gru,
    objective='val_acc',
    max_trials = 10,  #  total number of trials
    executions_per_trial=1, # the number of models that should be built and fit for each trial
    project_name ='gru1')


# In[73]:


tuner.search_space_summary()


# In[74]:


tuner.search(x_train_pad,final_target, epochs=15, validation_split=0.25, 
             callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3)]) 


# In[75]:


gru_best_model = tuner.get_best_models(num_models=1)[0]


# # THRESHOLD

# In[76]:


# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
model3_ = gru_best_model.predict(x_train_pad) # model2 결과

scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4,0.81,0.01):
    print(f'{threshold:.02f}, ',end='')
    preds = (model2_.reshape((-1))>threshold).astype('int')
    m = f1_score(target.to_numpy().reshape((-1)), preds, average='macro')   
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score:
        best_score = m
        best_threshold = threshold


# In[77]:


import matplotlib.pyplot as plt

# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()


# In[78]:


def model_3_to_binary(result):
    for i in range(len(result)):
        if result[i]>= 0.5: # threshold 조정이 필요하다면 여기서 조정하면 됩니다
            result[i] = 1
        else:
            result[i] = 0
    return result


# In[79]:


#합의 안된 데이터들 최종 예측
result2 = gru_best_model.predict(test_pad)
result2 = model_3_to_binary(result2)
y_final_result = pd.DataFrame(result2, index=fail,columns=['target'])
print(y_final_result['target'].value_counts())

#전체 타겟
y_total = pd.concat([final_target,y_final_result])
print(len(y_total))


# In[80]:


# 전체 데이터
total=pd.concat([final_train, final_test])
Fianl_data=pd.DataFrame({'review':total})
len(Fianl_data), Fianl_data


# In[81]:


Fianl_data['target']=y_total
Fianl_data.target.value_counts()


# In[82]:


# informative인 데이터 
X_informative = Fianl_data[Fianl_data['target']==1]  
X_informative.info()


# In[83]:


X_informative['target'].value_counts()


# In[84]:


X_informative[['review','target']].sample(20)


# In[ ]:





# In[ ]:




