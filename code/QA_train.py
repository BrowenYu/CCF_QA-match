import re,os
import codecs
import gc
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["RECOMPUTE"]="1"
import tensorflow as tf
import keras
import sys
import bert4keras
import numpy as np
import random as rd
import pandas as pd
import chardet
import pickle
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold

from keras.layers import *
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import *
# from keras import backend as K
from keras.legacy import interfaces
from tqdm import tqdm

from random import choice
from keras.callbacks import Callback
from keras.utils import to_categorical
import tensorflow

#数据读取及处理
train_left = pd.read_csv('./QA_dataset/train/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
# print(repr(train_left.q1.values[985]))
# a='还能谈😂lebron \n'
# a=re.sub('\s','',a)



train_right = pd.read_csv('./QA_dataset/train/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']

train_right['q2'] = train_right['q2'].fillna('你好')




df_train = train_left.merge(train_right, how='left')
df_train['q2'] = df_train['q2'].fillna('你好')


df_train=pd.read_csv('./Q-A-matching-of-real-estate-industry-main/data/train_merge_1129.csv')
# df_train=pickle.load(open('./QA_dataset/train/train.pkl', 'rb'))



# df_train['q1']=df_train['q1'].apply(lambda x:re.sub('\s','',x))
# df_train['q2']=df_train['q2'].apply(lambda x:re.sub('\s','',x))

# content_length = [len(i)+len(j)for i,j in zip(df_train.q1.values,df_train.q2.values)]
# df_train['length']=content_length
# print(np.percentile(content_length,95))
# content_length = pd.Series(content_length)
# print(df_train.length.value_counts())
# print(df_train.label.value_counts())
# print(max(content_length))
# print(df_train.describe())
# import matplotlib.pyplot as plt
# import matplotlib
#
# # 设置matplotlib正常显示中文和负号
# matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# plt.figure(figsize=(10,5))
# plt.hist(content_length, bins=20, range=(0,265),facecolor="blue", edgecolor="black", alpha=0.7)
# # 显示横轴标签
# plt.xlabel("长度区间")
# # 显示纵轴标签
# plt.ylabel("样本数")
# # 显示图标题
# plt.title("文本长度分布直方图")
# plt.show()


test_left = pd.read_csv('./QA_dataset/test/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']




test_right =  pd.read_csv('./QA_dataset/test/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']




df_test = test_left.merge(test_right, how='left')

print(df_test.info())
print(df_train.info())

# df_test['q1']=df_test['q1'].apply(lambda x:re.sub('\s','',x))
# df_test['q2']=df_test['q2'].apply(lambda x:re.sub('\s','',x))

# print(df_test.info())
# a=[len(i) for i in list(test_left['q1'].values)]
# b=[len(i) for i in list(test_right['q2'].values)]
#
# print(max(a),min(a),np.mean(a))
# print(max(b),min(b),np.mean(b))

class AdamWarmup(keras.optimizers.Optimizer):
    def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                 lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, kernel_weight_decay=0., bias_weight_decay=0.,
                 amsgrad=False, **kwargs):
        super(AdamWarmup, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.decay_steps = K.variable(decay_steps, name='decay_steps')
            self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
            self.min_lr = K.variable(min_lr, name='min_lr')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.kernel_weight_decay = K.variable(kernel_weight_decay, name='kernel_weight_decay')
            self.bias_weight_decay = K.variable(bias_weight_decay, name='bias_weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_kernel_weight_decay = kernel_weight_decay
        self.initial_bias_weight_decay = bias_weight_decay
        self.amsgrad = amsgrad

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        lr = K.switch(
            t <= self.warmup_steps,
            self.lr * (t / self.warmup_steps),
            self.lr * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
        )

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = m_t / (K.sqrt(v_t) + self.epsilon)

            if 'bias' in p.name or 'Norm' in p.name:
                if self.initial_bias_weight_decay > 0.0:
                    p_t += self.bias_weight_decay * p
            else:
                if self.initial_kernel_weight_decay > 0.0:
                    p_t += self.kernel_weight_decay * p
            p_t = p - lr_t * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'decay_steps': float(K.get_value(self.decay_steps)),
            'warmup_steps': float(K.get_value(self.warmup_steps)),
            'min_lr': float(K.get_value(self.min_lr)),
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'kernel_weight_decay': float(K.get_value(self.kernel_weight_decay)),
            'bias_weight_decay': float(K.get_value(self.bias_weight_decay)),
            'amsgrad': self.amsgrad,
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
maxlen = 100#seq最大长度
learning_rate = 5e-5#学习率
min_learning_rate = 3e-5

weight_decay = 0.001
nb_epochs=1

input_categories = ['q1','q2']
output_categories = 'label'

# config_path = './publish/bert_config.json'
# checkpoint_path = './publish/bert_model.ckpt'
# dict_path = './publish/vocab.txt'


config_path = './publish/bert_config.json'
checkpoint_path = './publish/bert_model.ckpt'
dict_path = './publish/vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
#重写keras_bert的token模块，将空格转换为[unused1]
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

# print(tokenizer.tokenize(a))
# print(to_categorical(1,2))
# print(to_categorical(0,2))

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    # ML = maxlen
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=16,shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            # np.random.shuffle(list(idxs))
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen] #seq1
                text2 = d[1][:maxlen]#seq2
                x1, x2 = tokenizer.encode(text,second_text=text2)#对seq编码
                y = d[2]#label
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []



#划分数据集，训练：验证=9：1
# df_train=df_train[['q1','q2','label']]
# df_test_copy=df_test[['q1','q2']]

# random_order = range(len(data))
# np.random.shuffle(list(random_order))
# train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
# valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]








def acc_top2(y_true, y_pred):
 return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=4)


def bertmodel(decay_steps,warmup_steps):
    model = build_transformer_model(
        config_path,
        checkpoint_path,
    )
    for l in model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    output = model([x1_in, x2_in])
    # output=model.get_layer('Transformer-%s-FeedForward-Norm' % 11).output
    output = Lambda(lambda x: x, output_shape=lambda s: s)(output)
    print(K.int_shape(output))
    avgpool_out=GlobalAveragePooling1D()(output)
    maxpool_out=GlobalMaxPooling1D()(output)
    t_out=Lambda(lambda x: x[:, -1])(output)
    e_out=Lambda(lambda x: x[:, 0])(output)
    output=Concatenate()([avgpool_out, maxpool_out, t_out, e_out])


    # bert_hidden_layers = [model.get_layer('Transformer-%s-FeedForward-Norm' % i).output for i in
    #                       range(23,6, -1)]
    # weight_b = Lambda(lambda x: x * (1 / 5))
    # output = [weight_b(item) for item in bert_hidden_layers]
    #
    # output = keras.layers.add(output)

    # bert_hidden_layers = [model.get_layer('Transformer-%s-FeedForward-Norm' % i).output for i in
    #                       range(11, -1, -1)]
    #
    # bert_hidden_layers = Lambda(lambda z: tf.convert_to_tensor(z))(bert_hidden_layers)
    #
    #
    # attention_vector = Dense(1, name='attention_vec',
    #                          kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None))(
    #     bert_hidden_layers)
    # attention_probs = Softmax()(attention_vector)
    # attention_mul = Lambda(lambda x: x[0] * x[1])([attention_probs, bert_hidden_layers])
    # # output=keras.layers.multiply([bert_hidden_layers,attention_probs])
    # output = Lambda(lambda z: tf.reduce_sum(z, axis=0, keepdims=False))(attention_mul)

    output = Dropout(0.5)(output)
    LSTM()
    # output = Lambda(lambda x: x[:, 0])(output)  # 模型训练结果的第一位 [cls] 进行预测
    output = Dense(2, activation='softmax',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(output)  # 加softmax线性层
    # [x1_in, x2_in]
    model = Model([x1_in, x2_in], output)
#     model.summary()
#     AdamWarmup(lr=min_learning_rate, decay_steps=decay_steps, warmup_steps=warmup_steps,
#                kernel_weight_decay=weight_decay)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(min_learning_rate),
        metrics=['accuracy',acc_top2]

    )
    print(model.summary())
    return model



DATA_LIST = []
for data_row in df_train.iloc[:].itertuples():
    DATA_LIST.append((data_row.query,data_row.reply,to_categorical(data_row.label, 2)))
DATA_LIST = np.array(DATA_LIST)

DATA_LIST_TEST = []
for data_row in df_test.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.q1,data_row.q2,to_categorical(0, 2)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)


def get_data(df):
    DATA_LIST = []
    for data_row in df.iloc[:].itertuples():
        DATA_LIST.append((data_row.q1, data_row.q2, to_categorical(data_row.label, 2)))
    DATA_LIST = np.array(DATA_LIST)
    return DATA_LIST

def run_cv(nfold, data,data_labels, data_test):
    id_len=6000
    # kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(range(id_len))
    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
    # kf=GroupKFold(n_splits=nfold,).split(X=data, groups=data[:,3])
    train_model_pred = np.zeros((len(data), 2))
    test_model_pred = np.zeros((len(data_test), 2))


    for i, (train_fold, test_fold) in enumerate(kf):

        X_train, X_valid, = data[train_fold, :], data[test_fold, :]
        # X_train, X_valid, = get_data(data[data.id.isin(train_fold)]), get_data(data[data.id.isin(test_fold)])
        decay_steps = int(nb_epochs * len(X_train) / 16)
        warmup_steps = int(0.1 * decay_steps)

        model = bertmodel(decay_steps,warmup_steps)
        early_stopping = EarlyStopping(monitor='val_acc', patience=4)  # 早停法，防止过拟合
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=3)  # 当评价指标不在提升时，减少学习率
        checkpoint = ModelCheckpoint('./QA_model/'+str(i) + '.hdf5', monitor='val_acc', verbose=2, save_best_only=True, mode='max',
                                     save_weights_only=True)  # 保存最好的模型

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)
        test_D = data_generator(data_test,shuffle=False)
        # 模型训练
        print(f'第{i+1}折')
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=10,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],

        )

        train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)

        del model
        gc.collect()  # 清理内存
        K.clear_session()  # clear_session就是清除一个session
    return train_model_pred, test_model_pred



train_model_pred, test_model_pred = run_cv(5, DATA_LIST, None, DATA_LIST_TEST)
test_pred = [np.argmax(x) for x in test_model_pred]
test_model_pred=test_model_pred/5




# train_D = data_generator(train_data)
# valid_D = data_generator(valid_data)
#
# model.fit_generator(
#     train_D.__iter__(),
#     steps_per_epoch=len(train_D),
#     epochs=5,
#     validation_data=valid_D.__iter__(),
#     validation_steps=len(valid_D)
# )
# testdata=df_test[['q1','q2']].to_numpy()
# def makeresult(testdata):
#     result=[]
#     for test in testdata:
#         _t1, _t2 = tokenizer.encode(first=test[0],second=test[1])
#         _t1, _t2 = np.array([_t1]), np.array([_t2])
#         label = model.predict([_t1, _t2])
#         result.append([label])
#     return result
# result=makeresult(testdata)




df_test['label']=test_pred
df_test=df_test[['id','id_sub','label']]
# df_test.to_csv("result.csv",index=0)
df_test.to_csv("result_1203.tsv",sep='\t',header=None,index=0)
pickle.dump(train_model_pred,open('./Q-A-matching-of-real-estate-industry-main/data/train_1.pkl','wb'))
pickle.dump(test_model_pred,open('./Q-A-matching-of-real-estate-industry-main/data/test_1.pkl','wb'))







# result = pd.read_csv('./result.csv')
# result['newlabel']=result['label'].apply(lambda x:re.findall(u'.*\\[\\[(.*)\\]\\].*', x))
# result['newlabel']=result['newlabel'].apply(lambda x:x[0])
# result['newlabel']=result['newlabel'].apply(lambda x:1 if float(x)>=0.5 else 0)
# result=result[['id','id_sub','newlabel']]
# # print(result['newlabel'])
# result.to_csv("newresult.tsv",sep='\t',header=None,index=0)