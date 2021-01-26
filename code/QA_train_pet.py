import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense

path = './QA_dataset/'

p = os.path.join(path, 'train', 'train.query.tsv')


def load_data(train_test='train',encode='utf-8'):
    D = {}
    with open(os.path.join(path, train_test, train_test + '.query.tsv'),encoding=encode) as f:
        for l in f:
            span = l.strip().split('\t')
            D[span[0]] = {'query': span[1], 'reply': []}

    with open(os.path.join(path, train_test, train_test + '.reply.tsv'),encoding=encode) as f:
        for l in f:
            span = l.strip().split('\t')
            if len(span) == 4:
                q_id, r_id, r, label = span
                label = int(label)
            else:
                label = None
                q_id, r_id, r = span
            D[q_id]['reply'].append([r_id, r, label])
    d = []
    for k, v in D.items():
        q_id = k
        q = v['query']
        reply = v['reply']

        for i, r in enumerate(reply):
            r_id, rc, label = r
            d.append([q_id, q, r_id, rc, label])
    return d
def load_train():
    d=[]
    df_train = pd.read_csv('./Q-A-matching-of-real-estate-industry-main/data/train_merge_1129.csv')
    for data_row in df_train.iloc[:].itertuples():
        d.append([1,data_row.query, 2,data_row.reply,data_row.label])
    return d




# train_data = load_data(train_test='train',encode="utf-8")
train_data=load_train()
test_data = load_data(train_test='test',encode="gbk")

num_classes = 32
maxlen = 128
batch_size = 8

# BERT base

# config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
# checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
# dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/vocab.txt'

config_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# pattern
pattern = '直接回答问题:'
mask_idx = [1]

id2label = {
    0: '间',
    1: '直'
}

label2id = {v: k for k, v in id2label.items()}
labels = list(id2label.values())


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    def __init__(self, prefix=False, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.prefix = prefix

    def __iter__(self, random=False):

        batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []

        for is_end, (q_id, q, r_id, r, label) in self.sample(random):
            label = int(label) if label is not None else None

            if label is not None or self.prefix:
                q = pattern + q

            token_ids, segment_ids = tokenizer.encode(q, r, maxlen=maxlen)

            if random:
                source_tokens, target_tokens = random_masking(token_ids)
            else:
                source_tokens, target_tokens = token_ids[:], token_ids[:]

            # mask label
            if label is not None:
                label_ids = tokenizer.encode(id2label[label])[0][1:-1]
                for m, lb in zip(mask_idx, label_ids):
                    source_tokens[m] = tokenizer._token_mask_id
                    target_tokens[m] = lb
            elif self.prefix:
                for i in mask_idx:
                    source_tokens[i] = tokenizer._token_mask_id

            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_target_ids = sequence_padding(batch_target_ids)

                yield [batch_token_ids, batch_segment_ids, batch_target_ids], None

                batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []


# shuffle
np.random.shuffle(train_data)
n = int(len(train_data) * 0.8)

train_generator = data_generator(data=train_data[: n]+test_data[:20000], batch_size=batch_size)
valid_generator = data_generator(data=train_data[n:], batch_size=batch_size)
test_generator = data_generator(data=test_data, batch_size=batch_size, prefix=True)
print(test_data[:10])

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        # self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                with_mlm=True # 加载bert/Roberta/ernie
#                                 model='nezha'
                                )

target_in = keras.layers.Input(shape=(None,))
print('target in ',K.int_shape(target_in))
print('bert out ',K.int_shape(model.output))
output = CrossEntropy(1)([target_in, model.output])
print('model out',output)


train_model = keras.models.Model(model.inputs + [target_in], output)

# AdamW = extend_with_weight_decay(Adam)
# AdamWG = extend_with_gradient_accumulation(AdamW)
#
# opt = AdamWG(learning_rate=1e-5, exclude_from_weight_decay=['Norm', 'bias'], grad_accum_steps=4)
opt=Adam(5e-6)
train_model.compile(optimizer=opt)
train_model.summary()

label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])


def predict(x):
    if len(x) == 3:
        x = x[:2]
    # print('model out',model.predict(x))
    # print('shape',K.int_shape(model.predict(x)))
    y_pred = model.predict(x)[:, mask_idx]
    # print(f'logits1:{K.int_shape(y_pred)}')
    # print(f'logits1:{y_pred}')
    y_pred = y_pred[:, 0, label_ids[:, 0]]
    # print(f'logits2:{K.int_shape(y_pred)}')
    # print(f'logits2:{y_pred}')
    y_pred = y_pred.argmax(axis=1)
    return y_pred


def evaluate(data):
    P, R, TP = 0., 0., 0.
    for d, _ in tqdm(data):
        x_true, y_true = d[:2], d[2]

        y_pred = predict(x_true)
        y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idx]])
        #         print(y_true, y_pred)
        R += y_pred.sum()
        P += y_true.sum()
        TP += ((y_pred + y_true) > 1).sum()

    print(P, R, TP)
    pre = TP / R
    rec = TP / P

    return 2 * (pre * rec) / (pre + rec)


# def evaluate(data):
#     label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])
#     total, right = 0., 0.
#     for x_true, _ in data:
#         x_true, y_true = x_true[:2], x_true[2]
#         # print(x_true)
#         # a=input()
#         y_pred = model.predict(x_true)[:, mask_idx]
#
#         y_pred = y_pred[:, 0, label_ids[:, 0]]
#         y_pred = y_pred.argmax(axis=1)
#         y_true = np.array([
#             labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idx]
#         ])
#         total += len(y_true)
#         right += (y_true == y_pred).sum()
#     return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(valid_generator)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights('best_pet_model_4.weights')
        print('acc :{}, best acc:{}'.format(acc, self.best_acc))


def write_to_file(path):
    preds = []
    for x, _ in tqdm(test_generator):
        # print(x[0])
        # a=input()
        pred = predict(x)
        preds.extend(pred)

    ret = []
    for data, p in zip(test_data, preds):
        ret.append([data[0], data[2], str(p)])

    with open(path, 'w') as f:
        for r in ret:
            f.write('\t'.join(r) + '\n')


if __name__ == '__main__':
    print(evaluate(valid_generator))
    evaluator = Evaluator()
    train_model.fit_generator(train_generator.forfit(),
                              steps_per_epoch=len(train_generator),
                              epochs=10,
                              callbacks=[evaluator])

    train_model.load_weights('best_pet_model_4.weights')
    # print(evaluate(test_generator))
    # print(evaluate(valid_generator))
    write_to_file('result_pet_1205.tsv')