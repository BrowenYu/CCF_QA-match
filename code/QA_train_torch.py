import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc, os
import time
import datetime
import random
from tqdm.auto import tqdm
from torch.utils import data
from torch.nn import CrossEntropyLoss
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer,BertConfig,BertModel,BertForSequenceClassification,BertPreTrainedModel,AdamW,get_linear_schedule_with_warmup
from sklearn.model_selection import KFold





# loss_weight=[max(loss_weight)/(i*2) for i in loss_weight]
# print(loss_weight)
num_labels=2
maxlen=256
batch_size = 16



#数据读取及处理
train_left = pd.read_csv('./QA_dataset/train/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']

train_right = pd.read_csv('./QA_dataset/train/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']


df_train = train_left.merge(train_right, how='left')
df_train['q2'] = df_train['q2'].fillna('你好')



test_left = pd.read_csv('./QA_dataset/test/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']

test_right =  pd.read_csv('./QA_dataset/test/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']

df_test = test_left.merge(test_right, how='left')

print(df_test.info())
print(df_train.info())


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()


    def forward(self, inputs, target):
        N = target.size(0)
        smooth = 1
        # print(inputs.size())
        # print('emission:',inputs)
        # inputs=torch.argmax(inputs,dim=1)
        input_flat = inputs.view(N, -1)
        target_flat = target.view(N, -1)
        # input_flat=input_flat.squeeze()
        # target_flat=target_flat.squeeze()
        # input_flat=input_flat.type(torch.FloatTensor)
        target_flat=target_flat.type(torch.cuda.FloatTensor)
        input_flat=input_flat.softmax(dim=1)
        input_flat=input_flat[:,1]
        target_flat=target_flat[:,0]

        intersection = input_flat * target_flat
        loss = (2 * intersection.sum(0) + smooth) / ((input_flat*input_flat).sum(0) + (target_flat*target_flat).sum(0) + smooth)
        loss = 1 - loss.sum()
        # print('dice loss:',loss)
        return loss


class Bert_QA(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_QA, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.dice_loss=DiceLoss()
        self.init_weights()
        # self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks,seq_ids,labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attn_masks,seq_ids)
        sequence_output = outputs[1]
        # sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        # attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            # print('predict softmax:',F.softmax(emission, 2).size())
            # print('labels:',labels.size())
            # print('attention_mask',attn_masks.size())
            loss_fct = CrossEntropyLoss()
            # loss =loss_fct(emission.view(-1, self.num_labels), labels.view(-1))
            loss=self.dice_loss(emission,labels.view(-1))
            # +self.dice_loss(emission, labels.view(-1))
            return loss
        else:
            return F.softmax(emission,dim=1)


def load_data(df):
    DATA_LIST = []
    for data_row in df.iloc[:].itertuples():
        DATA_LIST.append((data_row.q1, data_row.q2, int(data_row.label)))
    DATA_LIST = np.array(DATA_LIST)
    return DATA_LIST


class QA_Dataset(data.Dataset):
    def __init__(self,sentences):
        self.sentences = sentences
        self.tokenizer = BertTokenizer.from_pretrained("./chinese_roberta_wwm_ext_pytorch/vocab.txt")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        bert_tokens = [102]
        q1,q2,label=sentence[0],sentence[1],[np.int(sentence[2])]
        # print(q1,q2,label)
        # print(type(q1),type(q2),type(label))
        q1_ids=self.tokenizer.encode(q1)[1:-1]
        q2_ids = self.tokenizer.encode(q2)[1:-1]
        if len(q1_ids)+len(q2_ids)<=253:
            bert_tokens=bert_tokens+q1_ids+[103]+q2_ids+[103]

        bert_tokens = bert_tokens + [0] * (256 - len(bert_tokens))

        token_type_ids=[0]*(len(q1_ids)+1)+[1]*(255-len(q1_ids))
        atten_mask = [float(i > 0) for i in bert_tokens]
        # print(bert_tokens)
        # print(atten_mask)
        # print(token_type_ids)
        # print(type(label))


        return torch.LongTensor(bert_tokens),torch.LongTensor(atten_mask),torch.LongTensor(token_type_ids),torch.LongTensor(label)







tokenizer = BertTokenizer.from_pretrained("./chinese_roberta_wwm_ext_pytorch/vocab.txt")




# def predict(text,model):
#
#
#
#
#
#
#
# def evaluate(data,model):
#     """评测函数
#     """
#     X, Y, Z = 1e-10, 1e-10, 1e-10
#     for d in tqdm(data):
#         text = ''.join([i[0] for i in d])
#         R = set(predict(text,model))  # 预测
#         T = set([tuple(i) for i in d if i[1] != 'O'])  # 真实
#         X += len(R & T)
#         Y += len(R)
#         Z += len(T)
#     precision, recall_1 = X / Y, X / Z
#     f1 = 2 * precision * recall_1 / (precision + recall_1)
#     return f1, precision, recall_1


def get_model(total_steps):
    modelConfig = BertConfig.from_pretrained('./chinese_roberta_wwm_ext_pytorch/bert_config.json', num_labels=num_labels)

    model = Bert_QA.from_pretrained('./chinese_roberta_wwm_ext_pytorch/pytorch_model.bin', config=modelConfig)

    model.cuda('cuda:2')
    print(model)
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    return model,optimizer,scheduler

def get_data(train,valid):
    train_dataset = QA_Dataset(train)
    val_dataset = QA_Dataset(valid)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4)
    validation_dataloader = data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=1)
    return train_dataloader,validation_dataloader



# def val(i,model):
#     import glob
#     import codecs
#     X, Y, Z = 1e-10, 1e-10, 1e-10
#     val_data_flist = glob.glob(f'./round2_train/val_data_{i}/*.txt')
#     data_dir = f'./round2_train/val_data_{i}/'
#     # num=0
#     for file in val_data_flist:
#         if file.find(".ann") == -1 and file.find(".txt") == -1:
#             continue
#         file_name = file.split('/')[-1].split('.')[0]
#         r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
#         r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
#
#         R = []
#         with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
#
#             line = f.readlines()
#             # if num==0:
#             #     print(line)
#             #     num+=1
#             aa = test_predict(line,model)
#             for line in aa[0]:
#                 lines = line['label_type'] + " " + str(line['start_pos']) + ' ' + str(line['end_pos']) + "\t" + line[
#                     'res']
#                 R.append(lines)
#         T = []
#         with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 lines = line.strip('\n').split('\t')[1] + '\t' + line.strip('\n').split('\t')[2]
#                 T.append(lines)
#         R = set(R)
#         T = set(T)
#         X += len(R & T)
#         Y += len(R)
#         Z += len(T)
#     precision, recall = X / Y, X / Z
#     f1 = 2 * precision * recall / (precision + recall)
#     print('*' * 100)
#     print('f1={},  precision={},  recall={}'.format(f1, precision, recall))
#     print('*' * 100)
#
#     return f1


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)







epochs = 10

# Total number of training steps is number of batches * number of epochs.


# Create the learning rate scheduler.



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda:2")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Set the seed value all over the place to make this reproducible.



best_model=None
model_list=[]


DATA_LIST_TEST = []
for data_row in df_test.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.q1,data_row.q2))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)



def predict(test_res,model):
    for index,item in tqdm(enumerate(DATA_LIST_TEST)):
        q1,q2=item[0],item[1]
        q1_ids=tokenizer.encode(q1)
        q2_ids =tokenizer.encode(q2)[1:]
        bert_tokens=q1_ids+q2_ids
        token_type_ids=[0]*(len(q1_ids))+[1]*(len(q2_ids))
        atten_mask = [1]*len(bert_tokens)
        bert_tokens=torch.LongTensor([bert_tokens]).cuda('cuda:2')
        token_type_ids=torch.LongTensor([token_type_ids]).cuda('cuda:2')
        atten_mask=torch.LongTensor([atten_mask]).cuda('cuda:2')
        with torch.no_grad():
            logits=model(bert_tokens,atten_mask,token_type_ids)
        logits=logits.detach().cpu().numpy()
        test_res[index]+=logits[0]
    return test_res

# Store the average loss after each epoch so we can plot them.

# For each epoch...
id_len = 6000
nfold=6
kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(range(id_len))
test_model_pred=np.zeros((len(DATA_LIST_TEST), 2))
for i, (train_fold, test_fold) in enumerate(kf):
    print(f'第{i+1}折')
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    X_train, X_valid, = load_data(df_train[df_train.id.isin(train_fold)]), load_data(df_train[df_train.id.isin(test_fold)])
    train_dataloader, validation_dataloader=get_data(X_train,X_valid)

    total_steps = len(train_dataloader) * epochs

    model, optimizer, scheduler=get_model(total_steps)

    loss_values = []
    best_f1 = 0
    val_f1=0
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 20 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_seq_ids = batch[2].to(device)
            b_labels = batch[3].to(device)
            model.zero_grad()

            outputs = model(b_input_ids,b_input_mask,b_seq_ids,b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs
            if step % 20 == 0 and not step == 0:
                print(f'training loss of every 20 batch:{loss.item()}')

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))



        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_seq_ids = batch[2].to(device)
            b_labels = batch[3].to(device)


            with torch.no_grad():
                outputs = model(b_input_ids,b_input_mask,b_seq_ids)


            logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            b_labels = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, b_labels)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print('starting predicting')
    test_model_pred=predict(test_model_pred,model)
    torch.cuda.empty_cache()


test_pred = [np.argmax(x) for x in test_model_pred]

df_test['label']=test_pred
df_test=df_test[['id','id_sub','label']]
# df_test.to_csv("result.csv",index=0)
df_test.to_csv("result_1119.tsv",sep='\t',header=None,index=0)






