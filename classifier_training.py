import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import random
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
class pytorch_train():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_dict = {}
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.batch_size = 10
        self.epoch = 5
        self.lr = 1e-5
        self.eps = 1e-8
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)


    def _load_data(self):
        path = r'Data/market_sentiment.csv'
        df = pd.read_csv(path)
        df.set_index('id', inplace=True)
        for idx, label in enumerate(df.sentiment.unique()):
            self.label_dict[label] = idx
        df['sentiment'] = df.sentiment.replace(self.label_dict)
        return df

    def _prepare_data(self):
        df = self._load_data()
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.sentiment.values, test_size=0.15,
                                                          random_state=17, stratify=df.sentiment.values)
        df['data_type'] = ['not_set'] * df.shape[0]
        df.loc[X_train, 'data_type'] = 'train'
        df.loc[X_val, 'data_type'] = 'val'
        return df

    def create_torch_dataset(self):
        df = self._prepare_data()
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
        encoded_data_train = tokenizer.batch_encode_plus(
            df[df.data_type == 'train'].title.values,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt'
        )

        encoded_data_val = tokenizer.batch_encode_plus(
            df[df.data_type == 'val'].title.values,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(df[df.data_type == 'train'].sentiment.values)

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(df[df.data_type == 'val'].sentiment.values)

        dataset_train = TensorDataset(input_ids_train,
                                      attention_masks_train, labels_train)
        dataset_val = TensorDataset(input_ids_val,
                                    attention_masks_val,
                                    labels_val)

        dataloader_train = DataLoader(
            dataset_train,
            sampler=RandomSampler(dataset_train),
            batch_size=self.batch_size
        )

        dataloader_val = DataLoader(
            dataset_val,
            sampler=RandomSampler(dataset_val),
            batch_size= self.batch_size
        )

        return dataloader_train, dataloader_val

    def load_pretrained_model(self, model_name):
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_dict),
            output_attentions=False,
            output_hidden_states=False
        )

    def set_spec(self, size):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-5,
            eps=1e-8
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=size * self.epoch
            # num_training_steps=len(dataloader_train) * epochs
        )

    def f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(self, preds, labels):
        label_dict_inverse = {v: k for k, v in self.label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        # print(preds_flat)
        # print(labels)
        labels_flat = labels.flatten()
        print(preds_flat)
        print(labels_flat)
        for label in np.unique(labels_flat):
            y_pred = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_pred[y_pred == label])}/{len(y_true)}\n')



    def train_model(self, dataloader_train, dataloader_val):
        for epoch in tqdm(range(1, self.epoch + 1)):
            # print(f"training on epoch {epoch}")
            self.model.train()
            loss_train_total = 0
            progress_bar = tqdm(dataloader_train,
                                desc="Epoch {:1d}".format(epoch),
                                leave=False,
                                disable=False)
            for batch in progress_bar:
                # print(len(batch))
                self.model.zero_grad()
                batch = tuple(b.to(self.device) for b in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }

                output = self.model(**inputs)
                loss = output[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
            torch.save(self.model.state_dict(), f'models/BERT_ft_epoch{epoch}.pt')

            tqdm.write('\nEpoch {epoch}')
            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = self.evaluate(dataloader_val)
            val_f1 = self.f1_score_func(predictions, true_vals)
            tqdm.write(f'validation loss: {val_loss}')
            tqdm.write(f'F1 score (weighted): {val_f1}')


    def _evaluate(self, dataloader_val):
        self.model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in tqdm(dataloader_val):
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals


    def eval_sentiment_analyzer(self, dataloader_val, model_name):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=len(self.label_dict),
                                                              output_attentions=False,
                                                              output_hidden_states=False
                                                              )
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(f'models/{model_name}',  map_location=self.device))
        _, predictions, true_vals = self._evaluate(dataloader_val)
        self.accuracy_per_class(predictions, true_vals)

if __name__ == '__main__':

    pytorch_obj = pytorch_train()

    dataloader_train, dataloader_val = pytorch_obj.create_torch_dataset()

    pytorch_obj.load_pretrained_model('bert-base-uncased')

    pytorch_obj.set_spec(len(dataloader_train))

    pytorch_obj.train_model(dataloader_train, dataloader_val)

    #uncomment the 2 lines below after training the model to evaluate the performance
    # model_name = "BERT_ft_epoch5.model"
    # pytorch_obj.eval_sentiment_analyzer(dataloader_val, model_name)