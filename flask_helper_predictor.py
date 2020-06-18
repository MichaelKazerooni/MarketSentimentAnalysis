from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

class sentiment_analysis():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.__load_pretrained_model()


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __load_pretrained_model(self):
      model = BertForSequenceClassification.from_pretrained(
          'bert-base-uncased',
          num_labels = 3,
          output_attentions = False,
          output_hidden_states = False
      )

      model.to(self.device)
      # model.eval()
      with torch.no_grad():
          model.load_state_dict(torch.load('models/BERT_ft_epoch5.model', map_location=self.device))
          print('Model loaded correctly')
      return model

    def _prepare_text(self, text):
      tokenizer = BertTokenizer.from_pretrained(
          'bert-base-uncased',
          do_lower_case = True
      )
      encoded_data_val = tokenizer.encode_plus(
          text,
          add_special_tokens = True,
          return_attention_mask = True,
          pad_to_max_length = True,
          max_length = 256,
          return_tensors = 'pt'
      )
      return encoded_data_val


    def predict(self, text):
        dataloader_val = self._prepare_text(text)
        outputs = None
        self.model.eval()
        predictions= []
        inputs = {'input_ids':      dataloader_val['input_ids'].to(self.device),
                  'attention_mask': dataloader_val['attention_mask'].to(self.device),}
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = outputs[0].data.cpu().numpy()
        preds = np.argmax(preds, axis = 1)
        if preds == 0:
            return 'neutral'
        elif preds == 2:
            return 'positive'
        elif preds == 1:
            return 'negative'


        # text = "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility  contrary to earlier layoffs the company contracted the ranks of its office workers  the daily Postimees reported "
        # encoded_data_val = prepare_text(text)
        # model = load_pretrained_model()
        # print(f'sentiment is:  {evaluate_single_text(encoded_data_val, model)}')