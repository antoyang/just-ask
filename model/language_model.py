import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel


class Bert(nn.Module):
    """ Finetuned DistilBERT module """

    def __init__(self):
        super(Bert, self).__init__()
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.cls_token = self.bert_tokenizer.cls_token_id
        self.sep_token = self.bert_tokenizer.sep_token_id

    def forward(self, tokens):
        attention_mask = (tokens > 0).float()
        embds = self.bert(tokens, attention_mask=attention_mask)[0]
        return embds


class Sentence_Maxpool(nn.Module):
    """ Utilitary for the answer module """

    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x_in):
        x = self.fc(x_in)
        x = torch.max(x, dim=1)[0]
        if self.relu:
            x = F.relu(x)
        return x


class AModel(nn.Module):
    """
    Answer embedding module
    """

    def __init__(self, out_dim=512, sentence_dim=2048):
        super(AModel, self).__init__()
        self.bert = Bert()
        self.linear_text = nn.Linear(768, out_dim)

    def forward(self, answer):
        if len(answer.shape) == 3:
            bs, nans, lans = answer.shape
            answer = answer.view(bs * nans, lans)
            answer = self.bert(answer)
            answer = answer[:, 0, :]
            answer = answer.view(bs, nans, 768)
        else:
            answer = self.bert(answer)
            answer = answer[:, 0, :]
        answer = self.linear_text(answer)
        return answer
