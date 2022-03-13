import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
  def __init__(self, outputDim, freeze_bert_pretrained = True):
    super(BertClassifier, self).__init__()

    dim_in, H, dim_out = 768, 256, outputDim

    self.bert = BertModel.from_pretrained('bert-base-uncased')

    self.classifier = nn.Sequential(
        nn.Linear(dim_in, H),
        nn.ReLU(),
        nn.Linear(H, dim_out)
    )


    if freeze_bert_pretrained: # only alter finetuning layers!
      for param in self.bert.parameters():
        param.requires_grad = False
  
  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
    output = outputs[0][:, 0, :]

    logits = self.classifier(output)
    #sigmoid = nn.Sigmoid()
    #logits = sigmoid(logits)
    return logits
