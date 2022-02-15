import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
  def __init__(self, outputDim, freeze_bert_pretrained = True):
    super(BertClassifier, self).__init__()

    D_in, H, D_out = 768, 256, outputDim

    # Bert layer
    self.bert = BertModel.from_pretrained('bert-base-uncased')

    # Linear layer with ReLU
    self.classifier = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out)
    )

    if freeze_bert_pretrained: # freeze these parameters so only the linear layers can be modified
      for param in self.bert.parameters():
        param.requires_grad = False
  
  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
    first_hidden_state_cls = outputs[0][:, 0, :]

    logits = self.classifier(first_hidden_state_cls)

    return logits