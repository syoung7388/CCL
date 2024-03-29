from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch 


class TLM(nn.Module):
    def __init__(self, args):
        super(TLM, self).__init__()
        print('Target Language Model or Reference Language Model')
        self.args = args 
        self.config = AutoConfig.from_pretrained(args.model_name)
        self.config.hidden_dropout_prob = args.dropout
        self.config.attention_probs_dropout_prob = args.dropout
        self.bert = AutoModel.from_pretrained(args.model_name, config=self.config)
        self.bert.config.type_vocab_size = 2
        self.bert.embeddings.token_type_embeddings = nn.Embedding(2, self.bert.config.hidden_size)
        self.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size) 
        

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, output_hidden_states = True)
        bert_output = bert_output.last_hidden_state
        return bert_output


class ILM(nn.Module):
    def __init__(self, args):
        super(ILM, self).__init__()
        print('Inference Language Model')
        self.args = args 
        self.config = AutoConfig.from_pretrained(args.model_name)
        self.config.hidden_dropout_prob = args.dropout
        self.config.attention_probs_dropout_prob = args.dropout
        self.bert = AutoModel.from_pretrained(args.model_name, config=self.config)
        self.bert.config.type_vocab_size = 2
        self.bert.embeddings.token_type_embeddings = nn.Embedding(2, self.bert.config.hidden_size)
        self.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size) #X
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, output_hidden_states = True)
        bert_output = bert_output.last_hidden_state
        return bert_output

def get_linear_projection(args, classnum):
    input_size = args.hidden_size*args.max_length
    return nn.Sequential(
        nn.Linear(input_size, args.hidden_size),
        nn.Linear(args.hidden_size, classnum)
    )
    
class TargetNet(nn.Module):
    def __init__(self, args,  classnum):
        super(TargetNet, self).__init__()
        print('TargetNet or ReferenceNet')
        self.args = args 
        self.classnum = classnum
        self.tlm = TLM(self.args) 
        if self.args.model_load:
            self.tlm.load_state_dict(torch.load('./results/contrastive'+self.args.tlm_path))
        self.classifier_linear = get_linear_projection(args, self.classnum)

    def forward(self, input_ids, token_type_ids, attention_mask, label):
        bert_output = self.tlm(input_ids, token_type_ids, attention_mask)
        bert_output = bert_output.view(len(input_ids), -1)
        last_output = self.classifier_linear(bert_output)
        return {
            'last_output': last_output
        }


class InferNet(nn.Module):
    def __init__(self, args,  classnum):
        super(InferNet, self).__init__()
        print('InferNet')
        self.args = args 
        self.classnum = classnum
        self.ilm = ILM(self.args) 
        if self.args.model_load:
            self.ilm.load_state_dict(torch.load('./results/contrastive'+self.args.ilm_path))
        self.classifier_linear = get_linear_projection(args, self.classnum)

    def forward(self, input_ids, token_type_ids, attention_mask, label):
        bert_output = self.ilm(input_ids, token_type_ids, attention_mask)
        bert_output = bert_output.view(len(input_ids), -1)
        last_output = self.classifier_linear(bert_output)
        return {
            'last_output': last_output
        }