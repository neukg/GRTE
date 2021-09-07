from transformers.modeling_bert import BertModel,BertPreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,  #B num_generated_triples H
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0] #hidden_states.shape
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) #B m H
        outputs = (layer_output,) + outputs
        return outputs



class GRTE(BertPreTrainedModel):
    def __init__(self, config):
        super(GRTE, self).__init__(config)
        self.bert=BertModel(config=config)
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.Lr_e1=nn.Linear(config.hidden_size,config.hidden_size)
        self.Lr_e2=nn.Linear(config.hidden_size,config.hidden_size)

        self.elu=nn.ELU()
        self.Cr = nn.Linear(config.hidden_size, config.num_p*config.num_label)

        self.Lr_e1_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size)
        self.Lr_e2_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size)

        self.rounds=config.rounds

        self.e_layer=DecoderLayer(config)

        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)

    def forward(self, token_ids, mask_token_ids):
        '''
        :param token_ids:
        :param token_type_ids:
        :param mask_token_ids:
        :param s_loc:
        :return: s_pred: [batch,seq,2]
        op_pred: [batch,seq,p,2]
        '''

        embed=self.get_embed(token_ids, mask_token_ids)
        #embed:BLH
        L=embed.shape[1]


        e1 = self.Lr_e1(embed) # BLL H
        e2 = self.Lr_e2(embed)

        for i in range(self.rounds):
            h = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL 2H
            B, L = h.shape[0], h.shape[1]

            table_logist = self.Cr(h)  # BLL RM

            if i!=self.rounds-1:

                table_e1 = table_logist.max(dim=2).values
                table_e2 = table_logist.max(dim=1).values
                e1_ = self.Lr_e1_rev(table_e1)
                e2_ = self.Lr_e2_rev(table_e2)

                e1=e1+self.e_layer(e1_,embed,mask_token_ids)[0]
                e2=e2+self.e_layer(e2_,embed,mask_token_ids)[0]

        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        embed=self.dropout(embed)
        return embed
