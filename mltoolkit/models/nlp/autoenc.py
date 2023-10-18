# external imports
import torch
import numpy as np
from torch import nn
from torch.nn import functional as f
from typing import List
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# local imports
from mltoolkit.nn_modules import PositionalEncoding
from mltoolkit.utils import display

class TransformerAutoencoder(nn.Module):
    
    def __init__(self, cfg):

        super(TransformerAutoencoder, self).__init__()

        ################ set defaults from cfg ################

        # temporary vars
        self.dev = cfg.model.get('device', 'cpu')
        V = cfg.model['vocab_size']
        embedding_dim = cfg.model.get('embedding_dim', 512)
        num_decoder_layers = cfg.model.get('num_decoder_layers', 6)
        nhead = cfg.model.get('nhead', 8)
        dim_feed_forward = cfg.model.get('dim_feed_forward', 512)
        decoder_dropout = cfg.model.get('decoder_dropout', .1)
        mlp_dropout = cfg.model.get('mlp_dropout', .1)
        mlp_layers = cfg.model.get('mlp_layers', 3)
        pad_id = cfg.model.get('pad_token_id')
        encoder_name = cfg.model.get('encoder', 'all-mpnet-base-v1')
        enc_batch_size = cfg.model.get('encoder_batch_size', 100)
        up_conv_seq = cfg.model.get('up_conv_sequece', [16, 64, 128])
        up_conv_kernels = cfg.model.get('up_conv_kernel_sizes', [5, 9, 33])

        # persistent vars
        self.enc_batch_size = enc_batch_size
        self.max_seq_len = cfg.data['max_seq_len']
        self.devs = cfg.model['devices']
        self.embs_on_cpu = cfg.model.get('embeddings_on_cpu', True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.data['tokenizer_name'])

        #######################################################

        # initialize encoder
        self.encoder = SentenceTransformer(encoder_name)
        enc_dims = self.encoder.encode('').shape[0]

        # freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grade = False

        #initialize embeddings and positional encodings
        self.embeddings = nn.Embedding(
            V,
            embedding_dim,
            padding_idx = pad_id
        )

        self.pos = PositionalEncoding(
            embedding_dim,
        )

        if not self.embs_on_cpu:
            self.embeddings = self.embeddings.to(self.devs[0])
            self.pos = self.pos.to(self.devs[0])

        # initialize up-conv layers
        up_conv = []
        up_conv_seq = [1] + up_conv_seq
        for in_channels, out_channels, kernel in zip(
            up_conv_seq[:-1],
            up_conv_seq[1:],
            up_conv_kernels
        ):
            up_conv.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride=1,
                        padding=(kernel-1)//2,
                    ),
                    nn.ReLU(),
                )
            )
        up_conv.append(nn.Linear(
            enc_dims,
            enc_dims
        ))

        self.up_conv = nn.Sequential(*up_conv)
        self.up_conv = self.up_conv.to(self.devs[0])

        # initialize decoder
        self.n_subdecoders = len(self.devs)
        self.layers_per_subdecoder = num_decoder_layers // self.n_subdecoders
        
        decoder_layer = nn.TransformerDecoderLayer(
            embedding_dim,
            nhead,
            dim_feed_forward,
            dropout=decoder_dropout,
            activation='relu',
            batch_first=True,
        )

        self.decoder = nn.ModuleList([
            nn.TransformerDecoder(
                decoder_layer,
                self.layers_per_subdecoder,
            ).to(dev)
            for dev in self.devs
        ])

        # initialize classification head
        classifier= []
        for layer in range(mlp_layers):
            input_size = embedding_dim if layer == 0 else dim_feed_forward
            classifier.append(
                nn.Sequential(
                    nn.Linear(
                        input_size,
                        dim_feed_forward,
                    ),
                    #nn.LayerNorm(
                        #(dim_feed_forward,),
                    #),
                    nn.ReLU(),
                    nn.Dropout(p=mlp_dropout),
                )
            )
            nn.init.kaiming_uniform_(classifier[-1][0].weight)

        classifier.append(nn.Linear(
            dim_feed_forward,
            V
        ))

        self.mlp = nn.Sequential(*classifier).to(self.devs[-1])

    def forward(self, src_text, tgt_tokens, tgt_pad_mask):

        # for masking see https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

        encodings = self.encoder.encode(
            src_text,
            convert_to_tensor=True,
            batch_size=self.enc_batch_size,
            device=self.devs[0],
        )

        # up-conv and pad
        encodings = encodings[:, None, :]
        mem = self.up_conv(encodings)

        N, S_enc, E = mem.shape
        mem_padding = torch.full(
            (N, self.max_seq_len-S_enc, E), 
            self.tokenizer.pad_token_id
        ).to(encodings.device)
        
        mem = torch.hstack((mem, mem_padding))

        # create mask for up-convolutional encodings
        mem_pad_mask = torch.ones(N, self.max_seq_len, dtype=torch.bool)
        mem_pad_mask[:, :S_enc] = False
        mem_pad_mask = mem_pad_mask.to(encodings.device)

        # get embeddings
        embs = self.embeddings(tgt_tokens.to('cpu'))

        # apply positional encodings
        scores = self.pos(embs)

        scores = scores.to(self.devs[0])

        # set up attention mask
        N, S, E = scores.shape
        causal_attn_mask = \
            torch.triu(torch.ones(S, S, dtype=torch.bool))
        causal_attn_mask.fill_diagonal_(False)

        for dev, dec in zip(self.devs, self.decoder):
            try:
                scores = dec(
                    scores.to(dev),
                    mem.to(dev),
                    tgt_mask=causal_attn_mask.to(dev),
                    memory_mask=causal_attn_mask.to(dev),
                    tgt_key_padding_mask=tgt_pad_mask.to(dev),
                    memory_key_padding_mask=mem_pad_mask.to(dev),
                )
            except Exception as e:
                display.error(str(e))
                breakpoint()

        scores = self.mlp(scores)

        return scores

    def forward_decode(self, encodings, pred_tokens, pred_pad_mask):

        # up-conv and pad
        encodings = encodings[:, None, :]
        mem = self.up_conv(encodings)

        N, S_enc, E = mem.shape
        mem_padding = torch.full(
            (N, self.max_seq_len-S_enc, E), 
            self.tokenizer.pad_token_id
        ).to(encodings.device)
        
        mem = torch.hstack((mem, mem_padding))

        # create mask for up-convolutional encodings
        mem_pad_mask = torch.ones(N, self.max_seq_len, dtype=torch.bool)
        mem_pad_mask[:, :S_enc] = False
        mem_pad_mask = mem_pad_mask.to(encodings.device)

        # get embeddings
        embs = self.embeddings(pred_tokens.to('cpu'))

        # apply positional encodings
        scores = self.pos(embs)

        scores = scores.to(self.devs[0])

        # set up attention mask
        N, S, E = scores.shape
        causal_attn_mask = \
            torch.triu(torch.ones(S, S, dtype=torch.bool))
        causal_attn_mask.fill_diagonal_(False)

        for dev, dec in zip(self.devs, self.decoder):
            try:
                scores = dec(
                    scores.to(dev),
                    mem.to(dev),
                    tgt_mask=causal_attn_mask.to(dev),
                    #memory_mask=causal_attn_mask.to(dev),
                    tgt_key_padding_mask=pred_pad_mask.to(dev),
                    memory_key_padding_mask=mem_pad_mask.to(dev),
                )
            except Exception as e:
                display.error(str(e))
                breakpoint()

        scores = self.mlp(scores)

        return scores

    def decode(self, encodings):
    
        N = len(encodings)

        # initialize the tensor of tokens with bos token at [:, 0] and pad everywhere else
        pred_tokens = torch.full(
            (N, self.max_seq_len),
            self.tokenizer.pad_token_id
        )
        pred_tokens = pred_tokens.to(next(self.embeddings.parameters()).device)
        pred_tokens[:, 0] = self.tokenizer.bos_token_id

        pred_pad_mask = torch.ones(pred_tokens.shape, dtype=torch.bool)


        # keep track of whether an eos token is generated for each text sample
        eos_indices = torch.full((N,), -1)

        eos_token = self.tokenizer.eos_token_id
        pad_flags = torch.ones((N,), dtype=torch.bool)

        # run inference loop
        for i in range(self.max_seq_len-1):

            # unmask next target token
            pred_pad_mask[pad_flags, i] = False

            scores = self.forward_decode(encodings, pred_tokens, pred_pad_mask)

            preds = torch.argmax(scores, dim=-1)
            next_tokens = preds[:, i]
            pred_tokens[:, i+1] = next_tokens

            # set flags that stop opening up the pad mask
            pad_flags[next_tokens == eos_token] = False
            
            # break if eos hit for all samples
            if torch.all(eos_indices > 0):
                break
        
        output_sents = self.tokenizer.batch_decode(pred_tokens)
        
        return output_sents


    def encode(self, src_text):
        """
        retrieve text embeddings from the sentence encoder

        Input
            
        """

        encodings = self.encoder.encode(
            src_text,
            convert_to_tensor=True,
            batch_size=self.enc_batch_size,
            device=self.devs[0],
        )

        return encodings



class TextAutoencoder(nn.Module):
    
    def __init__(self, cfg):

        super(TextAutoencoder, self).__init__()

        ################ set defaults from cfg ################

        # temporary vars
        V = cfg.model['vocab_size']
        embedding_dim = cfg.model.get('embedding_dim', 768)
        num_encoder_layers = cfg.model.get('num_encoder_layers', 6)
        num_decoder_layers = cfg.model.get('num_decoder_layers', 6)
        nhead = cfg.model.get('nhead', 8)
        dim_feed_forward = cfg.model.get('dim_feed_forward', 768)
        dropout = cfg.model.get('dropout', .1)
        mlp_dropout = cfg.model.get('mlp_dropout', .1)
        pad_id = cfg.model.get('pad_token_id')
        deconv_seq = cfg.model.get('deconv_sequece', [16, 64, 128])
        deconv_kernels = cfg.model.get('deconv_kernel_sizes', [5, 9, 33])
        decode_style = cfg.model.get('decode_style', None)
        pos_encoding_type = cfg.model.get('pos_encoding_type', 'learned')

        # persistent vars
        self.max_seq_len = cfg.data['max_seq_len']
        self.devs = cfg.model['devices']
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.data['tokenizer_name'])
        self.decode_style = decode_style
        self.pos_encoding_type = pos_encoding_type

        #######################################################

        # initialize decoder
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead,
            dim_feed_forward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_encoder_layers,
        )

        #initialize embeddings and positional encodings
        self.enc_embeddings = nn.Embedding(
            V,
            embedding_dim,
            padding_idx = pad_id
        )
        #initialize embeddings and positional encodings
        self.dec_embeddings = nn.Embedding(
            V,
            embedding_dim,
            padding_idx = pad_id
        )

        if self.pos_encoding_type == 'sinusoid':
            self.enc_pos = PositionalEncoding(
                embedding_dim,
            )
            self.dec_pos = PositionalEncoding(
                embedding_dim,
            )
        elif self.pos_encoding_type == 'learned':
            self.enc_pos = nn.Parameter(
                torch.randn((self.max_seq_len, embedding_dim))
            )
            self.dec_pos = nn.Parameter(
                torch.randn((self.max_seq_len, embedding_dim))
            )
        else:
            pos_enc_types = ['learned', 'sinusoid']
            display.error(f'invalid positional encoding type. must be one of {pos_enc_types}')
            raise ValueError()

        if decode_style == 'conv':

            # initialize up-conv layers
            deconv = []
            deconv_seq = [1] + deconv_seq
            for in_channels, out_channels, kernel in zip(
                deconv_seq[:-1],
                deconv_seq[1:],
                deconv_kernels
            ):
                deconv.append(nn.Sequential(
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel,
                            stride=1,
                            padding=(kernel-1)//2,
                        ),
                        nn.ReLU(),
                    )
                )
            deconv.append(nn.Linear(
                embedding_dim,
                embedding_dim, 
            ))

            self.deconv = nn.Sequential(*deconv)
            self.deconv = self.deconv

        elif decode_style == 'gate':
            breakpoint()

        elif decode_style == 'add':
            breakpoint()

        else:
            styles = ['conv', 'gate', 'add']
            display.error(f'invalid argument for decode_style. style should be one of {styles}')
            raise ValueError()

        # initialize decoder
        decoder_layer = nn.TransformerDecoderLayer(
            embedding_dim,
            nhead,
            dim_feed_forward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_decoder_layers,
        )

        self.linear = nn.Linear(dim_feed_forward, V)

    def forward(self, src_tokens, src_pad_mask,  tgt_tokens, tgt_pad_mask):

        dev = src_tokens.device

        encodings = self.forward_encode(src_tokens, src_pad_mask)
        scores = self.forward_decode(encodings, tgt_tokens, tgt_pad_mask)

        return scores

    def forward_encode(self, src_tokens, src_pad_mask):
        
        dev = self.enc_embeddings.weight.device

        src_embeddings = self.enc_embeddings(src_tokens)
        if self.pos_encoding_type == 'sinusoid':
            src_embeddings = self.enc_pos(src_embeddings)
        elif self.pos_encoding_type == 'learned':
            src_embeddings += self.enc_pos
        else:
            display.error('invalid positional encoding type')
            raise ValueError()

        encodings = self.encoder(
            src_embeddings,
            src_key_padding_mask=src_pad_mask
        )

        # for some reasong the encoder reduces the diensions so i'm padding the output encodings
        N, S, E = encodings.shape
        if S < self.max_seq_len:
            padding = torch.zeros((N, self.max_seq_len-S, E)).to(dev)
            encodings = torch.hstack((encodings, padding))

        # take mean of encodings to get sentence embeddings
        encodings.masked_fill_(src_pad_mask[:, :, None], 0)
        encodings = torch.mean(encodings, dim=-2)

        return encodings


    def forward_decode(self, encodings, tgt_tokens, tgt_pad_mask):
        # for masking see https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

        dev = self.enc_embeddings.weight.device

        if self.decode_style == 'conv':
            # up-conv and pad
            encodings = encodings[:, None, :]
            mem = self.deconv(encodings)
        else:
            # TODO create section for gating
            breakpoint()
            raise ValueError()

        N, S_enc, E = mem.shape
        mem_padding = torch.full(
            (N, self.max_seq_len-S_enc, E), 
            self.tokenizer.pad_token_id
        ).to(dev)
        
        mem = torch.hstack((mem, mem_padding))

        # create mask for up-convolutional encodings
        mem_pad_mask = torch.ones(N, self.max_seq_len, dtype=torch.bool).to(dev)
        mem_pad_mask[:, :S_enc] = False

        # get decoder embeddings
        dec_embeddings = self.dec_embeddings(tgt_tokens)

        # apply positional encodings
        if self.pos_encoding_type == 'sinusoid':
            scores = self.dec_pos(dec_embeddings)
        elif self.pos_encoding_type == 'learned':
            scores = dec_embeddings + self.dec_pos

        # set up attention mask
        N, S, E = scores.shape
        causal_attn_mask = \
            torch.triu(torch.ones(S, S, dtype=torch.bool, device=dev))
        causal_attn_mask.fill_diagonal_(False)

        scores = self.decoder(
            scores,
            mem,
            tgt_mask=causal_attn_mask,
            #memory_mask=causal_attn_mask.to(dev),
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_pad_mask,
        )

        scores = self.linear(scores)

        return scores

    def encode(self, src_text):
        """
        retrieve text embeddings from the sentence encoder

        Input
            
        """

        dev = self.enc_embeddings.weight.device

        tokens = self.tokenizer(
            src_text,
            max_length=self.max_seq_len+1,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        tokens = tokens.to(dev)

        input_ids = tokens['input_ids'][:, 1:].clone()
        attn_mask = ~tokens['attention_mask'][:, 1:].to(torch.bool).clone()
        attn_mask[input_ids == self.tokenizer.eos_token_id] == True
        
        with torch.no_grad():
            encodings = self.forward_encode(input_ids, attn_mask)

        return encodings

    def decode(self, encodings, skip_special_tokens=True):
    
        dev = self.enc_embeddings.weight.device
        N = len(encodings)

        # initialize the tensor of tokens with bos token at [:, 0] and pad everywhere else
        pred_tokens = torch.full(
            (N, self.max_seq_len),
            self.tokenizer.pad_token_id,
            device=dev
        )
        pred_tokens = pred_tokens.to(dev)
        pred_tokens[:, 0] = self.tokenizer.bos_token_id

        pred_pad_mask = torch.ones(pred_tokens.shape, dtype=torch.bool, device=dev)


        # keep track of whether an eos token is generated for each text sample
        eos_indices = torch.full((N,), -1)

        eos_token = self.tokenizer.eos_token_id
        pad_flags = torch.ones((N,), dtype=torch.bool)

        # run inference loop
        for i in range(self.max_seq_len-1):

            # unmask next target token
            pred_pad_mask[pad_flags, i] = False

            scores = self.forward_decode(encodings, pred_tokens, pred_pad_mask)

            preds = torch.argmax(scores, dim=-1)
            next_tokens = preds[:, i]

            # set flags that stop opening up the pad mask
            pad_flags[next_tokens == eos_token] = False
            pred_tokens[pad_flags, i+1] = next_tokens[pad_flags]
            
            # break if eos hit for all samples
            if torch.all(eos_indices > 0):
                break
        
        output_sents = self.tokenizer.batch_decode(
            pred_tokens,
            skip_special_tokens=skip_special_tokens
        )
        
        return output_sents
