"""
this is for dealing with repetitive tokenizer code
"""

# external imports
import tokenizers
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    Tokenizer,
    pre_tokenizers,
)

# internal imports
from . import (
    strings,
    files
)

def fetch_tokenizer(
    path_str: str,
    ds,
    trgt_vocab_size=50000,
    min_freq=100,
    batch_size=1024,
    ver='bpe',
    override=False,
    special_tokens={'pad': '[PAD]'},
):
    """
    tries to read in an existing tokenizer and if it doesn't exists, one is created and saved.
    this does not allow you to customize the tokenizer so in the future i'll proabbly fix that
    """

    def batch_iterator(batch_size=1024):
        for i in range(0, len(ds['train']), batch_size):
            batch = [
                x if x is not None else ''
                for x in ds['train'][i : i + batch_size]["text"]
            ]
            yield batch

    path_str += \
            f'/v_{trgt_vocab_size}' + \
            f'/min_freq_{min_freq}/{ver}/tokenizer.json'
    if not files.path_exists(path_str) or override:
        if override:
            print(strings.green(
                'tokenizer exists but override flag is raised. training new tokenizer ...'
            ))
        else:
            print(strings.green(
                'tokenizer does not exist. training tokenizer ...'
            ))
        if ver == 'bpe':    
            model = tokenizers.models.BPE()
            trainer = tokenizers.trainers.BpeTrainer
        elif ver == 'word_level':
            model = tokenizers.models.WordLevel()
            trainer = tokenizers.trainers.WordLevelTrainer
        elif ver == 'word_piece':
            model = tokenizers.models.WordPiece()
            trainer = tokenizers.trainers.WordPieceTrainer
        else:
            raise ValueError('selected tokenizer is not implemented')

        trainer = trainer(
            vocab_size = trgt_vocab_size,
            min_frequency = min_freq,
            show_progress=True,
            special_tokens=list(special_tokens.values())
        )
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        print(strings.green(
            'training tokenizer'
        ))

        tokenizer.train_from_iterator(
            batch_iterator(batch_size=batch_size),
            trainer=trainer,
            length=len(ds['train']),
        )

        files.create_path(path_str, is_file=True)
        tokenizer.save(path_str)
        print(strings.green(
            f'tokenizer saved to {path_str}'
        ))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=special_tokens.get('bos', None),
            eos_token=special_tokens.get('eos', None),
            unk_token=special_tokens.get('unk', None),
            sep_token=special_tokens.get('sep', None),
            pad_token=special_tokens.get('pad', None),
            cls_token=special_tokens.get('cls', None),
            mask_token=special_tokens.get('mask', None),
        )

    else:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=path_str,
            bos_token=special_tokens.get('bos', None),
            eos_token=special_tokens.get('eos', None),
            unk_token=special_tokens.get('unk', None),
            sep_token=special_tokens.get('sep', None),
            pad_token=special_tokens.get('pad', None),
            cls_token=special_tokens.get('cls', None),
            mask_token=special_tokens.get('mask', None),
        )
        print(strings.green(
            f'loaded tokenizer from {path_str}'
        ))

    return tokenizer


