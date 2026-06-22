import numpy as np
import torch
import datasets


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2_mine(nsamples, model_path,  seed, seqlen):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # traindata = load_dataset("Salesforce/wikitext", 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset("Salesforce/wikitext", 'wikitext-2-raw-v1', split='test')

   
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # num_chunks = 50
    num_chunks = testenc.input_ids.size(1) // seqlen
    chunked_data = testenc.input_ids[:, :num_chunks * seqlen].view(-1, seqlen)
    test_data = [chunked_data[i][None] for i in range(chunked_data.shape[0])]

    chunked_data_training = trainenc.input_ids[:, :num_chunks * seqlen].view(-1, seqlen)
    train_data_for_test = [chunked_data_training[i][None] for i in range(chunked_data_training.shape[0])]
    
    import random
    random.seed(seed)
    ########## calibration data for GPTQ Method ############
    # trainloader = []
    # for _ in range(nsamples):
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))

    ########## calibration data for GPTQ Method ############
    calibloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attn_mask = trainenc.attention_mask[:, i:j]
        calibloader.append((inp, attn_mask))
    # return calibloader, test_data, trainenc, testenc, train_data_for_test
    return calibloader, test_data, trainenc,  testenc, train_data_for_test

   


def get_wikitext2(tokenizer, seqlen):
    testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testloader = []
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    # num_chunks = testenc.input_ids.size(1) // seqlen
    num_chunks = 100
    print(seqlen, num_chunks)
    chunked_data = testenc.input_ids[:, :num_chunks * seqlen].view(-1, seqlen)

    return [chunked_data[i][None] for i in range(chunked_data.shape[0])]

# def get_wikitext2(tokenizer, seqlen):
#     testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
#     testloader = []
#     testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
#     num_chunks = testenc.input_ids.size(1) // seqlen
#     print(seqlen, num_chunks)
#     chunked_data = testenc.input_ids[:, :num_chunks * seqlen].view(-1, seqlen)

#     return [chunked_data[i][None] for i in range(chunked_data.shape[0])]



# def get_c4(nsamples, seed, seqlen, model):
#     from datasets import load_dataset
#     traindata = load_dataset(
#         'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
#     )
#     valdata = load_dataset(
#         'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
#     )

#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

#     import random
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if trainenc.input_ids.shape[1] >= seqlen:
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))

#     import random
#     random.seed(0)
#     valenc = []
#     for _ in range(256):
#         while True:
#             i = random.randint(0, len(valdata) - 1)
#             tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
#             if tmp.input_ids.shape[1] >= seqlen:
#                 break
#         i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         valenc.append(tmp.input_ids[:, i:j])
#     valenc = torch.hstack(valenc)
#     class TokenizerWrapper:
#         def __init__(self, input_ids):
#             self.input_ids = input_ids
#     valenc = TokenizerWrapper(valenc)

#     return trainloader, valenc 


# def get_c4_new(nsamples, seed, seqlen, model):
#     from datasets import load_dataset
#     traindata = load_dataset(
#         'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
#     )
#     valdata = load_dataset(
#         'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
#     )

#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

#     import random
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if trainenc.input_ids.shape[1] >= seqlen:
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))

#     valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
#     valenc = valenc.input_ids[:, :(256 * seqlen)]

#     class TokenizerWrapper:
#         def __init__(self, input_ids):
#             self.input_ids = input_ids
#     valenc = TokenizerWrapper(valenc)

#     return trainloader, valenc


def get_c4_testenc(model_path, seqlen, nval=1100):
    """
    Load C4 validation split, concatenate nval documents, tokenise, and return
    a TokenizerWrapper with .input_ids of shape (1, N) — same format as the
    testenc returned by get_wikitext2_mine, compatible with evaluator_newVersion.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    valenc = tokenizer(' '.join(valdata[:nval]['text']), return_tensors='pt')

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return TokenizerWrapper(valenc.input_ids)


def get_loaders(
    name, model_path, nsamples=128, seed=0, seqlen=1024):
    
    if 'wikitext' in name:
        # return get_wikitext2(nsamples,model_path, seed, seqlen)
        return get_wikitext2_mine(nsamples,model_path, seed, seqlen)
    # if 'ptb' in name:
    #     if 'new' in name:
    #         return get_ptb_new(nsamples, seed, seqlen, model)
    #     return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)