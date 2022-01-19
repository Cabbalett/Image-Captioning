from typing import List, Dict, Tuple, Sequence, Any
from collections import Counter, defaultdict
from itertools import chain
import torchvision.transforms as transforms
from PIL import Image
import random
import torch
from pycocotools.coco import COCO
from tqdm import tqdm
random.seed(1234)

class Language(Sequence[List[str]]):
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_IDX = 1
    SOS_TOKEN = '<SOS>'
    SOS_TOKEN_IDX = 2
    EOS_TOKEN = '<EOS>'
    EOS_TOKEN_IDX = 3

    def __init__(self, anns: List) -> None:

        self._sentences: List[List[str]] = [self.tokenize(ann['caption']) for ann in anns]

        self.word2idx: Dict[str, int] = None
        self.idx2word: List[str] = None

    def tokenize(self, ann: List):
        words = ann.split()
        if '.' in words[-1]:
            words[-1] = words[-1][:-1]
        return words
    
    def build_vocab(self, min_freq: int=1) -> None:
        SPECIAL_TOKENS: List[str] = [Language.PAD_TOKEN, Language.UNK_TOKEN, Language.SOS_TOKEN, Language.EOS_TOKEN]
        self.idx2word = SPECIAL_TOKENS + [word for word, count in Counter(chain(*self._sentences)).items() if count >= min_freq]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
    
    def set_vocab(self, word2idx: Dict[str, int], idx2word: List[str]) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def __getitem__(self, index: int) -> List[str]:
        return self._sentences[index]
    
    def __len__(self) -> int:
        return len(self._sentences)

class NMTDataset():
    def __init__(self, anns: List, tgt: Language, coco,dataDir, dataType, max_len: int=30) -> None:
        assert tgt.word2idx is not None
        self._coco = coco
        self.dataDir = dataDir
        self.dataType = dataType
        self._anns = anns
        self._tgt = tgt
        self._max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[int, List[str]]:
        img = self._coco.loadImgs(self._anns[index]['image_id'])[0]
        img = Image.open('%s/images/%s/%s'%(self.dataDir,self.dataType,img['file_name'])).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(preprocess(self._anns[index]['caption'].split(), self._tgt.word2idx, self._max_len))

    def __len__(self) -> int:
        return len(self._anns)

def preprocess(
    raw_tgt_sentence: List[str],
    tgt_word2idx: Dict[str, int],
    max_len: int
) -> Tuple[List[int], List[int]]:

    UNK = Language.UNK_TOKEN_IDX
    SOS = Language.SOS_TOKEN_IDX
    EOS = Language.EOS_TOKEN_IDX


    tgt_sentence = []
    
    for word in raw_tgt_sentence:
        if '.' in word:
            word = word[:-1]
        if word in tgt_word2idx:
            tgt_sentence.append(tgt_word2idx[word])
        else:
            tgt_sentence.append(UNK)


    tgt_sentence = [SOS] + tgt_sentence[:max_len-2] + [EOS]
    return tgt_sentence

def bucketed_batch_indices(
    sentence_length: List[int],
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:

    batch_map = defaultdict(list)
    batch_indices_list = []

    tgt_len_min = min(sentence_length)

    for idx, tgt_len in enumerate(sentence_length):
        tgt = (tgt_len - tgt_len_min + 1) // (5)
        batch_map[tgt].append(idx)

    for key, value in batch_map.items():
        batch_indices_list += [value[i: i+batch_size] for i in range(0, len(value), batch_size)]

    random.shuffle(batch_indices_list)
    batch_indices_list = [x for x in batch_indices_list if len(x)==batch_size]

    return batch_indices_list

def collate_fn(
    batched_samples: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:

    batched_samples = sorted(batched_samples, key=lambda sample: len(sample[1]), reverse=True)
    # import pdb;pdb.set_trace()
    src_images = list(map(lambda sample: sample[0], batched_samples))
    src_images = torch.stack(src_images)
    tgt_sentences = []
    for sample in batched_samples:
        tgt_sentences.append(sample[1])

    tgt_sentences = torch.nn.utils.rnn.pad_sequence(tgt_sentences, batch_first=True)

    return src_images, tgt_sentences

def get_dataloaders(args):

    train_dataType=f'train{args.dataType}'
    annFile='{}/annotations/captions_{}.json'.format(args.dataDir,train_dataType)
    train_coco=COCO(annFile)

    val_dataType=f'val{args.dataType}'
    annFile='{}/annotations/captions_{}.json'.format(args.dataDir,val_dataType)
    val_coco=COCO(annFile)

    train_imgIds = train_coco.getImgIds()[:200]
    train_annIds = train_coco.getAnnIds(imgIds=train_imgIds)
    train_anns = train_coco.loadAnns(train_annIds)

    val_imgIds = val_coco.getImgIds()[:100]
    val_annIds = val_coco.getAnnIds(imgIds=val_imgIds)
    val_anns = val_coco.loadAnns(val_annIds)

    vocab = Language(train_anns)
    vocab.build_vocab()

    train_dataset = NMTDataset(train_anns, vocab, train_coco, args.dataDir, train_dataType)
    val_dataset = NMTDataset(val_anns, vocab, val_coco, args.dataDir, val_dataType)

    print("calculating sentence length...")
    train_sentence_length = list(map(lambda x: len(x[1]), tqdm(train_dataset)))
    val_sentence_length = list(map(lambda x: len(x[1]), tqdm(val_dataset)))

    max_pad_len = 5

    train_dataloader = torch.utils.data.dataloader.DataLoader(train_dataset, collate_fn=collate_fn, num_workers=2, batch_sampler=bucketed_batch_indices(train_sentence_length, batch_size=args.batch_size, max_pad_len=max_pad_len))
    val_dataloader = torch.utils.data.dataloader.DataLoader(val_dataset, collate_fn=collate_fn, num_workers=2, batch_sampler=bucketed_batch_indices(val_sentence_length, batch_size=args.batch_size, max_pad_len=max_pad_len))

    return train_dataloader, val_dataloader, len(vocab.word2idx)


