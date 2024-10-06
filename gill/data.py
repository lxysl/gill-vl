"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple, List

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from gill import utils


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, processor, tokenizer, precision: str = 'fp32') -> Dataset:
  assert split in ['train', 'val'
    ], 'Expected split to be one of "train" or "val", got {split} instead.'

  dataset_paths = []
  image_data_dirs = []
  train = split == 'train'

  # Default configs for datasets.
  # Folder structure should look like:
  if split == 'train':
    if 'cc3m' in args.dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/training/'))
    else:
      raise NotImplementedError

  elif split == 'val':
    if 'cc3m' in args.val_dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_val.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/validation'))
    else:
      raise NotImplementedError

    assert len(dataset_paths) == len(image_data_dirs) == 1, (dataset_paths, image_data_dirs)
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1:
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      CsvDataset(path, image_dir, processor, 'image',
        'caption', args.visual_model, tokenizer, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
        num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens, input_prompt=args.input_prompt)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], processor, 'image',
      'caption', args.visual_model, tokenizer, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
      num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens, input_prompt=args.input_prompt)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset


class CsvDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, processor, img_key,
               caption_key, feature_extractor_model: str=None, tokenizer=None,
               train: bool = True, max_len: int = 200, sep="\t", precision: str = 'fp32',
               image_size: int = 224, retrieval_token_idx: List[int] = [-1], gen_token_idx: List[int] = [-1],
               num_tokens: int = 1, num_clip_tokens: int = 1, input_prompt: str = ""):
    logging.debug(f'Loading tsv data from {input_filename}.')
    df = pd.read_csv(input_filename, sep=sep)

    self.base_image_dir = base_image_dir
    self.images = df[img_key].tolist()
    self.captions = df[caption_key].tolist()
    assert len(self.images) == len(self.captions)

    self.processor = processor
    self.image_size = image_size

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision
    self.retrieval_token_idx = retrieval_token_idx
    self.gen_token_idx = gen_token_idx
    self.num_tokens = num_tokens
    self.num_clip_tokens = num_clip_tokens

    self.font = None

    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    while True:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      clip_l_path = os.path.join(self.base_image_dir, 'clip_embs', str(self.images[idx]) + '.npy')
      # caption = "A picture of" + caption + [IMG0]...[IMG7]
      caption = str(self.captions[idx])
      caption = self.input_prompt + caption
      for i in range(self.num_tokens):
        caption += f'[IMG{i}]'
      messages = [  # just for fitting the format to extract image pixel_values
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": caption,
                }
            ]
        }
      ]

      try:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)

        # Only load if we are in generation mode.
        with open(clip_l_path, 'rb') as f:
          clip_emb = np.load(f, allow_pickle=True)   # (num_clip_tokens, 768)
          clip_emb = clip_emb[:self.num_clip_tokens, :]

        inputs = self.processor(
          text=text,
          images=image_inputs,
          videos=None,
          padding="max_length",  # caution: left padding
          max_length=self.max_len,
          truncation=True,  # truncation at right
          return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]
        input_len = inputs.attention_mask[0].sum()
        images = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw

        # input_ids: [...<|endoftext|>, <|im_start|>...<|im_start|>assistant\nA picture of caption[IMG0]...[IMG7], <|im_end|>]
        # labels:    [...         -100, x         ,-100,          x,     -100,x,      ...       x,-100, ..., -100, x         ]
        # ignore_index at: <|endoftext|>, system prompt, user prompt, assistant\n, [IMG0]...[IMG7]
        # supervised_index at: <|im_start|>, <|im_end|>, A picture of caption

        labels = self.tokenizer.encode(caption, add_special_tokens=False, return_tensors="pt")[0]
        labels[-8:] = -100
        labels = torch.cat([labels, torch.LongTensor([self.tokenizer.eos_token_id])])
        prefix_labels = input_ids[:-len(labels)].clone()
        for k, token in enumerate(prefix_labels):
          if token not in self.tokenizer.encode("<|im_start|><|im_end|>"):
            prefix_labels[k] = -100
        labels = torch.cat([prefix_labels, labels])

        # If IMG tokens are truncated, replace them with the correct token.
        # Qwen2-VL uses left padding, so the last token will always be eos when input length is less than max_len.
        if input_ids[-1] != self.tokenizer.eos_token_id:
          input_ids[-self.num_tokens-1:-1] = torch.tensor(self.gen_token_idx).to(dtype=input_ids.dtype, device=input_ids.device)
          input_ids[-1] = self.tokenizer.eos_token_id

        decode_caption = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        cap_img = utils.create_image_of_text(decode_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)

        return image_path, images, image_grid_thw, cap_img, input_ids, input_len, input_ids, input_len, labels, clip_emb
      except Exception as e:
        print(f'Error reading for {image_path} with caption {caption}: {e}')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)
