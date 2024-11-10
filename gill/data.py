"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple, List

import collections
import logging
import traceback
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


def custom_images_collate_fn(batch):
  image_path = [x[0] for x in batch]
  raw_images = torch.stack([x[1] for x in batch])
  images = [torch.tensor(x[2]) for x in batch]  # list of pixel values
  image_grid_thw = torch.stack([torch.tensor(x[3]) for x in batch])
  cap_img = torch.stack([x[4] for x in batch])
  captioning_input_ids = torch.stack([x[5] for x in batch])
  captioning_labels = torch.stack([x[6] for x in batch])
  captioning_start_id = torch.tensor([x[7] for x in batch])
  captioning_end_id = torch.tensor([x[8] for x in batch])
  gen_input_ids = torch.stack([x[9] for x in batch])
  gen_labels = torch.stack([x[10] for x in batch])
  gen_start_id = torch.tensor([x[11] for x in batch])
  gen_end_id = torch.tensor([x[12] for x in batch])
  clip_emb = torch.stack([torch.tensor(x[13]) for x in batch])
  return image_path, raw_images, images, image_grid_thw, cap_img, captioning_input_ids, captioning_labels, captioning_start_id, captioning_end_id, gen_input_ids, gen_labels, gen_start_id, gen_end_id, clip_emb


def custom_images_collate_fn_2(batch):
  image_path = [x[0] for x in batch]
  raw_images = torch.stack([x[1] for x in batch])
  images = [torch.tensor(x[2]) for x in batch]  # list of pixel values
  image_grid_thw = torch.stack([torch.tensor(x[3]) for x in batch])
  cap_img = torch.stack([x[4] for x in batch])
  captioning_input_ids = torch.stack([x[5] for x in batch])
  captioning_labels = torch.stack([x[6] for x in batch])
  captioning_start_id = torch.tensor([x[7] for x in batch])
  captioning_end_id = torch.tensor([x[8] for x in batch])
  gen_input_ids = torch.stack([x[9] for x in batch])
  gen_labels = torch.stack([x[10] for x in batch])
  gen_start_id = torch.tensor([x[11] for x in batch])
  gen_end_id = torch.tensor([x[12] for x in batch])
  clip_emb = torch.stack([torch.tensor(x[13]) for x in batch])
  emb_2 = torch.stack([torch.tensor(x[14]) for x in batch])
  return image_path, raw_images, images, image_grid_thw, cap_img, captioning_input_ids, captioning_labels, captioning_start_id, captioning_end_id, gen_input_ids, gen_labels, gen_start_id, gen_end_id, clip_emb, emb_2


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
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train_sampled.tsv'))
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
        'caption', None, tokenizer, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
        num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens, num_emb2_tokens=args.num_emb2_tokens, input_prompt=args.input_prompt, text_fc_mode=args.text_fc_mode)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], processor, 'image',
      'caption', None, tokenizer, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
      num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens, num_emb2_tokens=args.num_emb2_tokens, input_prompt=args.input_prompt, text_fc_mode=args.text_fc_mode)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset


class CsvDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, processor, img_key,
               caption_key, feature_extractor_model: str=None, tokenizer=None,
               train: bool = True, max_len: int = 128, sep="\t", precision: str = 'fp32',
               image_size: int = 224, retrieval_token_idx: List[int] = [-1], gen_token_idx: List[int] = [-1],
               num_tokens: int = 1, num_clip_tokens: int = 1, num_emb2_tokens: int = 1, input_prompt: str = "", text_fc_mode: str = 'gill_mapper'):
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
    self.num_emb2_tokens = num_emb2_tokens
    self.input_prompt = input_prompt
    self.text_fc_mode = text_fc_mode

    self.font = None

    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    while True:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      if self.text_fc_mode == 'gill_mapper_hunyuan':
        clip_l_path = os.path.join(self.base_image_dir, 'hydit_clip_embs', str(self.images[idx]) + '.npy')
        emb_2_path = os.path.join(self.base_image_dir, 'hydit_t5_embs', str(self.images[idx]) + '.npy')
      elif self.text_fc_mode == 'gill_mapper_kolors':
        clip_l_path = os.path.join(self.base_image_dir, 'kolors_embs', str(self.images[idx]) + '.npy')
        emb_2_path = os.path.join(self.base_image_dir, 'kolors_pooled_embs', str(self.images[idx]) + '.npy')
      else:
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
        # Only load if we are in generation mode.
        with open(clip_l_path, 'rb') as f:
          clip_emb = np.load(f, allow_pickle=True)   # (num_clip_tokens, 768)
          clip_emb = clip_emb[:self.num_clip_tokens, :]
        if self.text_fc_mode in ["gill_mapper_hunyuan", "gill_mapper_kolors"]:
          with open(emb_2_path, 'rb') as f:
            emb_2 = np.load(f, allow_pickle=True)   # (256, 2048) or (4096)
            emb_2 = np.reshape(emb_2, (-1, emb_2.shape[-1]))
            emb_2 = emb_2[:self.num_emb2_tokens, :]

        labels = self.tokenizer.encode(caption, add_special_tokens=False, return_tensors="pt")[0]
        labels[-self.num_tokens:] = -100
        captioning_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[:-1]  # remove the end '\n'

        image_inputs, _ = process_vision_info(messages)
        image_inputs = self.processor.image_processor(images=image_inputs, videos=None)
        images = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"][0]
        raw_images = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
        raw_images = T.ToTensor()(raw_images)

        # --- mode == 'captioning' ---
        # input_ids: [<|im_start|>...<|im_start|>assistant\nA picture of caption[IMG0]...[IMG7], <|im_end|>, <|endoftext|>...]
        # labels:    [x         ,-100,          x,     -100,x,      ...       x,-100, ..., -100, x         , -100         ...]
        # ignore_index at: <|endoftext|>, system prompt, user prompt, assistant\n, [IMG0]...[IMG7]
        # supervised_index at: <|im_start|>, <|im_end|>, A picture of caption

        # --- mode in ['retrival', 'generation'] ---
        # input_ids: [A picture of caption[IMG0]...[IMG7], <|endoftext|>...]
        # labels:    [x,      ...       x,-100, ..., -100, -100         ...]
        # ignore_index at: <|endoftext|>, [IMG0]...[IMG7]
        # supervised_index at: A picture of caption

        merge_length = self.processor.image_processor.merge_size**2
        index = 0
        while "<|image_pad|>" in captioning_text:
          captioning_text = captioning_text.replace(
            "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
          )
          index += 1
        captioning_text = captioning_text.replace("<|placeholder|>", "<|image_pad|>")
        captioning_input_ids = self.tokenizer.encode(
          captioning_text,
          padding_side="right",
          padding="max_length",
          max_length=self.max_len,
          truncation=True,
          add_special_tokens=False,
          return_tensors="pt"
        )[0]
        no_pad_captioning_ids = self.tokenizer.encode(captioning_text)
        captioning_start_id = len(no_pad_captioning_ids) - len(labels) - 1  # [-> A picture of ...]
        captioning_end_id = len(no_pad_captioning_ids) - self.num_tokens - 1  # [... A picture of caption[IMG0] <-]

        prefix_labels = captioning_input_ids[:captioning_start_id].clone()
        for k, token in enumerate(prefix_labels):
          if token not in self.tokenizer.encode("<|im_start|><|im_end|>"):
            prefix_labels[k] = -100
        captioning_labels = torch.cat([prefix_labels, labels, torch.full((self.max_len - len(prefix_labels) - len(labels),), -100, dtype=torch.long, device=labels.device)])

        gen_input_ids = self.tokenizer.encode(
          caption,
          padding_side="right",
          padding="max_length",
          max_length=self.max_len,
          truncation=True,
          add_special_tokens=False,
          return_tensors="pt"
        )[0]
        gen_start_id = 0  # [-> A picture of ...]
        gen_end_id = len(labels) - self.num_tokens  # [... A picture of caption[IMG0] <-]
        gen_labels = torch.cat([labels, torch.full((self.max_len - len(labels),), -100, dtype=torch.long, device=labels.device)])
        
        # If IMG tokens are truncated, replace them with the correct token.
        # Qwen2-VL uses left padding, so the last token will always be eos when input length is less than max_len.
        if captioning_input_ids[-1] not in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
          captioning_input_ids[-self.num_tokens-1:-1] = torch.tensor(self.gen_token_idx).to(dtype=captioning_input_ids.dtype, device=captioning_input_ids.device)
          captioning_input_ids[-1] = self.tokenizer.eos_token_id
          captioning_labels[-self.num_tokens-1:-1] = -100
          captioning_labels[-1] = self.tokenizer.eos_token_id
        if gen_input_ids[-1] not in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
          gen_input_ids[-self.num_tokens-1:-1] = torch.tensor(self.gen_token_idx).to(dtype=gen_input_ids.dtype, device=gen_input_ids.device)
          gen_labels[-self.num_tokens-1:-1] = -100

        decode_caption = self.tokenizer.decode(labels[:-self.num_tokens], skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        cap_img = utils.create_image_of_text(decode_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)

        if self.text_fc_mode in ["gill_mapper_hunyuan", "gill_mapper_kolors"]:
          return image_path, raw_images, images, image_grid_thw, cap_img, captioning_input_ids, captioning_labels, captioning_start_id, captioning_end_id, gen_input_ids, gen_labels, gen_start_id, gen_end_id, clip_emb, emb_2
        else:
          return image_path, raw_images, images, image_grid_thw, cap_img, captioning_input_ids, captioning_labels, captioning_start_id, captioning_end_id, gen_input_ids, gen_labels, gen_start_id, gen_end_id, clip_emb
      except Exception as e:
        print(f'Error reading for {image_path} with caption {caption}: {e}')
        traceback.print_exc()
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)
