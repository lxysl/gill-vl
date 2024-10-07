from typing import List, Optional
from collections import namedtuple
from diffusers import StableDiffusionPipeline
import json
import numpy as np
import os
import glob
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from PIL import Image, UnidentifiedImageError
from requests.exceptions import ConnectionError

from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, Qwen2VLForConditionalGeneration
from gill import utils
from gill import layers


class GILLArgs:
  freeze_lm: bool = True
  freeze_vm: bool = True
  vlm_version: str = 'Qwen/Qwen2-VL-2B-Instruct'
  # visual_encoder: str = 'openai/clip-vit-large-patch14'
  n_visual_tokens: int = 1
  task: str = 'captioning'
  ret_emb_dim: Optional[int] = 256
  gen_emb_dim: Optional[int] = 256
  text_emb_layers: List[int] = [-1]
  gen_token_idx: List[int] = [0]
  retrieval_token_idx: List[int] = [0]
  text_fc_mode: str = 'gill_mapper'
  ret_text_fc_mode: str = 'linear'
  num_tokens: int = 8
  num_clip_tokens: int = 77


class GILLModel(nn.Module):
  def __init__(self, processor, args: GILLArgs = GILLArgs()):
    super().__init__()
    self.processor = processor
    self.tokenizer = processor.tokenizer
    # self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
    self.image_token = self.tokenizer.cls_token_id
    assert args.text_emb_layers != set(args.text_emb_layers), 'text_emb_layers not unique'
    self.args = args
    self.num_tokens = args.num_tokens
    self.num_clip_tokens = args.num_clip_tokens

    vlm_version = args.vlm_version
    n_visual_tokens = args.n_visual_tokens
    print(f"Using {vlm_version} for the language model with {n_visual_tokens} visual tokens.")

    print("Restoring pretrained weights for the vision language model.")
    if 'Qwen2' in vlm_version:
      self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_version, torch_dtype="auto", device_map="auto"
      )
      self.lm = self.vlm.model
      self.visual_model = self.vlm.visual
      embed_dim = self.visual_model.config.embed_dim
    else:
      raise NotImplementedError

    self.vlm_version = vlm_version

    if self.args.freeze_lm:
      self.lm.eval()
      print("Freezing the LM.")
      for param in self.lm.parameters():
        param.requires_grad = False
    else:
      self.lm.train()

    self.retrieval_token_idx = args.retrieval_token_idx
    self.gen_token_idx = args.gen_token_idx
    self.lm.resize_token_embeddings(len(self.tokenizer))

    self.input_embeddings = self.lm.get_input_embeddings()

    if self.args.freeze_vm:
      print("Freezing the VM.")
      self.visual_model.eval()
      for param in self.visual_model.parameters():
        param.requires_grad = False
    else:
      self.visual_model.train()

    self.ret_text_hidden_fcs = nn.ModuleList([])
    self.gen_text_hidden_fcs = nn.ModuleList([])

    for layer_idx in self.args.text_emb_layers:
      if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers):
        if 'Qwen2' in self.vlm_version:  # Qwen2 models
          in_dim = self.lm.config.hidden_size
        else:
          raise NotImplementedError

        self.ret_text_hidden_fcs.append(
          layers.TextFcLayer(in_dim, self.args.ret_emb_dim, num_input_tokens=self.args.num_tokens,
                             num_output_tokens=1, mode=self.args.ret_text_fc_mode))
        self.gen_text_hidden_fcs.append(
          layers.TextFcLayer(in_dim, self.args.gen_emb_dim, num_input_tokens=self.args.num_tokens,
                             num_output_tokens=self.args.num_clip_tokens, mode=self.args.text_fc_mode))

      elif layer_idx < self.lm.config.num_hidden_layers:
        self.ret_text_hidden_fcs.append(
          layers.TextFcLayer(self.lm.config.hidden_size, self.args.ret_emb_dim, num_input_tokens=self.args.num_tokens,
                             num_output_tokens=1, mode=self.args.ret_text_fc_mode))
        self.gen_text_hidden_fcs.append(
          layers.TextFcLayer(self.lm.config.hidden_size, self.args.gen_emb_dim, num_input_tokens=self.args.num_tokens,
                             num_output_tokens=self.args.num_clip_tokens, mode=self.args.text_fc_mode))
      else:
        raise ValueError(f'Embedding of layer {layer_idx} was requested but model only has {self.lm.config.num_hidden_layers} layers.')

    # self.visual_embeddings = nn.Linear(hidden_size, embedding_dim)

    # Retrieval image FC layer.
    self.visual_fc = nn.Linear(embed_dim, self.args.ret_emb_dim)
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


  def get_visual_embs(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor, mode: str = 'captioning'):
    if mode not in ['captioning', 'retrieval', 'generation']:
      raise ValueError(f"mode should be one of ['captioning', 'retrieval', 'generation'], got {mode} instead.")

    # Extract visual embeddings from the vision encoder.
    if 'Qwen2' in self.vlm_version:
      pixel_values = pixel_values.type(self.visual_model.get_dtype())
      image_embeds, image_features = self.visual_model(pixel_values, grid_thw=image_grid_thw)  # image_embeds: text dim, image_features: visual dim
    else:
      raise NotImplementedError

    # Use the correct fc based on function argument.
    if mode == 'captioning':
      visual_embs = image_embeds
    elif mode == 'retrieval':
      visual_embs = self.visual_fc(image_features[:, 0, :])  # (2, D * n_visual_tokens)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))
    elif mode == 'generation':
      visual_embs = torch.zeros((pixel_values.shape[0], 1, 768), device=pixel_values.device)
    else:
      raise NotImplementedError

    return visual_embs


  def train(self, mode=True):
    super(GILLModel, self).train(mode=mode)
    # Overwrite train() to ensure frozen models remain frozen.
    if self.args.freeze_lm:
      self.lm.eval()
    if self.args.freeze_vm:
      self.visual_model.eval()


  def forward(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: torch.LongTensor,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    caption_start_id: Optional[torch.LongTensor] = None,
    caption_end_id: Optional[torch.LongTensor] = None,
    mode: str = 'captioning',
  ):
    visual_embs = self.get_visual_embs(pixel_values, image_grid_thw, mode)

    batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
    assert input_ids.shape[0] == batch_size, (input_ids.shape, visual_embs.shape)
    visual_embs_norm = ((visual_embs ** 2).sum(dim=-1) ** 0.5).mean()

    input_embs = self.input_embeddings(input_ids)  # (N, T, D)
    input_embs_norm = ((input_embs ** 2).sum(dim=-1) ** 0.5).mean()

    if mode == 'captioning':
      # Just add visual embeddings.
      visual_mask = (
        (input_ids == self.vlm.image_token_id)
        .unsqueeze(-1)
        .expand_as(input_embs)
        .to(input_embs.device)
      )
      visual_embs = visual_embs.to(input_embs.device, input_embs.dtype)
      input_embs = input_embs.masked_scatter(visual_mask, visual_embs)

      output = self.lm(inputs_embeds=input_embs,
                       labels=labels,
                       output_hidden_states=True)
    elif mode in ['retrieval', 'generation']:
      # update label of [IMG0] token from -100 to index of [IMG0].
      for input_id, label in zip(input_ids, labels):
        for k, token in enumerate(input_id):
          if token in [self.retrieval_token_idx[0], self.gen_token_idx[0]]:
            label[k] = input_id[k]
            break
      output = self.lm(inputs_embeds=input_embs,
                       labels=labels,
                       output_hidden_states=True)
    else:
      raise NotImplementedError

    last_embedding = None
    last_output_logit = None
    hidden_states = []
    llm_hidden_states = []

    if mode in ['retrieval', 'generation']:
      if mode == 'retrieval':
        text_hidden_fcs = self.ret_text_hidden_fcs
      else:
        text_hidden_fcs = self.gen_text_hidden_fcs

      for idx, fc_layer in zip(self.args.text_emb_layers, text_hidden_fcs):
        input_hidden_state = torch.stack([output.hidden_states[idx][i, caption_end_id[i]:caption_end_id[i]+self.num_tokens, :] for i in range(batch_size)], axis=0)
        input_embedding = torch.stack([input_embs[i, caption_end_id[i]:caption_end_id[i]+self.num_tokens, :] for i in range(batch_size)], axis=0)
        llm_hidden_states.append(input_hidden_state)
        hidden_states.append(fc_layer(input_hidden_state, input_embedding))  # (N, seq_len, 2048)

      # Add hidden states together.
      last_embedding = torch.stack(hidden_states, dim=-1).sum(dim=-1) #torch.stack([last_hidden_state[i, :, :] for i in range(batch_size)], axis=0)  # (N, T, D)
      last_output_logit = torch.stack([output.logits[i, -1, :] for i in range(batch_size)], axis=0)  # (N, D)

      # Compute retrieval loss.
      if mode == 'retrieval':
        assert visual_embs.shape[1] == 1, visual_embs.shape
        assert last_embedding.shape[1] == 1, last_embedding.shape
        visual_embs = visual_embs[:, 0, :]
        visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
        last_embedding = last_embedding[:, 0, :]
        last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        visual_embs = logit_scale * visual_embs
    elif mode == 'captioning':
      pass
    else:
      raise NotImplementedError

    return output, labels, last_embedding, last_output_logit, visual_embs, visual_embs_norm, input_embs_norm, llm_hidden_states

  def generate(self, embeddings = torch.FloatTensor, max_len: int = 32,
               temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 0,
               ret_scale_factor: float = 1.0, gen_scale_factor: float = 1.0,
               filter_value: float = -float('Inf')):
    """Runs greedy decoding and returns generated captions.

    Args:
      min_word_tokens: Minimum number of words to generate before allowing a [IMG] output.
      filter_value: Value to assign to tokens that should never be generated.
    Outputs:
      out: (N, T) int32 sequence of output tokens.
      output_embeddings: (N, T, 256) sequence of text output embeddings.
    """
    self.lm.eval()

    with torch.no_grad():  # no tracking history
      # init output with image tokens
      out = None
      output_embeddings = []
      output_logits = []

      for i in range(max_len):
        output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)

        for idx in self.args.text_emb_layers:
          output_embeddings.append(output.hidden_states[idx])

        logits = output.logits[:, -1, :]  # (N, vocab_size)
        if top_p == 1.0:
          logits = logits.cpu()
        output_logits.append(logits)

        # Prevent the model from generating the [IMG1..n] tokens.
        logits[:, self.retrieval_token_idx[1:]] = filter_value
        logits[:, self.gen_token_idx[1:]] = filter_value

        if (self.retrieval_token_idx or self.gen_token_idx) and self.retrieval_token_idx[0] != -1 and self.gen_token_idx[0] != -1:
          if i < min_word_tokens:
            # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
            logits[:, self.retrieval_token_idx] = filter_value
            logits[:, self.gen_token_idx] = filter_value
          else:
            # Multiply by scaling factor.
            if ret_scale_factor > 1:
              logits[:, self.retrieval_token_idx[0]] = logits[:, self.retrieval_token_idx[0]].abs() * ret_scale_factor
            if gen_scale_factor > 1:
              logits[:, self.gen_token_idx[0]] = logits[:, self.gen_token_idx[0]].abs() * gen_scale_factor

        if temperature == 0.0:
          if top_p != 1.0:
            raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
          next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
        else:
          logits = logits / temperature

          # Apply top-p filtering.
          if top_p < 1.0:
            assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # (N, D)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for j in range(sorted_indices.shape[0]):
              indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
              logits[j, indices_to_remove] = filter_value

          token_weights = logits.exp()   # (N, vocab_size)
          next_token = torch.multinomial(token_weights, 1)  # (N, 1)

        # Force generation of the remaining [IMG] tokens if [IMG0] is generated.
        if next_token.shape[0] == 1 and next_token.item() == self.retrieval_token_idx[0]:
          assert self.retrieval_token_idx == self.gen_token_idx, (self.retrieval_token_idx, self.gen_token_idx)
          next_token = torch.tensor(self.retrieval_token_idx)[None, :].long().to(embeddings.device)  # (1, num_tokens)
        else:
          next_token = next_token.long().to(embeddings.device)

        if out is not None:
          out = torch.cat([out, next_token], dim=-1)
        else:
          out = next_token

        next_embedding = self.input_embeddings(next_token)
        embeddings = torch.cat([embeddings, next_embedding], dim=1)

    return out, output_embeddings, output_logits


class GILL(nn.Module):
  def __init__(self, processor, model_args: Optional[GILLArgs] = None,
               path_array: Optional[List[str]] = None, emb_matrix: Optional[torch.tensor] = None,
               load_sd: bool = False, num_gen_images: int = 1, decision_model_path: Optional[str] = None):
    super().__init__()
    self.model = GILLModel(processor, model_args)
    self.path_array = path_array
    self.emb_matrix = emb_matrix
    self.load_sd = load_sd
    self.num_gen_images = num_gen_images
    self.idx2dec = {0: 'gen', 1: 'ret', 2: 'same'}
    self.decision_model = None

    # Load the Stable Diffusion model.
    if load_sd:
      model_id = "runwayml/stable-diffusion-v1-5"
      self.sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    if decision_model_path is not None:
      print('Loading decision model...')
      self.decision_model = nn.Sequential(*[
          nn.Dropout(0.5),
          nn.Linear(4096, 2),
      ])
      mlp_checkpoint = torch.load(decision_model_path)
      self.decision_model.load_state_dict(mlp_checkpoint['state_dict'], strict=True)
      self.decision_model.eval()

  def __call__(self, images: Tensor, image_grid_ths: Tensor, input_ids: Optional[Tensor] = None, labels: Optional[Tensor] = None,
               caption_start_id: Optional[Tensor] = None, caption_end_id: Optional[Tensor] = None,
               generate: bool = False, num_words: int = 32, temperature: float = 1.0, top_p: float = 1.0,
               ret_scale_factor: float = 1.0, gen_scale_factor: float = 1.0,
               min_word_tokens: int = 0, mode: str = 'captioning') -> Tensor:
    if generate:
      return self.model.generate(images, num_words, temperature=temperature, top_p=top_p,
                                 min_word_tokens=min_word_tokens, ret_scale_factor=ret_scale_factor,
                                 gen_scale_factor=gen_scale_factor)
    else:
      output = self.model(
        pixel_values = images,
        image_grid_ths = image_grid_ths,
        input_ids = input_ids,
        labels = labels,
        caption_start_id = caption_start_id,
        caption_end_id = caption_end_id,
        mode = mode)
      return output

  def generate_for_images_and_texts(
    self, prompts: List, num_words: int = 0, min_word_tokens: int = 0, ret_scale_factor: float = 1.0, gen_scale_factor: float = 1.0,
    top_p: float = 1.0, temperature: float = 0.0, max_num_rets: int = 1, generator=None, 
    always_add_bos : bool = False, guidance_scale: float = 7.5, num_inference_steps: int = 50):
    """
    Encode prompts into embeddings, and generates text and image outputs accordingly.

    Args:
      prompts: List of interleaved PIL.Image.Image and strings representing input to the model.
      num_words: Maximum number of words to generate for. If num_words = 0, the model will run its forward pass and return the outputs.
      min_word_tokens: Minimum number of actual words before generating an image.
      ret_scale_factor: Proportion to scale [IMG] token logits by. A higher value may increase the probability of the model generating [IMG] outputs.
      top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
      temperature: Used to modulate logit distribution.
      max_num_rets: Maximum number of images to return in one generation pass.
    Returns:
      return_outputs: List consisting of either str or List[PIL.Image.Image] objects, representing image-text interleaved model outputs.
    """
    input_embs = []
    input_ids = []
    add_bos = True

    with torch.no_grad():
      for p in prompts:
        if type(p) == Image.Image:
          # Encode as image.
          pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, p)
          pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
          pixel_values = pixel_values[None, ...]

          visual_embs = self.model.get_visual_embs(pixel_values, mode='captioning')  # (1, n_visual_tokens, D)
          input_embs.append(visual_embs)
        elif type(p) == str:
          text_ids = self.model.tokenizer(p, add_special_tokens=add_bos, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
          # Only add <bos> once unless the flag is set.
          if not always_add_bos:
            add_bos = False

          text_embs = self.model.input_embeddings(text_ids)  # (1, T, D)
          input_embs.append(text_embs)
          input_ids.append(text_ids)
        else:
          raise ValueError(f'Input prompts should be either PIL.Image.Image or str types, got {type(p)} instead.')
      input_embs = torch.cat(input_embs, dim=1)
      input_ids = torch.cat(input_ids, dim=1)

      if num_words == 0:
        raise NotImplementedError('Generation not implemented for num_words=0.')
      elif num_words > 0:
        generated_ids, generated_embeddings, _ = self.model.generate(input_embs, num_words, min_word_tokens=min_word_tokens,
          temperature=temperature, top_p=top_p, ret_scale_factor=ret_scale_factor, gen_scale_factor=gen_scale_factor)
        embeddings = generated_embeddings[-1][:, input_embs.shape[1]:]

        # Truncate to newline.
        newline_token_id = self.model.tokenizer('\n', add_special_tokens=False).input_ids[0]
        trunc_idx = 0
        for j in range(generated_ids.shape[1]):
          if generated_ids[0, j] == newline_token_id:
            trunc_idx = j
            break
        if trunc_idx > 0:
          generated_ids = generated_ids[:, :trunc_idx]
          embeddings = embeddings[:, :trunc_idx]
      else:
        raise ValueError

      # Save outputs as an interleaved list.
      return_outputs = []
      # Find up to max_num_rets [IMG] tokens, and their corresponding scores.
      all_ret_idx = [i for i, x in enumerate(generated_ids[0, :] == self.model.retrieval_token_idx[0]) if x][:max_num_rets]
      seen_image_idx = []  # Avoid showing the same image multiple times.

      last_ret_idx = 0
      if len(all_ret_idx) == 0:
        # No [IMG] tokens.
        caption = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return_outputs.append(utils.truncate_caption(caption))
      else:
        for ret_idx in all_ret_idx:
          assert generated_ids[0, ret_idx:ret_idx+self.model.num_tokens].cpu().detach().numpy().tolist() == self.model.retrieval_token_idx, (generated_ids[0, ret_idx:ret_idx+self.model.num_tokens], self.model.retrieval_token_idx)
          raw_emb = embeddings[:, ret_idx:ret_idx+self.model.num_tokens, :]  # (1, 8, 4096)
          assert len(self.model.args.text_emb_layers) == 1

          image_outputs = {
            'gen': [],
            'ret': [],
            'decision': None,
          }

          if self.emb_matrix is not None:
            # Produce retrieval embedding.
            ret_emb = self.model.ret_text_hidden_fcs[0](raw_emb, None)[:, 0, :]  # (1, 256)
            ret_emb = ret_emb / ret_emb.norm(dim=-1, keepdim=True)
            ret_emb = ret_emb.type(self.emb_matrix.dtype)  # (1, 256)
            scores = self.emb_matrix @ ret_emb.T

            # Downweight seen images.
            for seen_idx in seen_image_idx:
              scores[seen_idx, :] -= 1000

            # Get the top 3 images for each image.
            _, top_image_idx = scores.squeeze().topk(3)
            for img_idx in top_image_idx:
              # Find the first image that does not error out.
              try:
                seen_image_idx.append(img_idx)
                img = utils.get_image_from_url(self.path_array[img_idx])
                image_outputs['ret'].append((img, 'ret', scores[img_idx].item()))
                if len(image_outputs) == max_num_rets:
                  break
              except (UnidentifiedImageError, ConnectionError, OSError):
                pass

            # Make decision with MLP.
            if self.decision_model is not None:
              decision_emb = raw_emb[:, 0, :]  # (1, 4096)
              assert decision_emb.shape[1] == 4096, decision_emb.shape
              decision_logits = self.decision_model(decision_emb)
              probs = decision_logits.softmax(dim=-1).cpu().float().numpy().tolist()
              image_outputs['decision'] = [self.idx2dec[decision_logits.argmax().item()]] + probs
          else:
            # If no embedding matrix is provided, generate instead.
            image_outputs['decision'] = ['gen', [0, 1]]

          # Produce generation embedding.
          gen_prefix = ''.join([f'[IMG{i}]' for i in range(self.model.args.num_tokens)])
          gen_prefx_ids = self.model.tokenizer(gen_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
          gen_prefix_embs = self.model.input_embeddings(gen_prefx_ids)  # (1, T, D)
          gen_emb = self.model.gen_text_hidden_fcs[0](raw_emb, gen_prefix_embs)  # (1, 77, 768)

          if gen_emb.shape[1] != 77:
            print(f"Padding {gen_emb.shape} with zeros")
            bs = gen_emb.shape[0]
            clip_emb = 768
            gen_emb = gen_emb.reshape(bs, -1, clip_emb)  # (bs, T, 768)
            seq_len = gen_emb.shape[1]
            gen_emb = torch.cat([gen_emb, torch.zeros((bs, 77 - seq_len, clip_emb), device=gen_emb.device, dtype=gen_emb.dtype)], dim=1)
            print('Padded to', gen_emb.shape)

          gen_emb = gen_emb.repeat(self.num_gen_images, 1, 1)  # (self.num_gen_images, 77, 768)

          # OPTIM(jykoh): Only generate if scores are low.
          if self.load_sd:
            # If num_gen_images > 8, split into multiple batches (for GPU memory reasons).
            gen_max_bs = 8
            gen_images = []
            for i in range(0, self.num_gen_images, gen_max_bs):
              gen_images.extend(
                self.sd_pipe(prompt_embeds=gen_emb[i:i+gen_max_bs], generator=generator,
                             guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images)

            all_gen_pixels = []
            for img in gen_images:
              pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, img.resize((224, 224)).convert('RGB'))
              pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
              all_gen_pixels.append(pixel_values)
            
            if self.emb_matrix is not None:
              all_gen_pixels = torch.stack(all_gen_pixels, dim=0)
              gen_visual_embs = self.model.get_visual_embs(all_gen_pixels, mode='retrieval')  # (1, D)
              gen_visual_embs = gen_visual_embs / gen_visual_embs.norm(dim=-1, keepdim=True)
              gen_visual_embs = gen_visual_embs.type(self.emb_matrix.dtype)
              gen_rank_scores = (gen_visual_embs @ ret_emb.T).squeeze()
              sorted_score_idx = torch.argsort(-gen_rank_scores)

              # Rank images by retrieval score.
              if self.num_gen_images > 1:
                image_outputs['gen'] = [(gen_images[idx], gen_rank_scores[idx].item()) for idx in sorted_score_idx]
              else:
                image_outputs['gen'] = [(gen_images[0], gen_rank_scores.item())]
            else:
              image_outputs['gen'] = [(gen_images[0], 0)]
          else:
            image_outputs['gen'] = [gen_emb]

          caption = self.model.tokenizer.batch_decode(generated_ids[:, last_ret_idx:ret_idx], skip_special_tokens=True)[0]
          last_ret_idx = ret_idx + 1
          return_outputs.append(utils.truncate_caption(caption) + f' {gen_prefix}')
          return_outputs.append(image_outputs)

    return return_outputs

  def get_log_likelihood_scores(
    self, prompts: List):
    """
    Output the log likelihood of the given interleaved prompts.

    Args:
      prompts: List of interleaved PIL.Image.Image and strings representing input to the model.
    Returns:
      Log likelihood score of the prompt sequence.
    """
    input_embs = []
    input_ids = []
    add_bos = True

    for p in prompts:
      if type(p) == Image.Image:
        # Encode as image.
        pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, p)
        pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
        pixel_values = pixel_values[None, ...]

        visual_embs = self.model.get_visual_embs(pixel_values, mode='captioning')  # (1, n_visual_tokens, D)
        input_embs.append(visual_embs)
        id_ = torch.zeros(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100
        input_ids.append(id_)
      elif type(p) == str:
        text_ids = self.model.tokenizer(p, add_special_tokens=True, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
        if not add_bos:
          # Remove <bos> tag.
          text_ids = text_ids[:, 1:]
        else:
          # Only add <bos> once.
          add_bos = False

        text_embs = self.model.input_embeddings(text_ids)  # (1, T, D)
        input_embs.append(text_embs)
        input_ids.append(text_ids)
      else:
        raise ValueError(f'Input prompts should be either PIL.Image.Image or str types, got {type(p)} instead.')
    input_embs = torch.cat(input_embs, dim=1)
    input_ids = torch.cat(input_ids, dim=1)

    outputs = self.model.lm(inputs_embeds=input_embs, labels=input_ids, use_cache=False, output_hidden_states=True)
    return -outputs.loss.item()  


def load_gill(model_dir: str, load_ret_embs: bool = True, decision_model_fn: str = 'decision_model.pth.tar') -> GILL:
  model_args_path = os.path.join(model_dir, 'model_args.json')
  model_ckpt_path = os.path.join(model_dir, 'pretrained_ckpt.pth.tar')
  embs_paths = [s for s in glob.glob(os.path.join(model_dir, 'cc3m*.npy'))]

  if not os.path.exists(model_args_path):
    raise ValueError(f'model_args.json does not exist in {model_dir}.')
  if not os.path.exists(model_ckpt_path):
    raise ValueError(f'pretrained_ckpt.pth.tar does not exist in {model_dir}.')
  if not load_ret_embs or len(embs_paths) == 0:
    if len(embs_paths) == 0:
      print(f'cc3m.npy files do not exist in {model_dir}.')
    print('Running the model without retrieval.')
    path_array, emb_matrix = None, None
  else:
    # Load embeddings.
    # Construct embedding matrix for nearest neighbor lookup.
    path_array = []
    emb_matrix = []

    # These were precomputed for all CC3M images with `model.get_visual_embs(image, mode='retrieval')`.
    for p in embs_paths:
      with open(p, 'rb') as wf:
          train_embs_data = pkl.load(wf)
          path_array.extend(train_embs_data['paths'])
          emb_matrix.extend(train_embs_data['embeddings'])
    emb_matrix = np.stack(emb_matrix, axis=0)

    # Number of paths should be equal to number of embeddings.
    assert len(path_array) == emb_matrix.shape[0], (len(path_array), emb_matrix.shape)

  with open(model_args_path, 'r') as f:
    model_kwargs = json.load(f)

  # Initialize tokenizer.
  tokenizer = AutoTokenizer.from_pretrained(model_kwargs['opt_version'], use_fast=False)
  if tokenizer.pad_token is None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
  # Add an image token for loss masking (and visualization) purposes.
  tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer

  # Add [IMG] tokens to the vocabulary.
  model_kwargs['retrieval_token_idx'] = []
  for i in range(model_kwargs['num_tokens']):
      print(f'Adding [IMG{i}] token to vocabulary.')
      print(f'Before adding new token, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
      num_added_tokens = tokenizer.add_tokens(f'[IMG{i}]')
      print(f'After adding {num_added_tokens} new tokens, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
      ret_token_idx = tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
      assert len(ret_token_idx) == 1, ret_token_idx
      model_kwargs['retrieval_token_idx'].append(ret_token_idx[0])
  # Use the same RET tokens for generation.
  model_kwargs['gen_token_idx'] = model_kwargs['retrieval_token_idx']

  args = namedtuple('args', model_kwargs)(**model_kwargs)

  # Load decision model.
  if decision_model_fn is not None:
    decision_model_path = os.path.join(model_dir, decision_model_fn)
  else:
    decision_model_path = None

  # Initialize model for inference.
  model = GILL(tokenizer, args, path_array=path_array, emb_matrix=emb_matrix,
               load_sd=True, num_gen_images=1, decision_model_path=decision_model_path)
  model = model.eval()
  model = model.bfloat16()
  model = model.cuda()

  # Load pretrained linear mappings and [IMG] embeddings.
  checkpoint = torch.load(model_ckpt_path)
  state_dict = {}
  # This is needed if we train with DDP.
  for k, v in checkpoint['state_dict'].items():
      state_dict[k.replace('module.', '')] = v
  img_token_embeddings = state_dict['model.input_embeddings.weight'].cpu().detach()
  del state_dict['model.input_embeddings.weight']

  model.load_state_dict(state_dict, strict=False)
  # Copy over the embeddings of the [IMG] tokens (while loading the others from the pretrained LLM).
  with torch.no_grad():
      if 'share_ret_gen' in model_kwargs:
        assert model_kwargs['share_ret_gen'], 'Model loading only supports share_ret_gen=True for now.'
      model.model.input_embeddings.weight[-model_kwargs['num_tokens']:, :].copy_(img_token_embeddings)

  if load_ret_embs and len(embs_paths) > 0:
    logit_scale = model.model.logit_scale.exp()
    emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
    emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
    emb_matrix = logit_scale * emb_matrix
    model.emb_matrix = emb_matrix

  return model

