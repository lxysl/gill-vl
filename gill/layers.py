import torch
from torch import nn


class TextFcLayer(nn.Module):
  """Layers used in mapping text embeddings to visual outputs."""

  def __init__(self, in_dim: int, out_dim: int, out_dim2: int = None, num_input_tokens: int = 1, num_output_tokens: int = 1, mode: str = 'linear'):
    super().__init__()

    self.num_input_tokens = num_input_tokens
    self.num_output_tokens = num_output_tokens
    self.mode = mode

    if mode == 'linear':
      self.model = nn.Linear(in_dim, out_dim)
    elif mode == 'gill_mapper':
      assert num_output_tokens == 77, f"num_output_tokens should be 77 (CLIP), got {num_output_tokens} instead."
      hidden_dim = 512
      self.fc = nn.Linear(in_dim, hidden_dim)
      self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
      self.model = nn.Linear(hidden_dim, out_dim)
      self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
    elif mode == 'gill_mapper_hunyuan':
      assert num_output_tokens == 77 + 256, f"num_output_tokens should be 77 (CLIP) + 256 (T5), got {num_output_tokens} instead."
      hidden_dim = 768
      self.fc = nn.Linear(in_dim, hidden_dim)
      self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
      self.model = nn.Linear(hidden_dim, out_dim)
      self.model2 = nn.Linear(hidden_dim, out_dim2)
      self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
      self.tgt_mask = torch.block_diag(torch.ones(77, 77), torch.ones(256, 256)).bool()
    elif mode == 'gill_mapper_kolors':
      assert num_output_tokens == 256 + 1, f"num_output_tokens should be 256 (ChatGLM) + 1 (Pooled), got {num_output_tokens} instead."
      hidden_dim = 1024
      self.fc = nn.Linear(in_dim, hidden_dim)
      self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
      self.model = nn.Linear(hidden_dim, out_dim)
      self.model2 = nn.Linear(hidden_dim, out_dim2)
      self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
      self.tgt_mask = torch.block_diag(torch.ones(256, 256), torch.ones(1, 1)).bool()
    else:
      raise NotImplementedError(mode)

  def forward(self, x: torch.Tensor, input_embs: torch.Tensor) -> torch.Tensor:
    outputs = None
    
    if self.mode in ['gill_mapper', 'gill_mapper_hunyuan', 'gill_mapper_kolors']:
      x = x + input_embs

    if isinstance(self.model, nn.ModuleList):
      assert len(self.model) == x.shape[1] == self.num_input_tokens, (len(self.model), x.shape, self.num_input_tokens)
      outputs = []
      for i in range(self.num_input_tokens):
        outputs.append(self.model[i](x[:, i, :]))  # (N, D)
      outputs = torch.stack(outputs, dim=1)  # (N, T, D)
    else:
      if self.mode == 'gill_mapper':
        x = self.fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        outputs = self.model(x)
      elif self.mode == 'gill_mapper_hunyuan':
        x = self.fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1), tgt_mask=self.tgt_mask.to(x.device))
        outputs = self.model(x[:, :77, :])
        outputs2 = self.model2(x[:, 77:, :])
      elif self.mode == 'gill_mapper_kolors':
        x = self.fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        outputs = self.model(x[:, :256, :])
        outputs2 = self.model2(x[:, 256:, :])
      else:
        outputs = self.model(x)

      if outputs.shape[1] != self.num_output_tokens and self.mode == 'linear':
        if self.mode == 'linear':
          outputs = outputs[:, :self.num_output_tokens, :]
        else:
          raise NotImplementedError

    if self.mode == 'gill_mapper_hunyuan':
      assert (outputs.shape[1] * outputs.shape[2] == 77 * 1024) or (outputs2.shape[1] * outputs2.shape[2] == 256 * 2048), (outputs.shape, outputs2.shape)
      return outputs, outputs2  # (N, T, D), (N, T2, D2)
    elif self.mode == 'gill_mapper_kolors':
      assert (outputs.shape[1] * outputs.shape[2] == 256 * 4096) or (outputs2.shape[1] * outputs2.shape[2] == 1 * 4096), (outputs.shape, outputs2.shape)
      return outputs, outputs2  # (N, T, D), (N, T2, D2)
    else:
      assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * 768), (outputs.shape, self.num_output_tokens)
      return outputs  # (N, T, D)
