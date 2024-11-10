"""This script extracts text embeddings from the text encoder of Stable Diffusion
for a given dataset of captions, and saves them to disk.
The outputs are used in training GILL.

Example usage:
python scripts/preprocess_kolors_embeddings.py  datasets/cc3m_val.tsv data/cc3m/validation/kolors_embs data/cc3m/validation/kolors_pooled_embs
"""

import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import torch

# Load a slightly modified version of the Stable Diffusion pipeline.
# This allows us to extract text embeddings directly (without generating images).
from gill.custom_kolors import KolorsPipeline


# Default arguments for running preprocessing.
model_id = "Kwai-Kolors/Kolors-diffusers"
batch_size = 32
input_captions_fp = sys.argv[1]  # tab separated file of captions and image ids
emb_output_dir = sys.argv[2]  # output directory to save ChatGLM3-6B ext embeddings in
pooled_emb_output_dir = sys.argv[3]  # output directory to save ChatGLM3-6B pooled text embeddings in
os.makedirs(emb_output_dir, exist_ok=True)
os.makedirs(pooled_emb_output_dir, exist_ok=True)


def save_to_path(emb, path):
    """Save embeddings to disk."""
    try:
        with open(path, 'wb') as wf:
            np.save(wf, emb)
    except:
        print("Error with", path)
    return path


if __name__ == '__main__':
    pipe = KolorsPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    if not torch.cuda.is_available():
        print('WARNING: using CPU, this will be slow!')
    else:
        pipe = pipe.to("cuda")

    # Get existing files, so that we don't recompute them.
    existing_files = set([f.strip('.npy') for f in os.listdir(emb_output_dir)])
    existing_files_2 = set([f.strip('.npy') for f in os.listdir(pooled_emb_output_dir)])

    # Load captions and associated image ids.
    with open(input_captions_fp, 'r') as f:
        data = f.readlines()
        examples = data[1:]
        captions = []
        image_ids = []

        for x in examples:
            d = x.strip().split('\t')
            if d[1] not in existing_files or d[1] not in existing_files_2:
                captions.append(d[0])
                image_ids.append(d[1])
        assert len(captions) == len(image_ids)

    # Extract embeddings in batches.
    num_batches = int(np.ceil(len(captions) / batch_size))
    for i in tqdm(range(num_batches)):
        # if i == 100:
        #     break

        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_captions = captions[start_idx:end_idx]
        batch_ids = image_ids[start_idx:end_idx]
        prompt_embeds_1, prompt_embeds_2 = pipe(batch_captions, return_prompts_only=True)
        prompt_embeds_1 = prompt_embeds_1.detach().cpu().numpy()
        prompt_embeds_2 = prompt_embeds_2.detach().cpu().numpy()
        if i == 0:
            # (bs, 256, 4096), (bs, 4096)
            print(f"prompt_embeds_1.shape: {prompt_embeds_1.shape}, prompt_embeds_2.shape: {prompt_embeds_2.shape}")

        # Save embeddings to disk in parallel.
        Parallel(n_jobs=8)(delayed(save_to_path)(
            prompt_embeds_1[j, :, ...], os.path.join(emb_output_dir, f'{batch_ids[j]}.npy')
        ) for j in range(prompt_embeds_1.shape[0]))
        Parallel(n_jobs=8)(delayed(save_to_path)(
            prompt_embeds_2[j, :, ...], os.path.join(pooled_emb_output_dir, f'{batch_ids[j]}.npy')
        ) for j in range(prompt_embeds_2.shape[0]))

    # caption = "actor attends the premiere of film."
    # prompt_embeds_1, prompt_embeds_2 = pipe(caption, return_prompts_only=True)
