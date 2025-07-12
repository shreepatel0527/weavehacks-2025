#ESM runs a local model, I reduced the model size so it runs in <30 sec and outputs a residue heatmap
#pip install fair-esm torch

import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_esm_inference(sequence):
    # Load smallest ESM2 model (~8M parameters)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_representations = results["representations"][6]
    embeddings = token_representations.squeeze(0).cpu().numpy()  # shape: (L+2, 320)
    
    return embeddings

def plot_heatmap(embeddings, sequence, output_image):
    residue_embeddings = embeddings[1:-1]  # Remove CLS and EOS

    plt.figure(figsize=(12, 6))
    sns.heatmap(residue_embeddings, cmap="coolwarm", center=0, cbar_kws={"label": "Embedding Value"})
    plt.title("ESM2 Residue Embedding Heatmap")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Residue Index")
    plt.yticks(ticks=np.arange(len(sequence)) + 0.5, labels=list(sequence), rotation=0)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved heatmap to {output_image}")
    plt.close()

def export_embeddings_to_csv(embeddings, sequence, output_csv):
    residue_embeddings = embeddings[1:-1]  # Remove CLS and EOS
    df = pd.DataFrame(residue_embeddings)
    df.insert(0, "Residue", list(sequence))
    df.to_csv(output_csv, index=False)
    print(f"Saved embeddings to {output_csv}")

if __name__ == "__main__":
    sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETQ"
    embeddings = run_esm_inference(sequence)
    
    # Export heatmap
    plot_heatmap(embeddings, sequence, "esm2_embedding_heatmap.png")
    
    # Export CSV
    export_embeddings_to_csv(embeddings, sequence, "esm2_embeddings.csv")
