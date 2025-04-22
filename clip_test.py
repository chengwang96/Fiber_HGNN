import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["15 to 20", "15-20", "15~20"]).to(device)
import ipdb; ipdb.set_trace()

with torch.no_grad():
    text_features = model.encode_text(text)
    import ipdb; ipdb.set_trace()