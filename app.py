import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START = "<start>"
END = "<end>"
PAD = "<pad>"
UNK = "<unk>"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("best_model.pth", map_location=device)

    vocab_size = checkpoint["vocab_size"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    # ----- Architecture -----
    class Encoder(nn.Module):
        def __init__(self, input_dim=2048, hidden_size=512):
            super().__init__()
            self.fc = nn.Linear(input_dim, hidden_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.fc(x))

    class Decoder(nn.Module):
        def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
            super().__init__()

            self.embed = nn.Embedding(
                vocab_size,
                embed_dim,
                padding_idx=stoi[PAD]
            )

            self.dropout = nn.Dropout(0.5)

            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_size,
                batch_first=True
            )

            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, captions, hidden):
            emb = self.dropout(self.embed(captions))
            outputs, hidden = self.lstm(emb, hidden)
            outputs = self.dropout(outputs)
            logits = self.fc(outputs)
            return logits, hidden

    class ImageCaptionModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder(vocab_size)

        def forward(self, feats, captions):
            h0 = self.encoder(feats).unsqueeze(0)
            c0 = torch.zeros_like(h0)
            outputs, _ = self.decoder(captions, (h0, c0))
            return outputs

    model = ImageCaptionModel(vocab_size)

    # ----- Fix DataParallel "module." prefix if present -----
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()

    return model, stoi, itos


model, stoi, itos = load_model()

# --------------------------------------------------
# LOAD RESNET FEATURE EXTRACTOR
# --------------------------------------------------
@st.cache_resource
def load_resnet():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()
    return resnet


resnet = load_resnet()

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# --------------------------------------------------
# GREEDY DECODING
# --------------------------------------------------
def greedy_caption(feat, max_len=30):
    feat = feat.unsqueeze(0).to(device)

    h = model.encoder(feat).unsqueeze(0)
    c = torch.zeros_like(h)

    word = torch.tensor([[stoi[START]]]).to(device)
    caption = []

    for _ in range(max_len):
        out, (h, c) = model.decoder(word, (h, c))
        pred = out.argmax(-1)

        word = pred
        token = itos[pred.item()]

        if token == END:
            break

        caption.append(token)

    return " ".join(caption)


# --------------------------------------------------
# BEAM SEARCH
# --------------------------------------------------
def beam_search(feat, beam_width=3, max_len=30):
    feat = feat.unsqueeze(0).to(device)

    h = model.encoder(feat).unsqueeze(0)
    c = torch.zeros_like(h)

    sequences = [(0.0, [stoi[START]], h, c)]

    for _ in range(max_len):
        all_candidates = []

        for score, seq, h, c in sequences:

            if seq[-1] == stoi[END]:
                all_candidates.append((score, seq, h, c))
                continue

            word = torch.tensor([[seq[-1]]]).to(device)
            out, (h_new, c_new) = model.decoder(word, (h, c))

            log_probs = torch.log_softmax(out.squeeze(1), dim=-1)
            topk = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                token = topk.indices[0][i].item()

                # length normalization
                new_score = (score - topk.values[0][i].item()) / len(seq)

                all_candidates.append(
                    (new_score, seq + [token], h_new, c_new)
                )

        sequences = sorted(all_candidates, key=lambda x: x[0])[:beam_width]

    best_seq = sequences[0][1]

    return " ".join([
        itos[i] for i in best_seq
        if i not in [stoi[START], stoi[END], stoi[PAD]]
    ])


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("🖼️ Image Caption Generator")

decode_method = st.radio(
    "Choose decoding method:",
    ["Greedy Search", "Beam Search"]
)

beam_width = 3
if decode_method == "Beam Search":
    beam_width = st.slider(
        "Beam Width",
        min_value=2,
        max_value=7,
        value=3
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = resnet(img_tensor).view(1, -1)

    feat = feat.squeeze(0)

    with st.spinner("Generating caption..."):
        if decode_method == "Greedy Search":
            caption = greedy_caption(feat)
        else:
            caption = beam_search(feat, beam_width=beam_width)

    st.subheader("Generated Caption:")
    st.write(caption)
