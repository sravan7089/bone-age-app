import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import gdown
import shap

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Bone Age Predictor", layout="wide")

# ---------------------------
# Dataset class
# ---------------------------
class BoneAgeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        gender = np.array([row["male"]], dtype=np.float32)
        boneage = np.array([row["boneage"]], dtype=np.float32)
        return image, gender, boneage

# ---------------------------
# Model class
# ---------------------------
class BoneAgeModel(nn.Module):
    def __init__(self):
        super(BoneAgeModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        self.meta_fc = nn.Linear(1, 32)
        self.fc = nn.Linear(128 + 32, 1)

    def forward(self, x_img, x_meta):
        x1 = self.cnn(x_img)
        x2 = torch.relu(self.meta_fc(x_meta))
        x = torch.cat([x1, x2], dim=1)
        out = self.fc(x)
        return out

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Train", "Inference"])

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "folder_path" not in st.session_state:
    st.session_state.folder_path = None

# ---------------------------
# Page: Dataset Overview
# ---------------------------
if page == "Overview":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.png", width=200)

    st.title("📊 Dataset Overview")

    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    folder_path = st.text_input("Enter images folder path")

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.session_state.df = df
        st.session_state.folder_path = folder_path if folder_path else None

        st.write("Preview of dataset:")
        st.dataframe(df.head())

        if folder_path and os.path.isdir(folder_path):
            sample_files = os.listdir(folder_path)[:5]
            st.write("Sample Images:")
            for f in sample_files:
                try:
                    st.image(os.path.join(folder_path, f), width=100)
                except:
                    pass

# ---------------------------
# Page: Train
# ---------------------------
elif page == "Train":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.png", width=200)

    st.title("🧑‍🏫 Train Model")

    csv_file = st.file_uploader("Upload dataset CSV", type=["csv"])
    img_folder = st.text_input("Enter image folder path")

    if csv_file and img_folder:
        df = pd.read_csv(csv_file)
        df = df.sample(1000, random_state=42).reset_index(drop=True)
        st.write(f"Using {len(df)} samples for training (subset).")

        if st.button("Start Training"):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            dataset = BoneAgeDataset(df, img_folder, transform)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

            base_model = models.resnet18(pretrained=True)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()

            class CombinedModel(nn.Module):
                def __init__(self, base_model, in_features):
                    super().__init__()
                    self.cnn = base_model
                    self.fc1 = nn.Linear(in_features + 1, 128)
                    self.fc2 = nn.Linear(128, 1)

                def forward(self, x_img, x_meta):
                    x_img = self.cnn(x_img)
                    x = torch.cat([x_img, x_meta], dim=1)
                    x = torch.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x

            model = CombinedModel(base_model, in_features)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            num_epochs = 2
            progress = st.progress(0)
            status = st.empty()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, (imgs, meta, targets) in enumerate(train_loader):
                    imgs, meta, targets = imgs.to(device), meta.to(device), targets.to(device).unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(imgs, meta)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    batch_progress = (i + 1) / len(train_loader)
                    total_progress = (epoch + batch_progress) / num_epochs
                    progress.progress(total_progress)
                    status.text(f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

                st.write(f"Epoch {epoch+1} finished | Avg Loss: {running_loss/len(train_loader):.4f}")

            torch.save(model.state_dict(), "boneage_model.pth")
            st.success("✅ Training complete. Model saved as boneage_model.pth")

# ---------------------------
# Page: Inference
# ---------------------------
elif page == "Inference":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.png", width=200)

    st.title("🦴 Bone Age Prediction with Explainability")
    st.markdown("AI-powered skeletal maturity assessment using X-rays, gender, and age.")

    uploaded_img = st.file_uploader("📤 Upload X-ray Image", type=["png", "jpg", "jpeg"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    chrono_age = st.number_input("Chronological Age (months)", min_value=0, max_value=240, value=120)

    MODEL_PATH = "boneage_model.pth"
    FILE_ID = "1oD9hWWnwtFD8Qd4UwPMGruTyKYYZuBiK"
    URL = f"https://drive.google.com/uc?id={FILE_ID}"

    def ensure_model():
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Fetching model weights..."):
                try:
                    gdown.download(URL, MODEL_PATH, quiet=False)
                    st.success("✅ Model downloaded from Google Drive")
                except Exception as e:
                    st.error(f"❌ Could not download model: {e}")

    ensure_model()

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        meta_tensor = torch.tensor([[1 if gender == "Male" else 0, chrono_age]], dtype=torch.float32)

        base_model = models.resnet18(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Identity()

        class CombinedModel(nn.Module):
            def __init__(self, base_model, in_features):
                super().__init__()
                self.cnn = base_model
                self.fc1 = nn.Linear(in_features + 2, 128)
                self.fc2 = nn.Linear(128, 1)

            def forward(self, x_img, x_meta):
                x_img = self.cnn(x_img)
                x = torch.cat([x_img, x_meta], dim=1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = CombinedModel(base_model, in_features)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

        with torch.no_grad():
            prediction = model(img_tensor, meta_tensor).item()

        col1, col2 = st.columns([1,1])
        with col1:
            st.image(image, caption="Uploaded X-ray")
        with col2:
            st.markdown("### Prediction Results")
            st.metric("Chronological Age", f"{chrono_age} months")
            st.metric("Predicted Bone Age", f"{prediction:.1f} months",
                      delta=f"{prediction - chrono_age:.1f} months")

        tab1, tab2, tab3 = st.tabs(["Grad-CAM", "SHAP Metadata", "Age Comparison"])

        # ---------------------------
        # Tab 1: Grad-CAM
        # ---------------------------
        with tab1:
            st.subheader("🔍 Grad-CAM Heatmap")

            def generate_gradcam(model, img_tensor, meta_tensor, target_layer):
                model.eval()
                gradients, activations = [], []

                def backward_hook(module, grad_input, grad_output):
                    gradients.append(grad_output[0])

                def forward_hook(module, input, output):
                    activations.append(output)

                handle_fw = target_layer.register_forward_hook(forward_hook)
                handle_bw = target_layer.register_backward_hook(backward_hook)

                output = model(img_tensor.requires_grad_(), meta_tensor)
                output = output.squeeze()
                output.backward(retain_graph=True)

                grads = gradients[0].detach().cpu().numpy()[0]
                acts = activations[0].detach().cpu().numpy()[0]

                weights = np.mean(grads, axis=(1, 2))
                cam = np.zeros(acts.shape[1:], dtype=np.float32)
                for w, act in zip(weights, acts):
                    cam += w * act

                cam = np.maximum(cam, 0)
                cam = cam / cam.max()
                handle_fw.remove()
                handle_bw.remove()
                return cam

            target_layer = model.cnn.layer4[1].conv2
            cam = generate_gradcam(model, img_tensor, meta_tensor, target_layer)

            img_cv = np.array(image.resize((224, 224)))
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            superimposed = np.uint8(0.5 * heatmap + 0.5 * img_cv)

            if superimposed.ndim == 2:
                superimposed = cv2.cvtColor(superimposed, cv2.COLOR_GRAY2RGB)
            elif superimposed.shape[2] == 4:
                superimposed = cv2.cvtColor(superimposed, cv2.COLOR_RGBA2RGB)

            superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
            st.image(Image.fromarray(superimposed), caption="Grad-CAM Heatmap")

        # ---------------------------
        # Tab 2: SHAP Metadata
        # ---------------------------
        with tab2:
            st.subheader("📈 SHAP Metadata Explanation")

            def metadata_predict(x_meta_np):
                x_meta_tensor = torch.tensor(x_meta_np, dtype=torch.float32)
                with torch.no_grad():
                    preds = model(img_tensor.repeat(x_meta_tensor.size(0),1,1,1), x_meta_tensor)
                return preds.numpy()

            explainer = shap.Explainer(metadata_predict, np.array([[1 if gender == "Male" else 0, chrono_age]]))
            shap_values = explainer(np.array([[1 if gender == "Male" else 0, chrono_age]]))

            feature_names = ["Gender(Male=1)", "Chronological Age (months)"]
            shap_values.feature_names = feature_names

            # Robust bar plot
            shap_vals = shap_values.values[0]
            fig, ax = plt.subplots()
            ax.barh(feature_names, shap_vals, color="skyblue")
            for i, v in enumerate(shap_vals):
                ax.text(v + 0.01*np.sign(v), i, f"{v:.2f}", va='center')
            ax.set_xlabel("SHAP Value")
            ax.set_title("SHAP Metadata Contribution")
            st.pyplot(fig)

        # ---------------------------
        # Tab 3: Age Comparison
        # ---------------------------
        with tab3:
            st.subheader("📊 Chronological vs Predicted Age")
            ages = ["Chronological Age", "Predicted Bone Age"]
            values = [chrono_age, prediction]

            fig, ax = plt.subplots(figsize=(5,3))
            bars = ax.bar(ages, values, color=["skyblue", "salmon"])
            ax.set_ylabel("Age (months)")
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha="center", va="bottom")
            st.pyplot(fig)

            delta = prediction - chrono_age
            if delta > 0:
                st.info(f"🟢 Predicted bone age is **advanced** by {delta:.1f} months compared to chronological age.")
            elif delta < 0:
                st.warning(f"🔴 Predicted bone age is **delayed** by {-delta:.1f} months compared to chronological age.")
            else:
                st.success("✅ Predicted bone age matches chronological age.")
