




##################################################################################################
# app.py ##################################### WORKING  CODE
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
from sklearn.model_selection import train_test_split

# ---------------------------
# Custom Dataset
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
        img_path = os.path.join(self.img_dir, f"{row['id']}.png")  # assumes filenames like id.png
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        gender = np.array([row["male"]], dtype=np.float32)   # clinical metadata
        boneage = np.array([row["boneage"]], dtype=np.float32)  # target

        return image, gender, boneage


# ---------------------------
# Model: ResNet18 + metadata
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
# Streamlit App
# ---------------------------
#st.image("logo.png", width=300)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Train", "Inference"])
# page = st.sidebar.radio("Go to", ["Home", "Train", "Inference", "Evaluation"])


# Keep dataset + folder path across pages
if "df" not in st.session_state:
    st.session_state.df = None
if "folder_path" not in st.session_state:
    st.session_state.folder_path = None

# ---------------------------
# Page 1: Dataset Overview
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
# Page 2: Training
# ---------------------------

elif page == "Train":
    # Centered logo at top
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.png", width=200)

    st.title("🧑‍🏫 Train Model")

    csv_file = st.file_uploader("Upload dataset CSV", type=["csv"])
    img_folder = st.text_input("Enter image folder path")

    if csv_file and img_folder:
        df = pd.read_csv(csv_file)

        # Subsample for quick testing (⚡ you can remove later for full training)
        df = df.sample(1000, random_state=42).reset_index(drop=True)

        st.write(f"Using {len(df)} samples for training (subset).")

        if st.button("Start Training"):
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            from torchvision import models, transforms
            from PIL import Image
            import os

            class BoneAgeDataset(Dataset):
                def __init__(self, dataframe, img_dir, transform=None):
                    self.df = dataframe
                    self.img_dir = img_dir
                    self.transform = transform

                def __len__(self):
                    return len(self.df)

                def __getitem__(self, idx):
                    row = self.df.iloc[idx]
                    img_path = os.path.join(self.img_dir, str(row["id"]) + ".png")
                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    meta = torch.tensor([row["male"]], dtype=torch.float32)
                    target = torch.tensor(row["boneage"], dtype=torch.float32)
                    return image, meta, target

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            dataset = BoneAgeDataset(df, img_folder, transform)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

            # Model
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

            num_epochs = 2  # quick run
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

                    # Update UI
                    batch_progress = (i + 1) / len(train_loader)
                    total_progress = (epoch + batch_progress) / num_epochs
                    progress.progress(total_progress)
                    status.text(f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

                st.write(f"Epoch {epoch+1} finished | Avg Loss: {running_loss/len(train_loader):.4f}")

            torch.save(model.state_dict(), "boneage_model.pth")
            st.success("✅ Training complete. Model saved as boneage_model.pth")


# ---------------------------
# Page 3: Inference (placeholder)
# ---------------------------



# elif page == "Inference":
#     st.title("🧪 Bone Age Prediction with Explainability")

#     uploaded_img = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
#     gender = st.selectbox("Gender", ["Male", "Female"])

#     if uploaded_img:
#         import torch
#         import torch.nn as nn
#         import numpy as np
#         import shap
#         import matplotlib.pyplot as plt
#         from torchvision import models, transforms
#         from PIL import Image

#         # Preprocess image
#         image = Image.open(uploaded_img).convert("RGB")
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#         img_tensor = transform(image).unsqueeze(0)

#         # Metadata (gender only)
#         meta_tensor = torch.tensor([[1 if gender == "Male" else 0]], dtype=torch.float32)

#         # Load trained model
#         base_model = models.resnet18(pretrained=False)
#         in_features = base_model.fc.in_features
#         base_model.fc = nn.Identity()

#         class CombinedModel(nn.Module):
#             def __init__(self, base_model, in_features):
#                 super().__init__()
#                 self.cnn = base_model
#                 self.fc1 = nn.Linear(in_features + 1, 128)  # gender only
#                 self.fc2 = nn.Linear(128, 1)

#             def forward(self, x_img, x_meta):
#                 x_img = self.cnn(x_img)
#                 x = torch.cat([x_img, x_meta], dim=1)
#                 x = torch.relu(self.fc1(x))
#                 x = self.fc2(x)
#                 return x

#         model = CombinedModel(base_model, in_features)
#         model.load_state_dict(torch.load("boneage_model.pth", map_location="cpu"))
#         model.eval()

#         # Prediction
#         with torch.no_grad():
#             prediction = model(img_tensor, meta_tensor).item()

#         st.image(image, caption="Uploaded X-ray", use_container_width=True)
#         st.success(f"📊 Predicted Bone Age: **{prediction:.1f} months**")

#         # ---- Grad-CAM ----
#         st.subheader("🔍 Grad-CAM Heatmap (Image Explainability)")

#         def generate_gradcam(model, img_tensor, meta_tensor, target_layer):
#             model.eval()
#             gradients = []
#             activations = []

#             def backward_hook(module, grad_input, grad_output):
#                 gradients.append(grad_output[0])

#             def forward_hook(module, input, output):
#                 activations.append(output)

#             handle_fw = target_layer.register_forward_hook(forward_hook)
#             handle_bw = target_layer.register_backward_hook(backward_hook)

#             output = model(img_tensor, meta_tensor)
#             output.backward()

#             grads = gradients[0].detach().cpu().numpy()[0]
#             acts = activations[0].detach().cpu().numpy()[0]

#             weights = np.mean(grads, axis=(1, 2))
#             cam = np.zeros(acts.shape[1:], dtype=np.float32)
#             for w, act in zip(weights, acts):
#                 cam += w * act

#             cam = np.maximum(cam, 0)
#             cam = cam / cam.max()
#             handle_fw.remove()
#             handle_bw.remove()
#             return cam

#         target_layer = model.cnn.layer4[1].conv2
       
#         cam = generate_gradcam(model, img_tensor.requires_grad_(), meta_tensor, target_layer)

#         import cv2
#         img_cv = np.array(image.resize((224, 224)))

#         # 🔧 Resize CAM to image size
#         cam_resized = cv2.resize(cam, (224, 224))

#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#         # Overlay heatmap on image
#         superimposed = np.uint8(0.5 * heatmap + 0.5 * img_cv)

#         st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)


#         # ---- SHAP ----
#         st.subheader("📈 SHAP Values (Metadata Explainability)")

#         explainer = shap.Explainer(
#             lambda x: model(img_tensor, torch.tensor(x, dtype=torch.float32)).detach().numpy(),
#             meta_tensor.detach().numpy()
#         )

#         shap_values = explainer(meta_tensor.detach().numpy())

#         fig, ax = plt.subplots()
#         shap.plots.waterfall(shap_values[0], show=False)
#         st.pyplot(fig)
                  ##############working code
# elif page == "Inference":
#     st.title("🧪 Bone Age Prediction with Explainability")

#     uploaded_img = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     chrono_age = st.number_input("Chronological Age (months)", min_value=0, max_value=240, value=120)

#     if uploaded_img:
#         import torch
#         import torch.nn as nn
#         import numpy as np
#         import shap
#         import matplotlib.pyplot as plt
#         from torchvision import models, transforms
#         from PIL import Image
#         import cv2

#         # ---- Preprocess image ----
#         image = Image.open(uploaded_img).convert("RGB")
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#         img_tensor = transform(image).unsqueeze(0)

#         # ---- Metadata (gender + chrono age) ----
#         meta_tensor = torch.tensor([[1 if gender == "Male" else 0, chrono_age]], dtype=torch.float32)

#         # ---- Load pretrained model ----
#         base_model = models.resnet18(weights=None)
#         in_features = base_model.fc.in_features
#         base_model.fc = nn.Identity()

#         class CombinedModel(nn.Module):
#             def __init__(self, base_model, in_features):
#                 super().__init__()
#                 self.cnn = base_model
#                 self.fc1 = nn.Linear(in_features + 2, 128)  # gender + age
#                 self.fc2 = nn.Linear(128, 1)

#             def forward(self, x_img, x_meta):
#                 x_img = self.cnn(x_img)
#                 x = torch.cat([x_img, x_meta], dim=1)
#                 x = torch.relu(self.fc1(x))
#                 x = self.fc2(x)
#                 return x

#         model = CombinedModel(base_model, in_features)
#         #model.load_state_dict(torch.load("/content/drive/MyDrive/boneage_model_quick.pth", map_location="cpu"))
#         model.load_state_dict(torch.load("boneage_model_quick.pth", map_location="cpu"))

#         model.eval()

#         # ---- Prediction ----
#         with torch.no_grad():
#             prediction = model(img_tensor, meta_tensor).item()

#         st.image(image, caption="Uploaded X-ray", use_container_width=True)
#         st.success(f"📊 Predicted Bone Age: **{prediction:.1f} months**")

#         # ---- Grad-CAM ----
#         st.subheader("🔍 Grad-CAM Heatmap (Image Explainability)")

#         def generate_gradcam(model, img_tensor, meta_tensor, target_layer):
#             model.eval()
#             gradients = []
#             activations = []

#             def backward_hook(module, grad_input, grad_output):
#                 gradients.append(grad_output[0])

#             def forward_hook(module, input, output):
#                 activations.append(output)

#             handle_fw = target_layer.register_forward_hook(forward_hook)
#             handle_bw = target_layer.register_backward_hook(backward_hook)

#             output = model(img_tensor.requires_grad_(), meta_tensor)
#             output.backward()

#             grads = gradients[0].detach().cpu().numpy()[0]
#             acts = activations[0].detach().cpu().numpy()[0]

#             weights = np.mean(grads, axis=(1, 2))
#             cam = np.zeros(acts.shape[1:], dtype=np.float32)
#             for w, act in zip(weights, acts):
#                 cam += w * act

#             cam = np.maximum(cam, 0)
#             cam = cam / cam.max()
#             handle_fw.remove()
#             handle_bw.remove()
#             return cam

#         target_layer = model.cnn.layer4[1].conv2
#         cam = generate_gradcam(model, img_tensor, meta_tensor, target_layer)

#         img_cv = np.array(image.resize((224, 224)))
#         cam_resized = cv2.resize(cam, (224, 224))
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#         superimposed = np.uint8(0.5 * heatmap + 0.5 * img_cv)
#         st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

#         # ---- SHAP for metadata ----
#         st.subheader("📈 SHAP Values (Metadata Explainability)")

#         # Small wrapper for SHAP
#         def metadata_predict(x_meta_np):
#             x_meta_tensor = torch.tensor(x_meta_np, dtype=torch.float32)
#             with torch.no_grad():
#                 preds = model(img_tensor.repeat(x_meta_tensor.size(0),1,1,1), x_meta_tensor)
#             return preds.numpy()

#         explainer = shap.Explainer(metadata_predict, np.array([[1 if gender == "Male" else 0, chrono_age]]))
#         shap_values = explainer(np.array([[1 if gender == "Male" else 0, chrono_age]]))

#         # Plot SHAP values
#         # feature_names = ["Gender(Male=1)", "Chronological Age (months)"]
#         # fig, ax = plt.subplots()
#         # shap.plots.waterfall(shap_values[0], show=False, feature_names=feature_names)
#         # st.pyplot(fig)
#         feature_names = ["Gender(Male=1)", "Chronological Age (months)"]
#         # Attach feature names into SHAP Explanation
#         shap_values.feature_names = feature_names
#         # Plot waterfall without passing names directly
#         fig, ax = plt.subplots()
#         shap.plots.waterfall(shap_values[0], show=False)
#         st.pyplot(fig)
        

#         # ---- Comparison: Chronological Age vs Predicted Bone Age ----
#         st.subheader("📊 Chronological vs Predicted Bone Age")

#         ages = ["Chronological Age", "Predicted Bone Age"]
#         values = [chrono_age, prediction]

#         fig, ax = plt.subplots(figsize=(5,3))
#         bars = ax.bar(ages, values, color=["skyblue", "salmon"])
#         ax.set_ylabel("Age (months)")

#         # Annotate bars with values
#         for bar in bars:
#             yval = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha="center", va="bottom")

#         st.pyplot(fig)

#         # Highlight interpretation
#         if prediction > chrono_age:
#             st.info(f"🟢 Predicted bone age is **advanced** by {prediction - chrono_age:.1f} months compared to chronological age.")
#         elif prediction < chrono_age:
#             st.warning(f"🔴 Predicted bone age is **delayed** by {chrono_age - prediction:.1f} months compared to chronological age.")
#         else:
#             st.success("✅ Predicted bone age matches chronological age.")

        
######### re-design  ############################################################################
# elif page == "Inference":
#     st.title("🧪 Bone Age Prediction with Explainability")

#     uploaded_img = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     chrono_age = st.number_input("Chronological Age (months)", min_value=0, max_value=240, value=120)

#     if uploaded_img:
#         import torch
#         import torch.nn as nn
#         import numpy as np
#         import shap
#         import matplotlib.pyplot as plt
#         from torchvision import models, transforms
#         from PIL import Image
#         import cv2
#         import pandas as pd

#         # ---- Preprocess image ----
#         image = Image.open(uploaded_img).convert("RGB")
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#         img_tensor = transform(image).unsqueeze(0)
#         meta_tensor = torch.tensor([[1 if gender == "Male" else 0, chrono_age]], dtype=torch.float32)

#         # ---- Load pretrained model ----
#         base_model = models.resnet18(weights=None)
#         in_features = base_model.fc.in_features
#         base_model.fc = nn.Identity()

#         class CombinedModel(nn.Module):
#             def __init__(self, base_model, in_features):
#                 super().__init__()
#                 self.cnn = base_model
#                 self.fc1 = nn.Linear(in_features + 2, 128)
#                 self.fc2 = nn.Linear(128, 1)

#             def forward(self, x_img, x_meta):
#                 x_img = self.cnn(x_img)
#                 x = torch.cat([x_img, x_meta], dim=1)
#                 x = torch.relu(self.fc1(x))
#                 x = self.fc2(x)
#                 return x

#         model = CombinedModel(base_model, in_features)
#         model.load_state_dict(torch.load("boneage_model_quick.pth", map_location="cpu"))
#         model.eval()

#         # ---- Prediction ----
#         with torch.no_grad():
#             prediction = model(img_tensor, meta_tensor).item()

#         st.image(image, caption="Uploaded X-ray", use_container_width=True)
#         st.success(f"📊 Predicted Bone Age: **{prediction:.1f} months**")

#         # ---- Tabs for explanations ----
#         tab1, tab2, tab3 = st.tabs(["Grad-CAM", "SHAP Metadata", "Age Comparison"])

#         # ---------------- TAB 1: GRAD-CAM ----------------
#         with tab1:
#             st.subheader("🔍 Grad-CAM Heatmap")

#             def generate_gradcam(model, img_tensor, meta_tensor, target_layer):
#                 model.eval()
#                 gradients, activations = [], []

#                 def backward_hook(module, grad_input, grad_output):
#                     gradients.append(grad_output[0])

#                 def forward_hook(module, input, output):
#                     activations.append(output)

#                 handle_fw = target_layer.register_forward_hook(forward_hook)
#                 handle_bw = target_layer.register_backward_hook(backward_hook)

#                 output = model(img_tensor.requires_grad_(), meta_tensor)
#                 output = output.squeeze()   # ensure scalar
#                 output.backward(retain_graph=True)

#                 grads = gradients[0].detach().cpu().numpy()[0]
#                 acts = activations[0].detach().cpu().numpy()[0]

#                 weights = np.mean(grads, axis=(1, 2))
#                 cam = np.zeros(acts.shape[1:], dtype=np.float32)
#                 for w, act in zip(weights, acts):
#                     cam += w * act

#                 cam = np.maximum(cam, 0)
#                 cam = cam / cam.max()
#                 handle_fw.remove()
#                 handle_bw.remove()
#                 return cam

#             target_layer = model.cnn.layer4[1].conv2
#             cam = generate_gradcam(model, img_tensor, meta_tensor, target_layer)

#             img_cv = np.array(image.resize((224, 224)))
#             cam_resized = cv2.resize(cam, (224, 224))
#             heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#             superimposed = np.uint8(0.5 * heatmap + 0.5 * img_cv)

#             st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

#         # ---------------- TAB 2: SHAP ----------------
#         with tab2:
#             st.subheader("📈 SHAP Values (Metadata)")

#             def metadata_predict(x_meta_np):
#                 x_meta_tensor = torch.tensor(x_meta_np, dtype=torch.float32)
#                 with torch.no_grad():
#                     preds = model(img_tensor.repeat(x_meta_tensor.size(0),1,1,1), x_meta_tensor)
#                 return preds.numpy()

#             explainer = shap.Explainer(metadata_predict, np.array([[1 if gender == "Male" else 0, chrono_age]]))
#             shap_values = explainer(np.array([[1 if gender == "Male" else 0, chrono_age]]))

#             feature_names = ["Gender(Male=1)", "Chronological Age (months)"]
#             shap_values.feature_names = feature_names

#             try:
#                 fig, ax = plt.subplots()
#                 shap.plots.waterfall(shap_values[0], show=False)
#                 st.pyplot(fig)
#             except Exception:
#                 st.warning("⚠️ SHAP waterfall failed. Showing bar chart instead.")
#                 df_shap = pd.DataFrame({
#                     "Feature": feature_names,
#                     "SHAP Value": shap_values.values[0]
#                 })
#                 st.bar_chart(df_shap.set_index("Feature"))

#         # ---------------- TAB 3: COMPARISON ----------------
#         with tab3:
#             st.subheader("📊 Chronological vs Predicted Age")

#             ages = ["Chronological Age", "Predicted Bone Age"]
#             values = [chrono_age, prediction]

#             fig, ax = plt.subplots(figsize=(5,3))
#             bars = ax.bar(ages, values, color=["skyblue", "salmon"])
#             ax.set_ylabel("Age (months)")
#             for bar in bars:
#                 yval = bar.get_height()
#                 ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha="center", va="bottom")
#             st.pyplot(fig)

#             if prediction > chrono_age:
#                 st.info(f"🟢 Predicted bone age is **advanced** by {prediction - chrono_age:.1f} months.")
#             elif prediction < chrono_age:
#                 st.warning(f"🔴 Predicted bone age is **delayed** by {chrono_age - prediction:.1f} months.")
#             else:
#                 st.success("✅ Predicted bone age matches chronological age.")

# elif page == "Inference":
#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         st.image("logo.png", width=200)

#     st.set_page_config(page_title="Bone Age Predictor", layout="wide")

#     st.title("🦴 Bone Age Prediction with Explainability")
#     st.markdown("AI-powered skeletal maturity assessment using X-rays, gender, and age.")

#     uploaded_img = st.file_uploader("📤 Upload X-ray Image", type=["png", "jpg", "jpeg"])
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     chrono_age = st.number_input("Chronological Age (months)", min_value=0, max_value=240, value=120)

#     if uploaded_img:
#         import torch
#         import torch.nn as nn
#         import numpy as np
#         import shap
#         import matplotlib.pyplot as plt
#         from torchvision import models, transforms
#         from PIL import Image
#         import cv2
#         import pandas as pd

#         # ---- Preprocess image ----
#         image = Image.open(uploaded_img).convert("RGB")
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#         img_tensor = transform(image).unsqueeze(0)
#         meta_tensor = torch.tensor([[1 if gender == "Male" else 0, chrono_age]], dtype=torch.float32)

#         # ---- Load your FULL trained model ----
#         base_model = models.resnet18(weights=None)
#         in_features = base_model.fc.in_features
#         base_model.fc = nn.Identity()

#         class CombinedModel(nn.Module):
#             def __init__(self, base_model, in_features):
#                 super().__init__()
#                 self.cnn = base_model
#                 self.fc1 = nn.Linear(in_features + 2, 128)  # gender + age
#                 self.fc2 = nn.Linear(128, 1)

#             def forward(self, x_img, x_meta):
#                 x_img = self.cnn(x_img)
#                 x = torch.cat([x_img, x_meta], dim=1)
#                 x = torch.relu(self.fc1(x))
#                 x = self.fc2(x)
#                 return x

#         model = CombinedModel(base_model, in_features)

#         # ✅ load the model you trained on full dataset
#         model.load_state_dict(torch.load("boneage_model.pth", map_location="cpu"))
#         model.eval()

#         # ---- Prediction ----
#         with torch.no_grad():
#             prediction = model(img_tensor, meta_tensor).item()

# ---------------------------
# Page 3: Inference
# ---------------------------
elif page == "Inference":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.png", width=200)

    st.set_page_config(page_title="Bone Age Predictor", layout="wide")

    st.title("🦴 Bone Age Prediction with Explainability")
    st.markdown("AI-powered skeletal maturity assessment using X-rays, gender, and age.")

    uploaded_img = st.file_uploader("📤 Upload X-ray Image", type=["png", "jpg", "jpeg"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    chrono_age = st.number_input("Chronological Age (months)", min_value=0, max_value=240, value=120)

    # ---------------------------
    # Hybrid model fetch (GitHub LFS + Google Drive fallback)
    # ---------------------------
    import os, gdown

    MODEL_PATH = "boneage_model.pth"
#https://drive.google.com/file/d/1oD9hWWnwtFD8Qd4UwPMGruTyKYYZuBiK/view?usp=sharing
    FILE_ID = "1oD9hWWnwtFD8Qd4UwPMGruTyKYYZuBiK"  # 🔴 replace with your Google Drive file ID
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
        import torch
        import torch.nn as nn
        import numpy as np
        import shap
        import matplotlib.pyplot as plt
        from torchvision import models, transforms
        from PIL import Image
        import cv2
        import pandas as pd

        # ---- Preprocess image ----
        image = Image.open(uploaded_img).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        meta_tensor = torch.tensor([[1 if gender == "Male" else 0, chrono_age]], dtype=torch.float32)

        # ---- Load your FULL trained model ----
        base_model = models.resnet18(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Identity()

        class CombinedModel(nn.Module):
            def __init__(self, base_model, in_features):
                super().__init__()
                self.cnn = base_model
                self.fc1 = nn.Linear(in_features + 2, 128)  # gender + age
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

        # ---- Prediction ----
        with torch.no_grad():
            prediction = model(img_tensor, meta_tensor).item()

        # (rest of your inference code: results, Grad-CAM, SHAP, comparison tabs...)


         # Layout: Image left, results right
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
        with col2:
            st.markdown("### Prediction Results")
            st.metric("Chronological Age", f"{chrono_age} months")
            st.metric("Predicted Bone Age", f"{prediction:.1f} months", 
                      delta=f"{prediction - chrono_age:.1f} months")

        # ---- Tabs for explanations ----
        tab1, tab2, tab3 = st.tabs(["Grad-CAM", "SHAP Metadata", "Age Comparison"])

        # TAB 1: Grad-CAM
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
            st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

        # TAB 2: SHAP
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

            try:
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            except Exception:
                st.warning("⚠️ SHAP waterfall failed. Showing bar chart instead.")
                df_shap = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_values.values[0]
                })
                st.bar_chart(df_shap.set_index("Feature"))

        # TAB 3: Comparison # ---------------- TAB 3: COMPARISON ----------------
        # ---------------- TAB 3: COMPARISON ----------------
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
                st.markdown(
                    "💡 **Clinical Interpretation:** The model predicts this child's skeletal maturity is **advanced** "
                    "relative to their actual chronological age. In clinical practice, this may suggest **early or precocious skeletal development**, "
                    "which can occur in conditions such as **early puberty** or certain **endocrine disorders**."
                )
            elif delta < 0:
                st.warning(f"🔴 Predicted bone age is **delayed** by {-delta:.1f} months compared to chronological age.")
                st.markdown(
                    "💡 **Clinical Interpretation:** The model predicts this child's skeletal maturity is **delayed** "
                    "relative to their actual chronological age. In clinical settings, this may indicate **growth delay**, "
                    "which could be associated with conditions such as **growth hormone deficiency**, **malnutrition**, "
                    "or other **systemic illnesses**."
                )
            else:
                st.success("✅ Predicted bone age matches chronological age.")
                st.markdown(
                    "💡 **Clinical Interpretation:** The predicted bone age **matches** the child's chronological age. "
                    "This suggests that skeletal development is proceeding at a normal rate."
                )


        
        # # with tab3:
        #     st.subheader("📊 Chronological vs Predicted Age")
        #     ages = ["Chronological Age", "Predicted Bone Age"]
        #     values = [chrono_age, prediction]

        #     fig, ax = plt.subplots(figsize=(5,3))
        #     bars = ax.bar(ages, values, color=["skyblue", "salmon"])
        #     ax.set_ylabel("Age (months)")
        #     for bar in bars:
        #         yval = bar.get_height()
        #         ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha="center", va="bottom")
        #     st.pyplot(fig)

        #     if prediction > chrono_age:
        #         st.info(f"🟢 Predicted bone age is **advanced** by {prediction - chrono_age:.1f} months.")
        #     elif prediction < chrono_age:
        #         st.warning(f"🔴 Predicted bone age is **delayed** by {chrono_age - prediction:.1f} months.")
        #     else:
        #         st.success("✅ Predicted bone age matches chronological age.")



# ############################################################################################################










# ================================== #
# Bone Age Predictor - Full Streamlit App
# ================================== #

# import os
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import cv2

# # --------------------------- #
# # Page Config
# # --------------------------- #
# st.set_page_config(page_title="Bone Age Predictor", layout="wide", page_icon="🦴")

# # --------------------------- #
# # Sidebar Branding
# # --------------------------- #
# with st.sidebar:
#     logo_path = "logo.png"  # Place your logo in the same folder as app.py
#     if logo_path and os.path.exists(logo_path):
#         try:
#             logo_img = Image.open(logo_path)
#             st.image(logo_img, width=120)
#         except Exception:
#             st.warning("⚠️ Unable to load logo.")
#     else:
#         st.info("Logo file not found. Skipping logo display.")

#     st.markdown("### 🦴 Bone Age Predictor")
#     st.markdown("AI-powered tool for skeletal maturity assessment.")

#     page = st.radio("Navigate", ["Overview", "Train", "Inference"])

# # --------------------------- #
# # Keep dataset & folder path across pages
# # --------------------------- #
# if "df" not in st.session_state:
#     st.session_state.df = None
# if "folder_path" not in st.session_state:
#     st.session_state.folder_path = None

# # --------------------------- #
# # Overview Page
# # --------------------------- #
# if page == "Overview":
#     st.title("📊 Dataset Overview")
#     uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
#     folder_path = st.text_input("Enter images folder path")

#     if uploaded_csv is not None:
#         df = pd.read_csv(uploaded_csv)
#         st.session_state.df = df
#         st.session_state.folder_path = folder_path if folder_path else None
#         st.write("Preview of dataset:")
#         st.dataframe(df.head())

#         if folder_path and os.path.isdir(folder_path):
#             st.write("Sample Images:")
#             sample_files = os.listdir(folder_path)[:5]
#             cols = st.columns(len(sample_files))
#             for i, f in enumerate(sample_files):
#                 try:
#                     cols[i].image(os.path.join(folder_path, f), width=120)
#                 except:
#                     pass

# # --------------------------- #
# # Training Page
# # --------------------------- #
# elif page == "Train":
#     st.title("🧑‍🏫 Train Bone Age Model")
#     csv_file = st.file_uploader("Upload dataset CSV", type=["csv"])
#     img_folder = st.text_input("Enter image folder path")

#     if csv_file and img_folder:
#         df = pd.read_csv(csv_file)
#         df = df.sample(1000, random_state=42).reset_index(drop=True)  # quick testing
#         st.write(f"Using {len(df)} samples for training (subset).")

#         if st.button("Start Training"):
#             st.info("Training started... This may take a while.")

#             # Dataset and transform
#             transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406],
#                                      [0.229, 0.224, 0.225])
#             ])

#             class BoneAgeDataset(Dataset):
#                 def __init__(self, df, img_dir, transform=None):
#                     self.df = df
#                     self.img_dir = img_dir
#                     self.transform = transform
#                 def __len__(self):
#                     return len(self.df)
#                 def __getitem__(self, idx):
#                     row = self.df.iloc[idx]
#                     img_path = os.path.join(self.img_dir, str(row["id"]) + ".png")
#                     image = Image.open(img_path).convert("RGB")
#                     if self.transform:
#                         image = self.transform(image)
#                     meta = torch.tensor([row["male"]], dtype=torch.float32)
#                     target = torch.tensor(row["boneage"], dtype=torch.float32)
#                     return image, meta, target

#             dataset = BoneAgeDataset(df, img_folder, transform)
#             train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#             # Model
#             base_model = models.resnet18(pretrained=True)
#             in_features = base_model.fc.in_features
#             base_model.fc = nn.Identity()

#             class CombinedModel(nn.Module):
#                 def __init__(self, base_model, in_features):
#                     super().__init__()
#                     self.cnn = base_model
#                     self.fc1 = nn.Linear(in_features + 1, 128)
#                     self.fc2 = nn.Linear(128, 1)
#                 def forward(self, x_img, x_meta):
#                     x_img = self.cnn(x_img)
#                     x = torch.cat([x_img, x_meta], dim=1)
#                     x = torch.relu(self.fc1(x))
#                     x = self.fc2(x)
#                     return x

#             model = CombinedModel(base_model, in_features)
#             criterion = nn.MSELoss()
#             optimizer = optim.Adam(model.parameters(), lr=1e-4)
#             num_epochs = 2
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             model.to(device)

#             # Progress & Loss chart
#             loss_list = []
#             progress_bar = st.progress(0)
#             status = st.empty()

#             for epoch in range(num_epochs):
#                 running_loss = 0.0
#                 for i, (imgs, meta, targets) in enumerate(train_loader):
#                     imgs, meta, targets = imgs.to(device), meta.to(device), targets.to(device).unsqueeze(1)
#                     optimizer.zero_grad()
#                     outputs = model(imgs, meta)
#                     loss = criterion(outputs, targets)
#                     loss.backward()
#                     optimizer.step()
#                     running_loss += loss.item()

#                     # Update UI
#                     batch_progress = (i + 1) / len(train_loader)
#                     total_progress = (epoch + batch_progress) / num_epochs
#                     progress_bar.progress(total_progress)
#                     status.text(f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

#                 avg_loss = running_loss / len(train_loader)
#                 loss_list.append(avg_loss)
#                 st.write(f"Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

#             st.line_chart(loss_list)
#             torch.save(model.state_dict(), "boneage_model.pth")
#             st.success("✅ Training complete. Model saved as boneage_model.pth")

# # --------------------------- #
# # Inference Page
# # --------------------------- #
# # --------------------------- #
# # Inference Page with original Grad-CAM & SHAP
# # --------------------------- #
# elif page == "Inference":
#     st.title("🦴 Bone Age Prediction with Explainability")
#     st.markdown("Upload an X-ray, select gender and chronological age to predict bone age.")

#     uploaded_img = st.file_uploader("📤 Upload X-ray", type=["png", "jpg", "jpeg"])
#     col1, col2 = st.columns([1,1])
#     with col1:
#         gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True)
#     with col2:
#         chrono_age = st.number_input("Chronological Age (months)", min_value=0, max_value=240, value=120)

#     if uploaded_img:
#         image = Image.open(uploaded_img).convert("RGB")

#         # ---- Preprocess image ----
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#         img_tensor = transform(image).unsqueeze(0)
#         meta_tensor = torch.tensor([[1 if gender == "Male" else 0, chrono_age]], dtype=torch.float32)

#         # ---- Load pretrained model ----
#         base_model = models.resnet18(weights=None)
#         in_features = base_model.fc.in_features
#         base_model.fc = nn.Identity()

#         class CombinedModel(nn.Module):
#             def __init__(self, base_model, in_features):
#                 super().__init__()
#                 self.cnn = base_model
#                 self.fc1 = nn.Linear(in_features + 2, 128)
#                 self.fc2 = nn.Linear(128, 1)
#             def forward(self, x_img, x_meta):
#                 x_img = self.cnn(x_img)
#                 x = torch.cat([x_img, x_meta], dim=1)
#                 x = torch.relu(self.fc1(x))
#                 x = self.fc2(x)
#                 return x

#         model = CombinedModel(base_model, in_features)
#         model.load_state_dict(torch.load("boneage_model_quick.pth", map_location="cpu"))
#         model.eval()

#         # ---- Prediction ----
#         with torch.no_grad():
#             prediction = model(img_tensor, meta_tensor).item()

#         # --------------------------- #
#         # Two-column layout: image left, results right
#         # --------------------------- #
#         img_col, res_col = st.columns([1,1])
#         with img_col:
#             st.image(image, caption="Uploaded X-ray", use_container_width=True)
#         with res_col:
#             st.markdown("### Results")
#             col1_metric, col2_metric = st.columns(2)
#             col1_metric.metric("Chronological Age", f"{chrono_age} months")
#             col2_metric.metric("Predicted Bone Age", f"{prediction:.1f} months",
#                                 delta=f"{prediction - chrono_age:.1f} months")

#         # --------------------------- #
#         # Tabs for explainability
#         # --------------------------- #
#         tab1, tab2, tab3 = st.tabs(["Grad-CAM Heatmap", "SHAP Metadata", "Age Comparison"])

#         # --- TAB 1: Grad-CAM ---
#         with tab1:
#             st.subheader("🔍 Grad-CAM Heatmap")

#             def generate_gradcam(model, img_tensor, meta_tensor, target_layer):
#                 model.eval()
#                 gradients, activations = [], []

#                 def backward_hook(module, grad_input, grad_output):
#                     gradients.append(grad_output[0])

#                 def forward_hook(module, input, output):
#                     activations.append(output)

#                 handle_fw = target_layer.register_forward_hook(forward_hook)
#                 handle_bw = target_layer.register_backward_hook(backward_hook)

#                 output = model(img_tensor.requires_grad_(), meta_tensor)
#                 output = output.squeeze()
#                 output.backward(retain_graph=True)

#                 grads = gradients[0].detach().cpu().numpy()[0]
#                 acts = activations[0].detach().cpu().numpy()[0]

#                 weights = np.mean(grads, axis=(1, 2))
#                 cam = np.zeros(acts.shape[1:], dtype=np.float32)
#                 for w, act in zip(weights, acts):
#                     cam += w * act

#                 cam = np.maximum(cam, 0)
#                 cam = cam / cam.max()
#                 handle_fw.remove()
#                 handle_bw.remove()
#                 return cam

#             target_layer = model.cnn.layer4[1].conv2
#             cam = generate_gradcam(model, img_tensor, meta_tensor, target_layer)

#             img_cv = np.array(image.resize((224, 224)))
#             cam_resized = cv2.resize(cam, (224, 224))
#             heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#             superimposed = np.uint8(0.5 * heatmap + 0.5 * img_cv)

#             st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

#         # --- TAB 2: SHAP ---
#         with tab2:
#             st.subheader("📈 SHAP Values (Metadata)")
#             import shap

#             def metadata_predict(x_meta_np):
#                 x_meta_tensor = torch.tensor(x_meta_np, dtype=torch.float32)
#                 with torch.no_grad():
#                     preds = model(img_tensor.repeat(x_meta_tensor.size(0),1,1,1), x_meta_tensor)
#                 return preds.numpy()

#             explainer = shap.Explainer(metadata_predict, np.array([[1 if gender=="Male" else 0, chrono_age]]))
#             shap_values = explainer(np.array([[1 if gender=="Male" else 0, chrono_age]]))
#             feature_names = ["Gender(Male=1)", "Chronological Age (months)"]
#             shap_values.feature_names = feature_names

#             try:
#                 fig, ax = plt.subplots()
#                 shap.plots.waterfall(shap_values[0], show=False)
#                 st.pyplot(fig)
#             except Exception:
#                 st.warning("⚠️ SHAP waterfall failed. Showing bar chart instead.")
#                 df_shap = pd.DataFrame({
#                     "Feature": feature_names,
#                     "SHAP Value": shap_values.values[0]
#                 })
#                 st.bar_chart(df_shap.set_index("Feature"))

#         # --- TAB 3: Age Comparison ---
#         with tab3:
#             st.subheader("📊 Chronological vs Predicted Age")
#             ages = ["Chronological Age", "Predicted Bone Age"]
#             values = [chrono_age, prediction]
#             fig, ax = plt.subplots(figsize=(5,3))
#             bars = ax.bar(ages, values, color=["skyblue","salmon"])
#             for bar in bars:
#                 yval = bar.get_height()
#                 ax.text(bar.get_x() + bar.get_width()/2, yval+2, f"{yval:.1f}", ha="center")
#             st.pyplot(fig)

#             if prediction > chrono_age:
#                 st.info(f"🟢 Predicted bone age is **advanced** by {prediction - chrono_age:.1f} months.")
#             elif prediction < chrono_age:
#                 st.warning(f"🔴 Predicted bone age is **delayed** by {chrono_age - prediction:.1f} months.")
#             else:
#                 st.success("✅ Predicted bone age matches chronological age.")









# ############################################ with loss curves













# import os
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import cv2

# # --------------------------- #
# # Page Config
# # --------------------------- #
# st.set_page_config(page_title="Bone Age Predictor", layout="wide", page_icon="🦴")

# # --------------------------- #
# # Sidebar Branding
# # --------------------------- #
# with st.sidebar:
#     logo_path = "logo.png"
#     if logo_path and os.path.exists(logo_path):
#         try:
#             logo_img = Image.open(logo_path)
#             st.image(logo_img, width=120)
#         except Exception:
#             st.warning("⚠️ Unable to load logo.")
#     else:
#         st.info("Logo file not found. Skipping logo display.")

#     st.markdown("### 🦴 Bone Age Predictor")
#     st.markdown("AI-powered tool for skeletal maturity assessment.")

#     page = st.radio("Navigate", ["Overview", "Train", "Inference"])

# # --------------------------- #
# # Keep dataset & folder path across pages
# # --------------------------- #
# if "df" not in st.session_state:
#     st.session_state.df = None
# if "folder_path" not in st.session_state:
#     st.session_state.folder_path = None

# # --------------------------- #
# # Overview Page
# # --------------------------- #
# if page == "Overview":
#     st.title("📊 Dataset Overview")
#     uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
#     folder_path = st.text_input("Enter images folder path")

#     if uploaded_csv is not None:
#         df = pd.read_csv(uploaded_csv)
#         st.session_state.df = df
#         st.session_state.folder_path = folder_path if folder_path else None
#         st.write("Preview of dataset:")
#         st.dataframe(df.head())

#         if folder_path and os.path.isdir(folder_path):
#             st.write("Sample Images:")
#             sample_files = os.listdir(folder_path)[:5]
#             cols = st.columns(len(sample_files))
#             for i, f in enumerate(sample_files):
#                 try:
#                     cols[i].image(os.path.join(folder_path, f), width=120)
#                 except:
#                     pass

# # --------------------------- #
# # Training Page
# # --------------------------- #
# elif page == "Train":
#     st.title("🧑‍🏫 Train Bone Age Model")

#     st.info("Model already trained on full dataset. Showing saved loss curves.")

#     # Load pre-saved training log if you saved during Colab
#     log_path = "training_log.csv"  # optional if you saved loss history
#     if os.path.exists(log_path):
#         log_df = pd.read_csv(log_path)
#         st.line_chart(log_df[["loss", "mae", "rmse"]])
#     else:
#         st.warning("⚠️ No training log found. Please save 'training_log.csv' during training.")

#     if os.path.exists("boneage_model.pth"):
#         st.success("✅ Trained model loaded from boneage_model.pth")
#     else:
#         st.error("❌ Model file 'boneage_model.pth' not found. Place it in the project folder.")

# # --------------------------- #
# # Inference Page
# # --------------------------- #
# elif page == "Inference":
#     st.title("🦴 Bone Age Prediction with Explainability")
#     st.markdown("Upload an X-ray, select gender and chronological age to predict bone age.")

#     uploaded_img = st.file_uploader("📤 Upload X-ray", type=["png", "jpg", "jpeg"])
#     col1, col2 = st.columns([1,1])
#     with col1:
#         gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True)
#     with col2:
#         chrono_age = st.number_input("Chronological Age (months)", min_value=0, max_value=240, value=120)

#     if uploaded_img:
#         image = Image.open(uploaded_img).convert("RGB")

#         # ---- Preprocess image ----
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#         img_tensor = transform(image).unsqueeze(0)
#         meta_tensor = torch.tensor([[1 if gender == "Male" else 0, chrono_age]], dtype=torch.float32)

#         # ---- Load trained model ----
#         base_model = models.resnet18(weights=None)
#         in_features = base_model.fc.in_features
#         base_model.fc = nn.Identity()

#         class CombinedModel(nn.Module):
#             def __init__(self, base_model, in_features):
#                 super().__init__()
#                 self.cnn = base_model
#                 self.fc1 = nn.Linear(in_features + 2, 128)  # gender + age
#                 self.fc2 = nn.Linear(128, 1)
#             def forward(self, x_img, x_meta):
#                 x_img = self.cnn(x_img)
#                 x = torch.cat([x_img, x_meta], dim=1)
#                 x = torch.relu(self.fc1(x))
#                 x = self.fc2(x)
#                 return x

#         model = CombinedModel(base_model, in_features)
#         model.load_state_dict(torch.load("boneage_model.pth", map_location="cpu"))
#         model.eval()

#         # ---- Prediction ----
#         with torch.no_grad():
#             prediction = model(img_tensor, meta_tensor).item()

#         # --------------------------- #
#         # Layout: image + results
#         # --------------------------- #
#         img_col, res_col = st.columns([1,1])
#         with img_col:
#             st.image(image, caption="Uploaded X-ray", use_container_width=True)
#         with res_col:
#             st.markdown("### Results")
#             col1_metric, col2_metric = st.columns(2)
#             col1_metric.metric("Chronological Age", f"{chrono_age} months")
#             col2_metric.metric("Predicted Bone Age", f"{prediction:.1f} months",
#                                 delta=f"{prediction - chrono_age:.1f} months")

#         # --------------------------- #
#         # Tabs for explainability
#         # --------------------------- #
#         tab1, tab2, tab3 = st.tabs(["Grad-CAM Heatmap", "SHAP Metadata", "Age Comparison"])

#         # --- TAB 1: Grad-CAM ---
#         with tab1:
#             st.subheader("🔍 Grad-CAM Heatmap")

#             def generate_gradcam(model, img_tensor, meta_tensor, target_layer):
#                 model.eval()
#                 gradients, activations = [], []

#                 def backward_hook(module, grad_input, grad_output):
#                     gradients.append(grad_output[0])

#                 def forward_hook(module, input, output):
#                     activations.append(output)

#                 handle_fw = target_layer.register_forward_hook(forward_hook)
#                 handle_bw = target_layer.register_backward_hook(backward_hook)

#                 output = model(img_tensor.requires_grad_(), meta_tensor)
#                 output = output.squeeze()
#                 output.backward(retain_graph=True)

#                 grads = gradients[0].detach().cpu().numpy()[0]
#                 acts = activations[0].detach().cpu().numpy()[0]

#                 weights = np.mean(grads, axis=(1, 2))
#                 cam = np.zeros(acts.shape[1:], dtype=np.float32)
#                 for w, act in zip(weights, acts):
#                     cam += w * act

#                 cam = np.maximum(cam, 0)
#                 cam = cam / cam.max()
#                 handle_fw.remove()
#                 handle_bw.remove()
#                 return cam

#             target_layer = model.cnn.layer4[1].conv2
#             cam = generate_gradcam(model, img_tensor, meta_tensor, target_layer)

#             img_cv = np.array(image.resize((224, 224)))
#             cam_resized = cv2.resize(cam, (224, 224))
#             heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#             superimposed = np.uint8(0.5 * heatmap + 0.5 * img_cv)

#             st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

#         # --- TAB 2: SHAP ---
#         with tab2:
#             st.subheader("📈 SHAP Values (Metadata)")
#             import shap

#             def metadata_predict(x_meta_np):
#                 x_meta_tensor = torch.tensor(x_meta_np, dtype=torch.float32)
#                 with torch.no_grad():
#                     preds = model(img_tensor.repeat(x_meta_tensor.size(0),1,1,1), x_meta_tensor)
#                 return preds.numpy()

#             explainer = shap.Explainer(metadata_predict, np.array([[1 if gender=="Male" else 0, chrono_age]]))
#             shap_values = explainer(np.array([[1 if gender=="Male" else 0, chrono_age]]))
#             feature_names = ["Gender(Male=1)", "Chronological Age (months)"]
#             shap_values.feature_names = feature_names

#             try:
#                 fig, ax = plt.subplots()
#                 shap.plots.waterfall(shap_values[0], show=False)
#                 st.pyplot(fig)
#             except Exception:
#                 st.warning("⚠️ SHAP waterfall failed. Showing bar chart instead.")
#                 df_shap = pd.DataFrame({
#                     "Feature": feature_names,
#                     "SHAP Value": shap_values.values[0]
#                 })
#                 st.bar_chart(df_shap.set_index("Feature"))

#         # --- TAB 3: Age Comparison ---
#         with tab3:
#             st.subheader("📊 Chronological vs Predicted Age")
#             ages = ["Chronological Age", "Predicted Bone Age"]
#             values = [chrono_age, prediction]
#             fig, ax = plt.subplots(figsize=(5,3))
#             bars = ax.bar(ages, values, color=["skyblue","salmon"])
#             for bar in bars:
#                 yval = bar.get_height()
#                 ax.text(bar.get_x() + bar.get_width()/2, yval+2, f"{yval:.1f}", ha="center")
#             st.pyplot(fig)

#             if prediction > chrono_age:
#                 st.info(f"🟢 Predicted bone age is **advanced** by {prediction - chrono_age:.1f} months.")
#             elif prediction < chrono_age:
#                 st.warning(f"🔴 Predicted bone age is **delayed** by {chrono_age - prediction:.1f} months.")
#             else:
#                 st.success("✅ Predicted bone age matches chronological age.")


