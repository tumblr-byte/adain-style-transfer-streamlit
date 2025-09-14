import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="AdaIN Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# Your existing model code
def calc_mean_std(features, eps=1e-6):
    batch_size, c = features.size()[:2]
    features_reshaped = features.reshape(batch_size, c, -1)
    features_mean = features_reshaped.mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features_reshaped.std(dim=2).reshape(batch_size, c, 1, 1) + eps
    return features_mean, features_std

def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        self.slice4 = vgg[12:21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4

class RC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            h = F.relu(h)
        return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()

    def generate(self, content_images, style_images, alpha=1.0):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)
        return out

# Image preprocessing
test_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])






@st.cache_resource
def load_model():
    """Load the pre-trained AdaIN model from GitHub Release"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    
    MODEL_PATH = "model.pth"
    MODEL_URL = "https://github.com/tumblr-byte/adain-style-transfer-streamlit/releases/download/v1.0/model.pth"

    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading pre-trained model from GitHub...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded successfully!")

    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        st.success("✅ Pre-trained model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
    
    return model.to(device), device


def preprocess_image(image):
    """Preprocess image for model input"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return test_transform(image).unsqueeze(0)




def postprocess_image(tensor, device):
    """Convert model output back to displayable image"""
    tensor = denorm(tensor, device)
    tensor = tensor.cpu().squeeze(0)
    image = tensor.permute(1, 2, 0).numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image



def process_video_frame(model, device, content_frame, style_tensor, alpha):
    """Process a single video frame"""
    content_pil = Image.fromarray(cv2.cvtColor(content_frame, cv2.COLOR_BGR2RGB))
    content_tensor = preprocess_image(content_pil).to(device)
    
    with torch.no_grad():
        stylized = model.generate(content_tensor, style_tensor, alpha)
    
    stylized_frame = postprocess_image(stylized, device)
    return cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)

def main():
    st.title("🎨 AdaIN Style Transfer App")
    st.markdown("Transform your images and videos with artistic styles using Adaptive Instance Normalization!")
    
    # Load model
    model, device = load_model()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    alpha = st.sidebar.slider("Style Strength", 0.0, 1.0, 1.0, 0.1, 
                             help="0 = Original content, 1 = Full style transfer")
    
    # File uploaders
    st.header("📁 Upload Files")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Style Image")
        style_file = st.file_uploader(
            "Choose a style image (vintage, modern, artistic, etc.)", 
            type=['jpg', 'jpeg', 'png'],
            key="style"
        )
    
    with col2:
        st.subheader("Content Image/Video")
        content_file = st.file_uploader(
            "Choose content to stylize", 
            type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
            key="content"
        )
    
    if style_file and content_file:
        # Load and display style image
        style_image = Image.open(style_file).convert('RGB')
        style_tensor = preprocess_image(style_image).to(device)
        
        # Determine if content is image or video
        content_type = content_file.type
        is_video = content_type.startswith('video/')
        
        if is_video:
            st.header("🎬 Video Style Transfer")
            
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(content_file.getvalue())
                temp_path = tmp_file.name
            
            # Process video
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Show video info
            st.info(f"📊 Video Info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
            
            if st.button("🎬 Process Video", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create output video writer
                output_path = tempfile.mktemp(suffix='.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame
                        stylized_frame = process_video_frame(model, device, frame, style_tensor, alpha)
                        out.write(stylized_frame)
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    cap.release()
                    out.release()
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🖼️ Style Reference")
                        st.image(style_image, use_column_width=True)
                    
                    with col2:
                        st.subheader("🎥 Original Video")
                        st.video(temp_path)
                    
                    with col3:
                        st.subheader("🎨 Stylized Video")
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Stylized Video",
                            data=video_bytes,
                            file_name="stylized_video.mp4",
                            mime="video/mp4"
                        )
                    
                    st.success("✅ Video processing completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")
                finally:
                    # Cleanup
                    cap.release()
                    if 'out' in locals():
                        out.release()
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    if os.path.exists(output_path):
                        os.unlink(output_path)
        
        else:
            # Image processing
            st.header("🖼️ Image Style Transfer")
            
            content_image = Image.open(content_file).convert('RGB')
            content_tensor = preprocess_image(content_image).to(device)
            
            if st.button("🎨 Apply Style Transfer", type="primary"):
                with st.spinner("Applying style transfer..."):
                    with torch.no_grad():
                        stylized = model.generate(content_tensor, style_tensor, alpha)
                    stylized_image = postprocess_image(stylized, device)
                
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🖼️ Style Reference")
                    st.image(style_image, use_column_width=True)
                
                with col2:
                    st.subheader("📷 Original Content")
                    st.image(content_image, use_column_width=True)
                
                with col3:
                    st.subheader("🎨 Stylized Result")
                    st.image(stylized_image, use_column_width=True)
                    
                    # Download button
                    stylized_pil = Image.fromarray(stylized_image)
                    buf = io.BytesIO()
                    stylized_pil.save(buf, format='PNG')
                    
                    st.download_button(
                        label="⬇️ Download Stylized Image",
                        data=buf.getvalue(),
                        file_name="stylized_image.png",
                        mime="image/png"
                    )
    
    else:
        # Show examples and instructions
        st.header("🚀 How to Use")
        st.markdown("""
        1. **Upload a Style Image**: Choose an image with the artistic style you want to apply (vintage, modern, painterly, etc.)
        2. **Upload Content**: Choose the image or video you want to stylize
        3. **Adjust Style Strength**: Use the slider to control how strong the style transfer is
        4. **Process**: Click the button to apply style transfer
        5. **Download**: Save your stylized result!
        
        ### 🎨 Style Examples:
        - **Vintage**: Old photographs, sepia tones, film grain
        - **Modern**: Contemporary art, bold colors, geometric patterns
        - **Artistic**: Famous paintings, watercolors, oil paintings
        - **Abstract**: Non-representational art with interesting textures and colors
        """)
        
        st.header("ℹ️ About AdaIN Style Transfer")
        st.markdown("""
        This app uses **Adaptive Instance Normalization (AdaIN)** for real-time style transfer. 
        AdaIN transfers style by aligning the mean and variance of content features to match those of style features.
        
        - ✅ Fast processing
        - ✅ Preserves content structure  
        - ✅ Flexible style control
        - ✅ Works with images and videos
        """)

if __name__ == "__main__":

    main()
