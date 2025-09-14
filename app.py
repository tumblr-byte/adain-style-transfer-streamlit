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
import subprocess
import base64

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
    """Load the pre-trained AdaIN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    
    # For demo purposes, we'll create a dummy model
    # Replace this with your actual model loading code
    st.info("🔄 Loading pre-trained model...")
    try:

        MODEL_URL = "https://github.com/tumblr-byte/adain-style-transfer-streamlit/releases/download/v1.0/model.pth"
        model.load_state_dict(torch.load("MODEL_URL", map_location=device))
        model.eval()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Using untrained model for demo. Load your trained weights for actual style transfer.")
    
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

def create_video_with_images(frames, fps, output_path):
    """Create video from list of frames using imageio (more reliable)"""
    try:
        import imageio
        # Convert BGR frames to RGB for imageio
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        
        # Write video with imageio (better web compatibility)
        imageio.mimsave(
            output_path,
            rgb_frames,
            fps=fps,
            quality=8,  # Good quality
            macro_block_size=1  # Better compatibility
        )
        return True
    except ImportError:
        st.error("❌ Please install imageio: pip install imageio[ffmpeg]")
        return False
    except Exception as e:
        st.error(f"❌ Error creating video with imageio: {e}")
        return False

def process_with_opencv_fallback(frames, fps, output_path, width, height):
    """Fallback method using OpenCV with maximum compatibility settings"""
    try:
        # Use MJPG codec which has very good compatibility
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path.replace('.mp4', '.avi'), fourcc, fps, (width, height))
        
        if not out.isOpened():
            return False
            
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    except Exception as e:
        st.error(f"❌ OpenCV fallback failed: {e}")
        return False

def create_browser_compatible_video(frames, fps, width, height):
    """Create the most browser-compatible video possible"""
    temp_paths = []
    
    try:
        # Method 1: Try imageio with H.264 (best quality and compatibility)
        temp_path_1 = tempfile.mktemp(suffix='.mp4')
        temp_paths.append(temp_path_1)
        
        if create_video_with_images(frames, fps, temp_path_1):
            if os.path.exists(temp_path_1) and os.path.getsize(temp_path_1) > 0:
                st.success("✅ Video created with imageio (H.264)")
                return temp_path_1, temp_paths
        
        # Method 2: Try OpenCV with MJPG codec (very compatible but larger files)
        temp_path_2 = tempfile.mktemp(suffix='.avi')
        temp_paths.append(temp_path_2)
        
        if process_with_opencv_fallback(frames, fps, temp_path_2, width, height):
            if os.path.exists(temp_path_2) and os.path.getsize(temp_path_2) > 0:
                st.info("✅ Video created with OpenCV (MJPG) - larger file but very compatible")
                return temp_path_2, temp_paths
        
        # Method 3: Create WebM format (very web-compatible)
        try:
            temp_path_3 = tempfile.mktemp(suffix='.webm')
            temp_paths.append(temp_path_3)
            
            import imageio
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            
            imageio.mimsave(
                temp_path_3,
                rgb_frames,
                fps=fps,
                quality=8,
                format='FFMPEG',
                codec='libvpx-vp9'
            )
            
            if os.path.exists(temp_path_3) and os.path.getsize(temp_path_3) > 0:
                st.info("✅ Video created in WebM format (excellent web compatibility)")
                return temp_path_3, temp_paths
                
        except Exception as e:
            st.warning(f"WebM creation failed: {e}")
        
        return None, temp_paths
        
    except Exception as e:
        st.error(f"❌ All video creation methods failed: {e}")
        return None, temp_paths

def display_video_with_fallbacks(video_path, video_bytes):
    """Display video with multiple fallback methods"""
    try:
        # Method 1: Direct bytes display
        st.video(video_bytes)
        return True
    except Exception as e1:
        try:
            # Method 2: File path display
            st.video(video_path)
            return True
        except Exception as e2:
            try:
                # Method 3: Base64 embedded video (last resort)
                video_b64 = base64.b64encode(video_bytes).decode()
                video_html = f'''
                <video width="100%" height="auto" controls>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                    <source src="data:video/webm;base64,{video_b64}" type="video/webm">
                    Your browser does not support the video tag.
                </video>
                '''
                st.components.v1.html(video_html, height=400)
                return True
            except Exception as e3:
                st.error(f"❌ Could not display video: {e1}, {e2}, {e3}")
                return False

def main():
    st.title("🎨 AdaIN Style Transfer App")
    st.markdown("Transform your images and videos with artistic styles using Adaptive Instance Normalization!")
    
    # Load model
    model_result = load_model()
    model, device = model_result
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    alpha = st.sidebar.slider("Style Strength", 0.0, 1.0, 1.0, 0.1, 
                             help="0 = Original content, 1 = Full style transfer")
    
    # Video settings
    st.sidebar.subheader("🎥 Video Settings")
    max_video_size = st.sidebar.selectbox(
        "Max Video Resolution",
        options=[256, 512, 720],
        index=1,
        help="Lower resolution = faster processing and smaller files"
    )
    
    max_frames = st.sidebar.number_input(
        "Max Frames to Process",
        min_value=10,
        max_value=300,
        value=100,
        help="Limit frames for faster processing and testing"
    )
    
    # File uploaders
    st.header("📁 Upload Files")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Style Image")
        style_file = st.file_uploader(
            "Choose a style image", 
            type=['jpg', 'jpeg', 'png'],
            key="style"
        )
    
    with col2:
        st.subheader("Content Image/Video")
        content_file = st.file_uploader(
            "Choose content to stylize", 
            type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'webm'],
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
                temp_input_path = tmp_file.name
            
            # Process video
            cap = cv2.VideoCapture(temp_input_path)
            if not cap.isOpened():
                st.error("❌ Could not open video file. Please try a different format.")
                os.unlink(temp_input_path)
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Limit frames for processing
            frames_to_process = min(max_frames, total_frames)
            
            # Calculate output dimensions
            aspect_ratio = original_width / original_height
            if original_width > original_height:
                width = min(max_video_size, original_width)
                height = int(width / aspect_ratio)
            else:
                height = min(max_video_size, original_height)
                width = int(height * aspect_ratio)
            
            # Ensure dimensions are even
            width = width if width % 2 == 0 else width - 1
            height = height if height % 2 == 0 else height - 1
            
            st.info(f"📊 Processing {frames_to_process} frames | {original_width}x{original_height} → {width}x{height} | {fps:.1f} FPS")
            
            if st.button("🎬 Process Video", type="primary"):
                
                # Install imageio if not available
                try:
                    import imageio
                except ImportError:
                    st.error("❌ Please install imageio for video processing:")
                    st.code("pip install imageio[ffmpeg]")
                    return
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed_frames = []
                frame_count = 0
                
                try:
                    while frame_count < frames_to_process:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Resize frame
                        if (frame.shape[1] != width) or (frame.shape[0] != height):
                            frame = cv2.resize(frame, (width, height))
                        
                        # Process frame
                        stylized_frame = process_video_frame(model, device, frame, style_tensor, alpha)
                        processed_frames.append(stylized_frame)
                        
                        frame_count += 1
                        progress = frame_count / frames_to_process
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{frames_to_process}")
                    
                    cap.release()
                    
                    if not processed_frames:
                        st.error("❌ No frames were processed successfully.")
                        return
                    
                    st.info(f"🎬 Creating video from {len(processed_frames)} processed frames...")
                    
                    # Create browser-compatible video
                    video_path, temp_paths = create_browser_compatible_video(processed_frames, fps, width, height)
                    
                    if video_path and os.path.exists(video_path):
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("🖼️ Style Reference")
                            st.image(style_image, use_container_width=True)
                        
                        with col2:
                            st.subheader("🎥 Original Video")
                            st.video(temp_input_path)
                        
                        with col3:
                            st.subheader("🎨 Stylized Video")
                            
                            try:
                                with open(video_path, 'rb') as f:
                                    video_bytes = f.read()
                                
                                # Try multiple display methods
                                display_success = display_video_with_fallbacks(video_path, video_bytes)
                                
                                if display_success:
                                    st.success("✅ Video processed successfully!")
                                else:
                                    st.warning("⚠️ Video created but display failed. Download should work.")
                                
                                # Always provide download button
                                file_extension = os.path.splitext(video_path)[1]
                                st.download_button(
                                    label=f"⬇️ Download Stylized Video ({file_extension})",
                                    data=video_bytes,
                                    file_name=f"stylized_video{file_extension}",
                                    mime=f"video/{file_extension[1:]}" if file_extension != '.webm' else "video/webm"
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error handling output video: {e}")
                    
                    else:
                        st.error("❌ Failed to create output video with any method.")
                        st.info("💡 Try reducing the resolution or number of frames.")
                
                except Exception as e:
                    st.error(f"❌ Error during video processing: {e}")
                
                finally:
                    # Cleanup
                    if cap is not None:
                        cap.release()
                    
                    # Clean up all temporary files
                    all_temp_files = [temp_input_path] + temp_paths
                    for path in all_temp_files:
                        if os.path.exists(path):
                            try:
                                os.unlink(path)
                            except:
                                pass
        
        else:
            # Image processing (unchanged)
            st.header("🖼️ Image Style Transfer")
            
            content_image = Image.open(content_file).convert('RGB')
            content_tensor = preprocess_image(content_image).to(device)
            
            if st.button("🎨 Apply Style Transfer", type="primary"):
                with st.spinner("Applying style transfer..."):
                    with torch.no_grad():
                        stylized = model.generate(content_tensor, style_tensor, alpha)
                    stylized_image = postprocess_image(stylized, device)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🖼️ Style Reference")
                    st.image(style_image, use_container_width=True)
                
                with col2:
                    st.subheader("📷 Original Content")
                    st.image(content_image, use_container_width=True)
                
                with col3:
                    st.subheader("🎨 Stylized Result")
                    st.image(stylized_image, use_container_width=True)
                    
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
        # Instructions
        st.header("🚀 How to Use")
        st.markdown("""
        1. **Install Dependencies**: Make sure you have imageio installed:
           ```bash
           pip install imageio[ffmpeg]
           ```
        
        2. **Upload Style Image**: Choose an artistic style image
        
        3. **Upload Content**: Choose image or video to stylize
        
        4. **Adjust Settings**: Control style strength and video quality
        
        5. **Process**: Click to apply style transfer
        
        6. **Download**: Save your result
        
        ### 🎥 Video Processing Notes:
        - Limited to first 100 frames by default (adjustable)
        - Lower resolution = faster processing
        - Multiple video formats supported (MP4, WebM, AVI)
        - Automatic fallback to most compatible format
        """)
        


if __name__ == "__main__":
    main()

