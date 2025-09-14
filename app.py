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
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.info("Please ensure the model URL is correct and accessible.")
            return None, device

    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        st.success("✅ Pre-trained model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device
    
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

def convert_video_for_web(input_path, output_path):
    """Convert video to web-compatible format using ffmpeg if available"""
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        
        # Convert to H.264 with web-compatible settings
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',  # Enable web streaming
            '-pix_fmt', 'yuv420p',  # Ensure compatibility
            '-y',  # Overwrite output file
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_web_compatible_video(input_path, fps, width, height):
    """Create a web-compatible video writer"""
    # Try different codec options in order of preference
    codec_options = [
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ]
    
    for codec_name, fourcc in codec_options:
        try:
            writer = cv2.VideoWriter(input_path, fourcc, fps, (width, height))
            if writer.isOpened():
                return writer, codec_name
            writer.release()
        except Exception as e:
            continue
    
    return None, None

def main():
    st.title("🎨 AdaIN Style Transfer App")
    st.markdown("Transform your images and videos with artistic styles using Adaptive Instance Normalization!")
    
    # Load model
    model_result = load_model()
    if model_result[0] is None:
        st.error("❌ Could not load the model. Please check the model path and try again.")
        return
    
    model, device = model_result
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    alpha = st.sidebar.slider("Style Strength", 0.0, 1.0, 1.0, 0.1, 
                             help="0 = Original content, 1 = Full style transfer")
    
    # Video quality settings
    st.sidebar.subheader("🎥 Video Settings")
    max_video_size = st.sidebar.selectbox(
        "Max Video Resolution",
        options=[512, 720, 1080],
        index=0,
        help="Higher resolution = better quality but slower processing"
    )
    
    skip_frames = st.sidebar.number_input(
        "Process Every N Frames",
        min_value=1,
        max_value=10,
        value=1,
        help="Skip frames to speed up processing (1 = process all frames)"
    )
    
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
            
            # Calculate output dimensions
            aspect_ratio = original_width / original_height
            if original_width > original_height:
                width = min(max_video_size, original_width)
                height = int(width / aspect_ratio)
            else:
                height = min(max_video_size, original_height)
                width = int(height * aspect_ratio)
            
            # Ensure dimensions are even (required for some codecs)
            width = width if width % 2 == 0 else width - 1
            height = height if height % 2 == 0 else height - 1
            
            # Show video info
            st.info(f"📊 Video Info: {original_width}x{original_height} → {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
            
            if st.button("🎬 Process Video", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create temporary output path
                temp_output_path = tempfile.mktemp(suffix='.mp4')
                final_output_path = tempfile.mktemp(suffix='.mp4')
                
                # Create video writer
                writer, codec_used = create_web_compatible_video(temp_output_path, fps, width, height)
                
                if writer is None:
                    st.error("❌ Could not create video writer. Please try a different video format.")
                    cap.release()
                    os.unlink(temp_input_path)
                    return
                
                st.info(f"🎥 Using {codec_used} codec for video encoding")
                
                frame_count = 0
                processed_frames = 0
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Skip frames if specified
                        if frame_count % skip_frames == 0:
                            # Resize frame if needed
                            if (frame.shape[1] != width) or (frame.shape[0] != height):
                                frame = cv2.resize(frame, (width, height))
                            
                            # Process frame
                            stylized_frame = process_video_frame(model, device, frame, style_tensor, alpha)
                            writer.write(stylized_frame)
                            processed_frames += 1
                        else:
                            # For skipped frames, write the last processed frame or original
                            if processed_frames > 0:
                                writer.write(stylized_frame)
                            else:
                                if (frame.shape[1] != width) or (frame.shape[0] != height):
                                    frame = cv2.resize(frame, (width, height))
                                writer.write(frame)
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames} (Processed: {processed_frames})")
                    
                    cap.release()
                    writer.release()
                    
                    # Verify the output file exists and has content
                    if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
                        st.error("❌ Failed to create output video file.")
                        return
                    
                    # Try to convert to web-compatible format using ffmpeg
                    conversion_success = convert_video_for_web(temp_output_path, final_output_path)
                    
                    # Use the best available output
                    display_path = final_output_path if conversion_success and os.path.exists(final_output_path) else temp_output_path
                    
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
                            # Read video bytes for display and download
                            with open(display_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            if len(video_bytes) > 0:
                                # Display video
                                st.video(video_bytes)
                                
                                # Download button
                                st.download_button(
                                    label="⬇️ Download Stylized Video",
                                    data=video_bytes,
                                    file_name="stylized_video.mp4",
                                    mime="video/mp4"
                                )
                                
                                if conversion_success:
                                    st.success("✅ Video processed and optimized for web!")
                                else:
                                    st.success("✅ Video processed successfully!")
                                    st.info("💡 For better web compatibility, consider installing ffmpeg")
                            else:
                                st.error("❌ Processed video file is empty")
                                
                        except Exception as e:
                            st.error(f"❌ Error reading processed video: {e}")
                            st.info("💡 The video processing may have failed. Please try with a different video or smaller resolution.")
                
                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")
                    st.info("💡 Try reducing the video resolution or using a different video format.")
                
                finally:
                    # Cleanup
                    if cap is not None:
                        cap.release()
                    if writer is not None:
                        writer.release()
                    
                    # Clean up temporary files
                    for path in [temp_input_path, temp_output_path, final_output_path]:
                        if os.path.exists(path):
                            try:
                                os.unlink(path)
                            except:
                                pass
        
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
        # Show examples and instructions
        st.header("🚀 How to Use")
        st.markdown("""
        1. **Upload a Style Image**: Choose an image with the artistic style you want to apply (vintage, modern, painterly, etc.)
        2. **Upload Content**: Choose the image or video you want to stylize
        3. **Adjust Settings**: Use the sidebar to control style strength and video quality
        4. **Process**: Click the button to apply style transfer
        5. **Download**: Save your stylized result!
        
        ### 🎨 Style Examples:
        - **Vintage**: Old photographs, sepia tones, film grain
        - **Modern**: Contemporary art, bold colors, geometric patterns
        - **Artistic**: Famous paintings, watercolors, oil paintings
        - **Abstract**: Non-representational art with interesting textures and colors
        
        ### 🎥 Video Processing Tips:
        - Lower resolution = faster processing
        - Skip frames option speeds up processing for long videos
        - Install ffmpeg for best video compatibility
        """)
        
        st.header("ℹ️ About AdaIN Style Transfer")
        st.markdown("""
        This app uses **Adaptive Instance Normalization (AdaIN)** for real-time style transfer. 
        AdaIN transfers style by aligning the mean and variance of content features to match those of style features.
        
        - ✅ Fast processing
        - ✅ Preserves content structure  
        - ✅ Flexible style control
        - ✅ Works with images and videos
        - ✅ Multiple video codec support
        - ✅ Web-optimized output
        """)

        # System info
        st.header("🖥️ System Info")
        device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"🔧 Processing Device: {device_info}")
        
        if device_info == "CPU":
            st.warning("⚠️ Running on CPU. Video processing will be slower. Consider using a GPU for better performance.")

if __name__ == "__main__":
    main()
