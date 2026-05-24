import os
import glob
import numpy as np
import torch
from PIL import Image
import urllib.request

def download_bsd500():
    """Download BSD500 images from a reliable source"""
    os.makedirs("dataset/Train400", exist_ok=True)
    
    # Use scikit-image built-in + generate synthetic training patches
    try:
        from skimage import data as skdata
        from skimage import color, transform
        
        # Get all available test images from scikit-image
        img_funcs = [
            skdata.camera, skdata.astronaut, skdata.horse,
            skdata.coffee, skdata.cat, skdata.coins,
            skdata.clock, skdata.moon, skdata.page,
            skdata.text, skdata.chelsea, skdata.hubble_deep_field,
]
        
        count = 0
        for func in img_funcs:
            try:
                img = func()
                if len(img.shape) == 3:
                    img = color.rgb2gray(img)
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                Image.fromarray(img).save(f"dataset/Train400/{count:04d}.png")
                count += 1
            except:
                pass
        print(f"Got {count} images from scikit-image")
    except ImportError:
        print("scikit-image not available, skipping built-in images")

    # Download from Martin's BSD dataset (Berkeley)
    bsd_url = "https://www2.ecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    tgz_path = "BSDS300-images.tgz"
    
    if not os.path.exists(tgz_path):
        print(f"Downloading BSD300 from Berkeley...")
        try:
            urllib.request.urlretrieve(bsd_url, tgz_path)
            print("Download complete.")
        except Exception as e:
            print(f"Berkeley download failed: {e}")
            # Fallback: use only scikit-image data
            return
    
    if os.path.exists(tgz_path):
        import tarfile
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(".")
        # Move images to Train400
        src_dirs = [
            "BSDS300/images/train",
            "BSDS300/images/test",
        ]
        count = len(glob.glob("dataset/Train400/*.png"))
        for src_dir in src_dirs:
            if os.path.exists(src_dir):
                for img_path in glob.glob(src_dir + "/*.*"):
                    try:
                        img = Image.open(img_path).convert('L')
                        img.save(f"dataset/Train400/{count:04d}.png")
                        count += 1
                    except:
                        pass
        print(f"Total images after BSD300: {count}")


def augment_with_transforms(img_list, target_count=400):
    """Augment images with flips and rotations to reach target count"""
    augmented = []
    for img_path in img_list:
        img = Image.open(img_path).convert('L')
        augmented.append(np.array(img, dtype=np.float32))
        # Flip horizontal
        augmented.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.float32))
        # Rotate 90
        augmented.append(np.array(img.transpose(Image.ROTATE_90), dtype=np.float32))
        # Rotate 180
        augmented.append(np.array(img.transpose(Image.ROTATE_180), dtype=np.float32))
        if len(augmented) >= target_count:
            break
    return augmented


def generate_train_data(patch_size=33, stride=14):
    """Generate train_data.pt from images"""
    train_path = "dataset/Train400"
    img_list = sorted(glob.glob(train_path + "/*.png") + 
                      glob.glob(train_path + "/*.jpg") +
                      glob.glob(train_path + "/*.bmp"))
    
    if len(img_list) == 0:
        print("ERROR: Still no images. Generating synthetic data...")
        generate_synthetic_data(patch_size)
        return
    
    print(f"Found {len(img_list)} images")
    # Augment if too few
    images = augment_with_transforms(img_list, 400)
    print(f"After augmentation: {len(images)} images")
    
    patches = []
    for img_arr in images:
        img_arr = img_arr / 255.0
        h, w = img_arr.shape
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = img_arr[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
    
    patches = np.array(patches, dtype=np.float32)
    patches_tensor = torch.from_numpy(patches)
    
    print(f"Generated {len(patches_tensor)} patches of size {patch_size}x{patch_size}")
    
    save_path = "dataset/train_data.pt"
    torch.save((patches_tensor, patches_tensor), save_path)
    print(f"Saved to {save_path}")
    
    # Verify
    data, labels = torch.load(save_path)
    print(f"Verification: {len(data)} samples, shape: {data[0].shape}")


def generate_synthetic_data(patch_size=33):
    """Last resort: generate enough synthetic natural-like patches"""
    print("Generating synthetic training patches...")
    np.random.seed(42)
    
    num_patches = 5000
    patches = []
    
    for i in range(num_patches):
        # Create patches with natural image statistics (1/f noise)
        freq = np.fft.fftfreq(patch_size)
        fx, fy = np.meshgrid(freq, freq)
        power = 1.0 / (np.sqrt(fx**2 + fy**2) + 0.01)
        phase = np.random.uniform(0, 2*np.pi, (patch_size, patch_size))
        spectrum = power * np.exp(1j * phase)
        patch = np.real(np.fft.ifft2(spectrum))
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-10)
        patches.append(patch.astype(np.float32))
    
    patches_tensor = torch.from_numpy(np.array(patches))
    save_path = "dataset/train_data.pt"
    torch.save((patches_tensor, patches_tensor), save_path)
    print(f"Generated {num_patches} synthetic patches, saved to {save_path}")


if __name__ == "__main__":
    if os.path.exists("dataset/train_data.pt"):
        os.remove("dataset/train_data.pt")
    download_bsd500()
    img_count = len(glob.glob("dataset/Train400/*.png") + 
                    glob.glob("dataset/Train400/*.jpg"))
    print(f"\nTotal images in Train400: {img_count}")
    
    generate_train_data()