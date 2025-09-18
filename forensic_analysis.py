#!/usr/bin/env python3
"""
Script untuk analisis forensik gambar menggunakan MMFusion-IML
Deteksi dan lokalisasi pemalsuan gambar.
Versi ini telah dimodifikasi untuk memungkinkan impor modular.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize

# Setup path untuk import modul
sys.path.append('.')

try:
    from models.cmnext_conf import CMNeXtWithConf
    from models.modal_extract import ModalitiesExtractor
    from configs.cmnext_init_cfg import _C as config, update_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Pastikan Anda berada di direktori root MMFusion-IML")
    sys.exit(1)

def load_image(image_path):
    """Load dan preprocess gambar"""
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Analisis gambar dasar (output ke console)
        print("=== ANALISIS GAMBAR (MM-Fusion) ===")
        print(f"Format: {img.format}, Ukuran: {img.size}, Mode: {img.mode}")
        
        try:
            exif_data = img._getexif()
            if exif_data:
                print(f"EXIF tags ditemukan: {len(exif_data)}")
            else:
                print("Tidak ada metadata EXIF")
        except:
            print("Tidak dapat membaca metadata EXIF")
        
        return img
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def setup_models():
    """Setup model untuk inference"""
    try:
        # Load configuration
        cfg = update_config(config, 'experiments/ec_example_phase2.yaml')
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Menggunakan device: {device}")
        
        # Load models
        modal_extractor = ModalitiesExtractor(cfg.MODEL.MODALS[1:], cfg.MODEL.NP_WEIGHTS)
        model = CMNeXtWithConf(cfg.MODEL)
        
        # Load weights
        ckpt_path = 'ckpt/early_fusion_detection.pth'
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint tidak ditemukan: {ckpt_path}")
            return None, None, None, None
            
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        model.load_state_dict(ckpt['state_dict'])
        modal_extractor.load_state_dict(ckpt['extractor_state_dict'])
        
        model.to(device)
        modal_extractor.to(device)
        
        model.eval()
        modal_extractor.eval()
        
        return model, modal_extractor, device, cfg
        
    except Exception as e:
        print(f"Error setting up models: {e}")
        return None, None, None, None

def analyze_forgery(image_path, model, modal_extractor, device, cfg):
    """Analisis gambar untuk deteksi pemalsuan"""
    try:
        # Load and preprocess image
        img = load_image(image_path)
        if img is None:
            return None, None, None
        
        # Transform image
        transform = transforms.Compose([
            transforms.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        print("\n=== PROSES ANALISIS FORENSIK (MM-Fusion) ===")
        
        # Extract modalities
        print("Mengekstrak fitur modalitas...")
        modals = modal_extractor(img_tensor)
        
        # Normalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_norm = (img_tensor - mean) / std
        
        # Prepare input
        inp = [images_norm] + modals
        
        # Run inference
        print("Menjalankan analisis deep learning MM-Fusion...")
        with torch.no_grad():
            anomaly, confidence, detection = model(inp)
        
        # Process results
        det_score = detection.item()
        heatmap = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
        
        print(f"\n=== HASIL ANALISIS (MM-Fusion) ===")
        print(f"Skor deteksi pemalsuan: {det_score:.4f}")
        
        return det_score, heatmap, confidence
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def visualize_results(original_img, heatmap, det_score, image_path, output_dir='results'):
    """Visualisasi hasil analisis"""
    os.makedirs(output_dir, exist_ok=True)

    # Dapatkan dimensi gambar asli
    original_width, original_height = original_img.size

    # Ubah ukuran heatmap agar sesuai dengan ukuran gambar asli untuk overlay
    # Menggunakan interpolasi INTER_CUBIC untuk hasil yang lebih halus
    resized_heatmap = cv2.resize(heatmap, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

    plt.figure(figsize=(15, 6))

    # 1. Gambar Original
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Gambar Original', fontsize=12, fontweight='bold')
    plt.axis('off')

    # 2. Heatmap Asli (resolusi rendah)
    plt.subplot(1, 3, 2)
    im = plt.imshow(heatmap, cmap='RdBu_r', vmin=0, vmax=1)
    plt.title('Heatmap Deteksi Manipulasi', fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # 3. Overlay dengan heatmap yang sudah diubah ukurannya
    plt.subplot(1, 3, 3)
    plt.imshow(original_img, alpha=0.7)
    # Gunakan heatmap yang diubah ukurannya untuk overlay
    plt.imshow(resized_heatmap, cmap='RdBu_r', alpha=0.5, vmin=0, vmax=1)
    plt.title(f'Overlay (Score: {det_score:.3f})', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f'{base_name}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    heatmap_path = os.path.join(output_dir, f'{base_name}_heatmap.png')
    # Simpan heatmap asli (resolusi rendah) untuk konsistensi
    plt.imsave(heatmap_path, heatmap, cmap='RdBu_r', vmin=0, vmax=1)

    # Fungsi show() dinonaktifkan agar tidak menampilkan GUI saat diimpor
    # plt.show()
    plt.close() # Menutup figure untuk membebaskan memori

    return output_path, heatmap_path

def generate_report(det_score, image_path, output_paths):
    """Generate laporan hasil analisis (console output)"""
    print("\n" + "="*60)
    print("LAPORAN ANALISIS FORENSIK GAMBAR (MM-Fusion)")
    print("="*60)
    print(f"\n- Gambar Dianalisis: {os.path.basename(image_path)}")
    print(f"- Skor Kepercayaan: {det_score:.4f}")
    
    if det_score > 0.7:
        print("- Status: TINGGI - Kemungkinan besar gambar dimanipulasi.")
    elif det_score > 0.4:
        print("- Status: SEDANG - Ditemukan indikasi manipulasi.")
    else:
        print("- Status: RENDAH - Kemungkinan besar gambar asli.")
    
    print("\n- File Hasil Dibuat:")
    for name, path in output_paths.items():
        print(f"  - {name}: {path}")
    print("="*60)

def run_mmfusion_analysis(image_path):
    """
    Fungsi pembungkus untuk menjalankan pipeline analisis forensik MM-Fusion.
    Mengembalikan path ke visualisasi, skor deteksi, dan heatmap mentah.
    """
    print("Memulai analisis forensik gambar dengan MM-Fusion...")

    # 1. Setup models
    model, modal_extractor, device, cfg = setup_models()
    if model is None:
        print("Gagal setup model MM-Fusion.")
        return None, None, None

    # 2. Load original image
    try:
        original_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Gagal membuka gambar untuk visualisasi: {e}")
        return None, None, None

    # 3. Run analysis
    det_score, heatmap, confidence = analyze_forgery(image_path, model, modal_extractor, device, cfg)

    # 4. Visualize and save results
    if det_score is not None and heatmap is not None:
        output_path, heatmap_path = visualize_results(original_img, heatmap, det_score, image_path)
        
        # Generate console report
        output_paths = {"Visualisasi Lengkap": output_path, "Heatmap": heatmap_path}
        generate_report(det_score, image_path, output_paths)
        
        # Kembalikan path, skor, dan heatmap mentah untuk digunakan di aplikasi lain
        return output_path, det_score, heatmap
    else:
        print("Analisis MM-Fusion gagal.")
        return None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analisis Forensik Gambar menggunakan MM-Fusion')
    parser.add_argument('image_path', help='Path ke gambar yang akan dianalisis')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: File gambar tidak ditemukan: {args.image_path}")
        sys.exit(1)
        
    analysis_result_path, score, _ = run_mmfusion_analysis(args.image_path)
    
    if analysis_result_path:
        print(f"\nAnalisis selesai. Visualisasi utama disimpan di: {analysis_result_path}")
    else:
        print("\nAnalisis tidak berhasil diselesaikan.")
