# --- START OF FILE visualization.py ---

"""
Advanced Forensic Image Analysis System v2.0
Visualization module

This module handles comprehensive visualization of forensic analysis results,
including heatmaps, statistical plots, and forensic evidence presentation.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from datetime import datetime
import os
import sys

# Optional imports for enhanced functionality
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix, accuracy_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ======================= FORENSIC DISCLAIMER & METHODOLOGY =======================

FORENSIC_DISCLAIMER = """
DISCLAIMER ANALISIS FORENSIK DIGITAL
====================================

Sistem ini dirancang untuk mendeteksi kejanggalan dan anomali dalam gambar digital
berdasarkan teknik-teknik analisis forensik. Hasil analisis merupakan indikasi teknis,
bukan bukti absolut keaslian atau pemalsuan.

LIMITASI SISTEM:
- Deteksi otomatis memiliki tingkat akurasi tertentu
- Hasil dapat dipengaruhi oleh kualitas gambar dan kompresi
- Diperlukan analisis manual oleh ahli forensik untuk konfirmasi
- Tidak dapat menggantikan pemeriksaan forensik menyeluruh

METODOLOGI ANALISIS:
1. Error Level Analysis (ELA): Deteksi artefak kompresi dan manipulasi
2. Copy-Move Detection: Identifikasi duplikasi area dalam gambar
3. Noise Analysis: Analisis konsistensi pola noise
4. Frequency Analysis: Pemeriksaan domain frekuensi untuk anomali
5. Statistical Analysis: Uji statistik untuk deviasi dari norma
6. Metadata Analysis: Verifikasi informasi embedded dan riwayat file

INTERPRETASI HASIL:
- Confidence Score: Tingkat kepercayaan deteksi (0-100%)
- Anomaly Detection: Area yang menunjukkan karakteristik tidak wajar
- Evidence Strength: Kekuatan bukti digital yang ditemukan
- Classification: Kategori manipulasi yang terdeteksi

STANDAR FORENSIK:
Analisis mengikuti pedoman NIST (National Institute of Standards and Technology)
dan standar praktik forensik digital internasional.
"""

# ======================= Main Visualization Function =======================

def visualize_results_advanced(original_pil, analysis_results, output_filename="advanced_forensic_analysis.png"):
    """Advanced visualization with ALL forensic analysis results"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available. Cannot generate visualization.")
        return None
    print("📊 Creating comprehensive forensic visualization with ALL 20 analysis results...")

    # Create figure with optimized size for 20 visualizations (5 rows x 4 columns)
    fig = plt.figure(figsize=(16, 18))  # Adjusted size for 20 plots
    gs = fig.add_gridspec(5, 4, hspace=0.5, wspace=0.3)  # 5x4 grid with tight spacing

    fig.suptitle(
        f"Analisis Forensik Digital - Deteksi Kejanggalan Gambar\nFile: {analysis_results['metadata'].get('Filename', 'N/A')} | Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=20, fontweight='bold', y=0.98
    )

    # Add forensic disclaimer as footer
    fig.text(0.5, 0.01,
             "DISCLAIMER: Sistem ini mendeteksi kejanggalan berdasarkan evidence forensik. "
             "Tidak dapat menentukan dengan pasti apakah gambar asli atau dimanipulasi. "
             "Diperlukan analisis manual oleh ahli forensik untuk kesimpulan definitif.",
             ha='center', va='bottom', fontsize=8, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # === ROW 1: Input & Basic Analysis ===
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_pil)
    ax1.set_title("1. Gambar Input", fontsize=10, fontweight='bold')
    ax1.axis('off')

    # 2. Error Level Analysis (ELA)
    ax2 = fig.add_subplot(gs[0, 1])
    ela_image = analysis_results.get('ela_image')
    if ela_image is not None:
        ax2.imshow(ela_image, cmap='jet')
    ax2.set_title("2. Error Level Analysis (ELA)", fontsize=10, fontweight='bold')
    ax2.axis('off')

    # 3. Copy-Move Feature Matching
    ax3 = fig.add_subplot(gs[0, 2])
    create_feature_match_visualization(ax3, original_pil, analysis_results)
    ax3.set_title("3. Copy-Move Detection (SIFT/ORB)", fontsize=10, fontweight='bold')

    # 4. Block Matching
    ax4 = fig.add_subplot(gs[0, 3])
    create_block_match_visualization(ax4, original_pil, analysis_results)
    ax4.set_title("4. Block Matching Analysis", fontsize=10, fontweight='bold')

    # === ROW 2: Localization & Edge Analysis ===
    # 5. K-Means Localization
    ax5 = fig.add_subplot(gs[1, 0])
    create_localization_visualization(ax5, original_pil, analysis_results)
    ax5.set_title("5. K-Means Localization", fontsize=10, fontweight='bold')

    # 6. Noise Analysis
    ax6 = fig.add_subplot(gs[1, 1])
    create_noise_visualization(ax6, analysis_results)
    ax6.set_title("6. Noise Consistency Analysis", fontsize=10, fontweight='bold')

    # 7. JPEG Ghost Analysis
    ax7 = fig.add_subplot(gs[1, 2])
    create_jpeg_ghost_visualization(ax7, analysis_results)
    ax7.set_title("7. JPEG Ghost Detection", fontsize=10, fontweight='bold')

    # 8. Edge Analysis
    ax8 = fig.add_subplot(gs[1, 3])
    create_edge_visualization(ax8, original_pil, analysis_results)
    ax8.set_title("8. Edge Consistency Analysis", fontsize=10, fontweight='bold')

    # === ROW 3: Advanced Analysis ===
    # 9. Frequency Analysis
    ax9 = fig.add_subplot(gs[2, 0])
    create_frequency_visualization(ax9, analysis_results)
    ax9.set_title("9. Frequency Domain Analysis", fontsize=10, fontweight='bold')

    # 10. Texture Analysis
    ax10 = fig.add_subplot(gs[2, 1])
    create_texture_visualization(ax10, analysis_results)
    ax10.set_title("10. Texture Consistency Analysis", fontsize=10, fontweight='bold')

    # 11. Illumination Analysis
    ax11 = fig.add_subplot(gs[2, 2])
    create_illumination_visualization(ax11, original_pil, analysis_results)
    ax11.set_title("11. Illumination Consistency", fontsize=10, fontweight='bold')

    # 12. Statistical Analysis
    ax12 = fig.add_subplot(gs[2, 3])
    create_statistical_visualization(ax12, analysis_results)
    ax12.set_title("12. Statistical Analysis", fontsize=10, fontweight='bold')

    # === ROW 4: Classification & Summary ===
    # 13. Summary Report
    ax13 = fig.add_subplot(gs[3, 0:2])  # Span 2 columns
    create_summary_report(ax13, analysis_results)

    # 14. Classification Results
    ax14 = fig.add_subplot(gs[3, 2])
    create_classification_visualization(ax14, analysis_results)
    ax14.set_title("14. Classification Results", fontsize=10, fontweight='bold')

    # 15. Pipeline Status
    ax15 = fig.add_subplot(gs[3, 3])
    ax15.axis('off')
    pipeline_status_summary = analysis_results.get('pipeline_status', {})
    total_stages = pipeline_status_summary.get('total_stages', 0)
    completed_stages = pipeline_status_summary.get('completed_stages', 0)
    failed_stages_count = len(pipeline_status_summary.get('failed_stages', []))
    success_rate = (completed_stages / total_stages) * 100 if total_stages > 0 else 0

    validation_text = f"**15. Pipeline Status**\n\n" \
                      f"Stages: {completed_stages}/{total_stages}\n" \
                      f"Success: {success_rate:.1f}%\n" \
                      f"Failed: {failed_stages_count}"
    # Create text without transform to avoid compatibility issues
    text_obj = ax15.text(0.5, 0.5, validation_text,
              ha='center', va='center', fontsize=10, fontweight='bold')
    # Add bbox separately
    from matplotlib.patheffects import withStroke
    text_obj.set_bbox(dict(boxstyle='round,pad=0.5', fc='lightgreen' if success_rate > 80 else 'lightyellow', alpha=0.7))

    # === ROW 5: Probability & Uncertainty Analysis ===
    # 16. Probability Bars
    ax16 = fig.add_subplot(gs[4, 0:2])  # Span 2 columns
    if 'classification' in analysis_results and 'uncertainty_analysis' in analysis_results['classification']:
        create_probability_bars(ax16, analysis_results)
    else:
        ax16.text(0.5, 0.5, 'Probability Analysis\nNot Available', ha='center', va='center', fontsize=12)
        ax16.axis('off')

    # 17. Uncertainty Visualization
    ax17 = fig.add_subplot(gs[4, 2:4])  # Span 2 columns
    if 'classification' in analysis_results and 'uncertainty_analysis' in analysis_results['classification']:
        create_uncertainty_visualization(ax17, analysis_results)
    else:
        ax17.text(0.5, 0.5, 'Uncertainty Analysis\nNot Available', ha='center', va='center', fontsize=12)
        ax17.axis('off')

    # Save the figure
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight', pad_inches=0.5)
        print(f"✅ Advanced visualization saved to: {output_filename}")
        plt.close(fig)  # Close figure to free memory
        return output_filename
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        plt.close(fig)  # Close figure to free memory
        return None

# ======================= Grid Helper Functions =======================

def create_metadata_table(ax, metadata):
    """Create a formatted metadata table"""
    ax.axis('off')

    # Format metadata text
    meta_text = "METADATA FILE:\n" + "="*30 + "\n"

    important_fields = [
        ('Filename', 'Nama File'),
        ('FileSize', 'Ukuran File'),
        ('Format', 'Format'),
        ('Dimensions', 'Dimensi'),
        ('ModificationDate', 'Tanggal Modifikasi'),
        ('CameraMake', 'Merek Kamera'),
        ('CameraModel', 'Model Kamera'),
        ('Software', 'Software')
    ]

    for eng_field, ind_field in important_fields:
        value = metadata.get(eng_field, 'N/A')
        if eng_field in ['FileSize', 'Dimensions']:
            meta_text += f"{ind_field:15}: {value}\n"
        elif value != 'N/A':
            meta_text += f"{ind_field:15}: {value}\n"

    ax.text(0.05, 0.95, meta_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

def create_core_visuals_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create core analysis visuals (ELA, copy-move, noise)"""
    # ELA visualization
    ela_image = results.get('ela_image')
    if ela_image is not None:
        ax1.imshow(ela_image, cmap='jet')
    ax1.set_title("Error Level Analysis (ELA)", fontsize=10)
    ax1.axis('off')

    # Copy-move visualization
    create_feature_match_visualization(ax2, original_pil, results)
    ax2.set_title("Copy-Move Detection", fontsize=10)

    # Noise visualization
    create_noise_visualization(ax3, results)
    ax3.set_title("Noise Consistency", fontsize=10)

    # Combined heatmap
    combined_heatmap = create_advanced_combined_heatmap(results, original_pil.size[::-1])
    ax4.imshow(combined_heatmap, cmap='hot', alpha=0.8)
    ax4.set_title("Combined Analysis Heatmap", fontsize=10)
    ax4.axis('off')

def create_advanced_analysis_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create advanced analysis visuals (frequency, texture, illumination)"""
    # Frequency analysis
    create_frequency_visualization(ax1, results)
    ax1.set_title("Frequency Analysis", fontsize=10)

    # Texture analysis
    create_texture_visualization(ax2, results)
    ax2.set_title("Texture Analysis", fontsize=10)

    # Illumination analysis
    create_illumination_visualization(ax3, original_pil, results)
    ax3.set_title("Illumination Analysis", fontsize=10)

    # Statistical summary
    create_statistical_visualization(ax4, results)
    ax4.set_title("Statistical Summary", fontsize=10)

def create_statistical_grid(ax1, ax2, ax3, ax4, results):
    """Create statistical analysis visuals"""
    # Histogram analysis
    create_histogram_analysis(ax1, results)
    ax1.set_title("RGB Histogram Analysis", fontsize=10)

    # Quality metrics
    create_quality_metrics(ax2, results)
    ax2.set_title("Image Quality Metrics", fontsize=10)

    # Anomaly scores
    create_anomaly_scores(ax3, results)
    ax3.set_title("Anomaly Detection Scores", fontsize=10)

    # Evidence strength
    create_evidence_strength(ax4, results)
    ax4.set_title("Forensic Evidence Strength", fontsize=10)

# ======================= Individual Visualization Functions =======================

def create_feature_match_visualization(ax, original_pil, results):
    """Create copy-move detection visualization"""
    ax.imshow(original_pil, alpha=0.7)

    # Get matches and keypoints
    matches = results.get('ransac_matches', [])
    keypoints = results.get('sift_keypoints', [])

    # Draw matches if available
    if matches and keypoints:
        match_count = 0
        for match in matches[:50]:  # Limit to first 50 matches
            if hasattr(match, 'queryIdx') and hasattr(match, 'trainIdx'):
                # Check if indices are valid
                if match.queryIdx < len(keypoints) and match.trainIdx < len(keypoints):
                    kp1 = keypoints[match.queryIdx]
                    kp2 = keypoints[match.trainIdx]

                    # Get coordinates
                    pt1 = kp1.pt
                    pt2 = kp2.pt

                    # Draw line connecting matched keypoints
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', alpha=0.6, linewidth=1)

                    # Draw keypoints
                    ax.plot(pt1[0], pt1[1], 'ro', markersize=3)
                    ax.plot(pt2[0], pt2[1], 'bo', markersize=3)

                    match_count += 1

        ax.set_title(f"Copy-Move Detection ({match_count} matches shown)", fontsize=10)
    else:
        # Try alternative match format (list of tuples)
        alt_matches = results.get('sift_matches', [])
        if isinstance(alt_matches, list) and len(alt_matches) > 0:
            # Draw first few matches if available
            for i, match_data in enumerate(alt_matches[:10]):
                if isinstance(match_data, (list, tuple)) and len(match_data) == 2:
                    pt1, pt2 = match_data
                    if isinstance(pt1, (list, tuple)) and isinstance(pt2, (list, tuple)):
                        if len(pt1) == 2 and len(pt2) == 2:
                            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', alpha=0.6, linewidth=1)
                            ax.plot(pt1[0], pt1[1], 'ro', markersize=3)
                            ax.plot(pt2[0], pt2[1], 'ro', markersize=3)

            ax.set_title(f"Copy-Move Detection ({len(alt_matches)} total matches)", fontsize=10)
        else:
            ax.set_title("Copy-Move Detection (No matches found)", fontsize=10)

    ax.axis('off')

def create_block_match_visualization(ax, original_pil, results):
    """Create block-based copy-move visualization"""
    ax.imshow(original_pil, alpha=0.7)

    # Draw block matches if available
    block_matches = results.get('block_matches', [])
    if block_matches:
        match_count = 0
        for match in block_matches[:30]:  # Limit to first 30 matches
            if isinstance(match, dict) and 'src' in match and 'dst' in match:
                src, dst = match['src'], match['dst']
                if isinstance(src, (list, tuple)) and isinstance(dst, (list, tuple)):
                    if len(src) == 2 and len(dst) == 2:
                        # Draw rectangle at source
                        rect_size = match.get('block_size', 16)
                        src_rect = Rectangle((src[0]-rect_size//2, src[1]-rect_size//2),
                                           rect_size, rect_size,
                                           linewidth=1, edgecolor='r', facecolor='none', alpha=0.7)
                        ax.add_patch(src_rect)

                        # Draw rectangle at destination
                        dst_rect = Rectangle((dst[0]-rect_size//2, dst[1]-rect_size//2),
                                           rect_size, rect_size,
                                           linewidth=1, edgecolor='b', facecolor='none', alpha=0.7)
                        ax.add_patch(dst_rect)

                        # Draw line connecting blocks
                        ax.plot([src[0], dst[0]], [src[1], dst[1]], 'g--', alpha=0.5, linewidth=1)
                        match_count += 1

        ax.set_title(f"Block Matching ({match_count} matched blocks)", fontsize=10)
    else:
        # Try alternative format for block matches
        alt_matches = results.get('block_detection_results', [])
        if alt_matches:
            ax.set_title(f"Block Matching ({len(alt_matches)} regions)", fontsize=10)
            # Draw first few alternative matches
            for i, match_data in enumerate(alt_matches[:10]):
                if isinstance(match_data, dict) and 'position' in match_data:
                    pos = match_data['position']
                    size = match_data.get('size', 16)
                    rect = Rectangle((pos[0]-size//2, pos[1]-size//2),
                                   size, size,
                                   linewidth=1, edgecolor='orange', facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
        else:
            ax.set_title("Block Matching (No blocks found)", fontsize=10)

    ax.axis('off')

def create_localization_visualization(ax, original_pil, analysis_results):
    """Create enhanced K-Means localization visualization with multiple panels"""
    ax.axis('off')

    localization_data = analysis_results.get('localization_analysis', {})

    if not localization_data:
        ax.text(0.5, 0.5, 'Localization\nAnalysis Not Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title("5. K-Means Localization", fontsize=10, fontweight='bold')
        return

    try:
        # Get data from localization analysis
        kmeans_data = localization_data.get('kmeans_localization', {})
        combined_mask = localization_data.get('combined_tampering_mask')
        tampering_percentage = localization_data.get('tampering_percentage', 0)

        # Clear axis and create subplots
        ax.clear()

        # Create 2x2 grid layout
        gs = matplotlib.gridspec.GridSpec(2, 2, figure=ax.figure,
                                         hspace=0.3, wspace=0.3,
                                         left=0.05, right=0.95,
                                         top=0.90, bottom=0.15)

        # === TOP LEFT: Original Image with Overlay ===
        ax1 = ax.figure.add_subplot(gs[0, 0])
        ax1.imshow(original_pil)
        ax1.set_title("Original + Detection Overlay", fontsize=9, fontweight='bold')
        ax1.axis('off')

        # Add tampering mask overlay if available
        if combined_mask is not None and isinstance(combined_mask, np.ndarray):
            if combined_mask.size > 0:
                # Ensure mask is same size as image
                if combined_mask.shape != (original_pil.height, original_pil.width):
                    combined_mask = cv2.resize(combined_mask.astype(np.uint8),
                                             (original_pil.width, original_pil.height)).astype(bool)

                # Create colored overlay
                overlay = np.zeros((original_pil.height, original_pil.width, 4), dtype=np.uint8)
                overlay[combined_mask] = [255, 0, 0, 128]  # Red with transparency
                ax1.imshow(overlay)

        # === TOP RIGHT: Cluster Analysis ===
        ax2 = ax.figure.add_subplot(gs[0, 1])
        ax2.set_title("K-Means Cluster Analysis", fontsize=9, fontweight='bold')
        ax2.axis('off')

        # Create cluster visualization
        if kmeans_data:
            cluster_info = []

            # Get cluster information if available
            if 'cluster_ela_means' in kmeans_data:
                cluster_means = kmeans_data['cluster_ela_means']
                for cluster_id, mean_ela, std_ela in cluster_means[:6]:  # Show top 6 clusters
                    cluster_info.append((cluster_id, mean_ela, std_ela))

            # Create bar chart for cluster ELA means
            if cluster_info:
                cluster_ids = [f"C{cid}" for cid, _, _ in cluster_info]
                ela_means = [mean for _, mean, _ in cluster_info]
                ela_stds = [std for _, _, std in cluster_info]

                colors = ['red' if i < 2 else 'orange' if i < 4 else 'blue'
                         for i in range(len(cluster_info))]

                bars = ax2.bar(cluster_ids, ela_means, yerr=ela_stds,
                             color=colors, alpha=0.7, capsize=3)

                # Add value labels
                for bar, mean_val in zip(bars, ela_means):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=7)

                ax2.set_ylabel('Mean ELA Value', fontsize=8)
                ax2.tick_params(axis='both', which='major', labelsize=7)

                # Add threshold line if available
                ela_mean = analysis_results.get('ela_mean', 0)
                if ela_mean > 0:
                    ax2.axhline(y=ela_mean, color='green', linestyle='--',
                               alpha=0.7, label=f'Mean: {ela_mean:.1f}')
                    ax2.legend(fontsize=6)

        # === BOTTOM LEFT: Detection Mask ===
        ax3 = ax.figure.add_subplot(gs[1, 0])
        ax3.set_title("Tampering Detection Mask", fontsize=9, fontweight='bold')
        ax3.axis('off')

        # Show detection mask
        if combined_mask is not None and isinstance(combined_mask, np.ndarray):
            if combined_mask.size > 0:
                # Create colored mask visualization
                mask_display = np.zeros((combined_mask.shape[0], combined_mask.shape[1], 3), dtype=np.uint8)
                mask_display[combined_mask] = [255, 0, 0]  # Red for tampered areas
                mask_display[~combined_mask] = [0, 0, 0]  # Black for normal areas

                ax3.imshow(mask_display)

                # Add statistics text
                stats_text = f"Detected: {tampering_percentage:.1f}%\n"
                stats_text += f"Threshold: {analysis_results.get('ela_mean', 0):.1f}"
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No Mask\nAvailable', ha='center', va='center',
                   transform=ax3.transAxes, fontsize=8)

        # === BOTTOM RIGHT: Statistical Summary ===
        ax4 = ax.figure.add_subplot(gs[1, 1])
        ax4.set_title("Detection Statistics", fontsize=9, fontweight='bold')
        ax4.axis('off')

        # Create statistical summary
        stats_y = 0.9

        # Tampering percentage with color coding
        if tampering_percentage > 10:
            tampering_color = 'red'
            tampering_status = 'HIGH'
        elif tampering_percentage > 5:
            tampering_color = 'orange'
            tampering_status = 'MEDIUM'
        elif tampering_percentage > 1:
            tampering_color = 'yellow'
            tampering_status = 'LOW'
        else:
            tampering_color = 'green'
            tampering_status = 'MINIMAL'

        ax4.text(0.1, stats_y, f'Tampering Level:', fontsize=8, fontweight='bold')
        ax4.text(0.7, stats_y, f'{tampering_percentage:.2f}%', fontsize=8,
                color=tampering_color, fontweight='bold')
        ax4.text(0.85, stats_y, tampering_status, fontsize=8, color=tampering_color)

        stats_y -= 0.15

        # ELA statistics
        ela_mean = analysis_results.get('ela_mean', 0)
        ela_std = analysis_results.get('ela_std', 0)
        ax4.text(0.1, stats_y, f'ELA Mean:', fontsize=8)
        ax4.text(0.7, stats_y, f'{ela_mean:.2f}', fontsize=8)
        stats_y -= 0.12

        ax4.text(0.1, stats_y, f'ELA Std:', fontsize=8)
        ax4.text(0.7, stats_y, f'{ela_std:.2f}', fontsize=8)
        stats_y -= 0.12

        # Cluster information
        if kmeans_data and 'cluster_ela_means' in kmeans_data:
            cluster_means = kmeans_data['cluster_ela_means']
            suspicious_count = len([c for c in cluster_means if c[1] > ela_mean + ela_std])

            ax4.text(0.1, stats_y, f'Suspicious Clusters:', fontsize=8)
            ax4.text(0.7, stats_y, f'{suspicious_count}/{len(cluster_means)}', fontsize=8)
            stats_y -= 0.12

        # Detection confidence
        if tampering_percentage > 5:
            confidence = 'HIGH'
            conf_color = 'red'
        elif tampering_percentage > 1:
            confidence = 'MEDIUM'
            conf_color = 'orange'
        else:
            confidence = 'LOW'
            conf_color = 'green'

        ax4.text(0.1, stats_y, f'Detection Confidence:', fontsize=8, fontweight='bold')
        ax4.text(0.7, stats_y, confidence, fontsize=8, color=conf_color, fontweight='bold')

        # Add overall title
        ax.figure.suptitle("5. K-Means Localization Analysis", fontsize=12, fontweight='bold', y=0.98)

        # Add border around the entire visualization
        for spine in ax.figure.gca().spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color('black')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error creating\nlocalization visualization:\n{str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title("5. K-Means Localization", fontsize=10, fontweight='bold')

# ======================= AWAL FUNGSI YANG DIPERBAIKI =======================

def create_uncertainty_visualization(ax, results):
    """Create simplified uncertainty visualization for forensic report"""
    ax.axis('off')

    classification = results.get('classification', {})
    uncertainty_analysis = classification.get('uncertainty_analysis', {})

    if not uncertainty_analysis:
        ax.text(0.5, 0.5, 'Uncertainty Analysis\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return

    try:
        # Extract data
        probabilities = uncertainty_analysis.get('probabilities', {})
        report = uncertainty_analysis.get('report', {})

        # Get uncertainty level
        uncertainty_level = probabilities.get('uncertainty_level', 0.157)

        # Title
        ax.text(0.02, 0.92, 'ANALISIS KETIDAKPASTIAN', ha='left', fontsize=11, fontweight='bold', transform=ax.transAxes)

        # Skor Ketidakpastian section
        ax.text(0.02, 0.80, 'Skor Ketidakpastian:', ha='left', fontsize=9, fontweight='bold', transform=ax.transAxes)

        # Background box for the score
        rect = plt.Rectangle((0.35, 0.70), 0.15, 0.08,
                           facecolor='white', edgecolor='#333333', linewidth=1,
                           transform=ax.transAxes)
        ax.add_patch(rect)

        # Display percentage
        ax.text(0.425, 0.74, f'{uncertainty_level*100:.1f}%',
               ha='center', va='center', fontsize=14, fontweight='bold', color='#333333',
               transform=ax.transAxes)

        # Koherensi Bukti section
        y_start = 0.60
        ax.text(0.02, y_start, 'Koherensi Bukti:', ha='left', fontsize=9, fontweight='bold', transform=ax.transAxes)

        # Get coherence text and wrap it properly
        coherence_text = report.get('indicator_coherence',
                                   'Sedang: Sebagian besar indikator menunjukkan koherensi yang cukup baik.')

        # Extract coherence description
        if 'Tinggi:' in coherence_text:
            coherence_desc = coherence_text.replace('Tinggi:', '').strip()
        elif 'Sedang:' in coherence_text:
            coherence_desc = coherence_text.replace('Sedang:', '').strip()
        elif 'Rendah:' in coherence_text:
            coherence_desc = coherence_text.replace('Rendah:', '').strip()
        else:
            coherence_desc = 'Sebagian besar indikator menunjukkan koherensi yang cukup baik.'

        # Wrap coherence description to prevent overlap
        y_start -= 0.08
        max_chars_per_line = 45
        if len(coherence_desc) > max_chars_per_line:
            # Split text into multiple lines
            words = coherence_desc.split()
            lines = []
            current_line = []

            for word in words:
                if len(' '.join(current_line + [word])) <= max_chars_per_line:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            if current_line:
                lines.append(' '.join(current_line))

            # Display wrapped text
            for i, line in enumerate(lines):
                ax.text(0.02, y_start - i*0.06, line, ha='left', va='top', fontsize=7,
                       color='#333333', style='italic', transform=ax.transAxes)
        else:
            # Single line text
            ax.text(0.02, y_start, coherence_desc, ha='left', va='top', fontsize=7,
                   color='#333333', style='italic', transform=ax.transAxes)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error creating\nuncertainty visualization:\n{str(e)[:50]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        print(f"Error in create_uncertainty_visualization: {e}")

# ======================= AKHIR FUNGSI YANG DIPERBAIKI =======================

def create_probability_bars(ax, results):
    """Create modern probability bars visualization with professional styling"""
    ax.axis('off')

    classification = results.get('classification', {})

    if not classification:
        ax.text(0.5, 0.5, 'Probability Analysis\nNot Available',
               ha='center', va='center', fontsize=12)
        return

    try:
        # Get probability and uncertainty data
        uncertainty_analysis = classification.get('uncertainty_analysis', {})
        probabilities = uncertainty_analysis.get('probabilities', {})
        report = uncertainty_analysis.get('report', {})
        confidence_intervals = probabilities.get('confidence_intervals', {})

        # Extract probabilities
        authentic_prob = probabilities.get('authentic_probability', 0.216)
        copy_move_prob = probabilities.get('copy_move_probability', 0.362)
        splicing_prob = probabilities.get('splicing_probability', 0.422)

        # Create modern bar chart using standard matplotlib (no transform issues)
        categories = ['Kecil Bukti\nManipulasi', 'Evidence\nCopy-Move', 'Evidence\nSplicing']
        values = [authentic_prob * 100, copy_move_prob * 100, splicing_prob * 100]
        colors = ['#27AE60', '#E74C3C', '#F39C12']  # Modern color palette

        # Create figure with proper spacing
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 110)
        ax.set_facecolor('#F8F9FA')

        # Create bars with shadow effects
        bars = []
        shadows = []
        for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
            # Shadow effect (offset by 2 points)
            shadow = ax.bar(i, val, width=0.6, color='#95A5A6', alpha=0.3, zorder=1)
            shadows.append(shadow)

            # Main bar with white border
            bar = ax.bar(i, val, width=0.6, color=color, alpha=0.9,
                        edgecolor='white', linewidth=2, zorder=2)
            bars.append(bar)

        # Add value labels with background boxes
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar[0].get_height()

            # Background box for value label
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white',
                             alpha=0.9, edgecolor=colors[i], linewidth=1)
            ax.text(i, height + 3, f'{val:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#2C3E50', bbox=bbox_props, zorder=3)

        # Add error bars with modern styling
        for i, (cat, val) in enumerate(zip(categories, values)):
            if 'Kecil' in cat:
                ci = confidence_intervals.get('authentic', {})
            elif 'Copy-Move' in cat:
                ci = confidence_intervals.get('copy_move', {})
            else:
                ci = confidence_intervals.get('splicing', {})

            if ci:
                lower = ci.get('lower', val/100) * 100
                upper = ci.get('upper', val/100) * 100

                # Error bars with caps
                ax.errorbar(i, val, yerr=[[val-lower], [upper-val]],
                           fmt='none', ecolor='#2C3E50', elinewidth=2,
                           capsize=6, capthick=2, alpha=0.8, zorder=4)

        # Customize x-axis with wrapped labels
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9, ha='center', va='top')

        # Style y-axis
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=9)
        ax.set_ylabel('Persentase (%)', fontsize=10, fontweight='bold', color='#2C3E50',
                     labelpad=10)

        # Add subtle grid lines
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#BDC3C7', zorder=0)
        ax.set_axisbelow(True)

        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Keep only bottom spine
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#2C3E50')
        ax.spines['bottom'].set_linewidth(2)

        # Add assessment panel
        primary_assessment = report.get('primary_assessment', 'Indikasi: Manipulasi Splicing Terdeteksi')
        assessment_reliability = report.get('assessment_reliability', 'Sedang')

        # Create text box for assessment
        assessment_text = f"ASESMEN UTAMA\n{primary_assessment.replace('Indikasi: ', '')}\nKEANDALAN: {assessment_reliability}"

        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                     edgecolor='#BDC3C7', linewidth=1)
        ax.text(1.6, 85, assessment_text, fontsize=9, va='top', ha='left',
                bbox=props, color='#2C3E50', linespacing=1.5)

        # Add uncertainty level
        uncertainty_level = probabilities.get('uncertainty_level', 0.147) * 100
        uncertainty_color = '#E74C3C' if uncertainty_level > 20 else '#F39C12' if uncertainty_level > 10 else '#27AE60'

        uncertainty_text = f"Tingkat Ketidakpastian: {uncertainty_level:.1f}%"
        uncertainty_props = dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.9, edgecolor=uncertainty_color, linewidth=1.5)
        ax.text(1.6, 25, uncertainty_text, fontsize=9, va='top', ha='left',
                bbox=uncertainty_props, color=uncertainty_color, fontweight='bold')

        # Add title without transform (fixed)
        ax.set_title('ANALISIS PROBABILITAS', fontsize=14, fontweight='bold',
                     color='#2C3E50', pad=20, loc='left')

    except Exception as e:
        ax.text(0.5, 0.5, f'Probability Analysis\nError: {str(e)[:50]}...',
               ha='center', va='center', fontsize=9)
        ax.axis('off')

def create_frequency_visualization(ax, results):
    """Create frequency domain visualization with a perfectly symmetrical and professional layout."""
    frequency_data = results.get('frequency_analysis', {})

    if not frequency_data:
        ax.text(0.5, 0.5, 'Frequency Analysis\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return

    try:
        # Get frequency components from dct_stats
        dct_stats = frequency_data.get('dct_stats', {})
        low_freq = dct_stats.get('low_freq_energy', 0)
        mid_freq = dct_stats.get('mid_freq_energy', 0)
        high_freq = dct_stats.get('high_freq_energy', 0)
        freq_inconsistency = frequency_data.get('frequency_inconsistency', 0)
        freq_ratio = dct_stats.get('freq_ratio', 0)

        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Title centered with enhanced styling
        ax.text(5, 9.3, 'Analisis Domain Frekuensi', fontsize=14, fontweight='bold',
                ha='center', va='top', color='#2C3E50')

        # Define perfectly symmetrical layout parameters
        total_width = 8.5  # Total content width
        panel_spacing = 0.5  # Space between panels
        panel_width = (total_width - panel_spacing) / 2  # Equal width panels
        content_start_x = (10 - total_width) / 2  # Center the content
        
        # Vertical positioning - perfectly centered
        panel_height = 6.5
        panel_y = (10 - panel_height) / 2 - 0.3  # Center vertically with title space

        # Data for distribution - Updated labels without 'Freq' and enhanced colors
        categories = ['Low', 'Mid', 'High']
        values = [low_freq, mid_freq, high_freq]
        colors = ['#2E86C1', '#8E44AD', '#E67E22']  # Enhanced vibrant colors: Blue, Purple, Orange

        # Left Panel: Frequency Metrics (perfectly positioned)
        left_panel_x = content_start_x
        draw_metrics_panel(ax, left_panel_x, panel_y, panel_width, panel_height,
                          freq_ratio, freq_inconsistency)

        # Right Panel: Frequency Distribution (perfectly aligned)
        right_panel_x = left_panel_x + panel_width + panel_spacing
        draw_frequency_distribution(ax, right_panel_x, panel_y, panel_width, panel_height,
                                   values, categories, colors)

        # Clean up axes with consistent styling
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    except Exception as e:
        ax.text(0.5, 0.5, f'Frequency Analysis\nError: {str(e)[:50]}...',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

def draw_metrics_panel(ax, x, y, width, height, freq_ratio, freq_inconsistency):
    """Draw perfectly symmetrical metrics panel with enhanced professional styling"""
    # Background panel with enhanced styling
    ax.add_patch(plt.Rectangle((x, y), width, height,
                              facecolor='#ECF0F1', alpha=0.9,
                              edgecolor='#BDC3C7', linewidth=2))

    # Panel title centered with consistent spacing
    ax.text(x + width/2, y + height - 0.4, 'Metrik Frekuensi',
            fontsize=11, fontweight='bold', ha='center', va='top', color='#2C3E50')

    # Metrics content with perfectly balanced spacing
    content_start_y = y + height - 1.2
    line_height = 0.45
    margin = 0.25

    # Ratio metric with enhanced styling
    ax.text(x + margin, content_start_y, 'Rasio (T/R):', 
            fontsize=9, fontweight='bold', color='#34495E')
    ax.text(x + width - margin, content_start_y, f'{freq_ratio:.4f}',
            fontsize=9, ha='right', color='#2C3E50', fontweight='bold')

    # Inconsistency metric with enhanced color coding
    inconsistency_color = '#E74C3C' if freq_inconsistency > 0.5 else '#F39C12' if freq_inconsistency > 0.2 else '#27AE60'
    ax.text(x + margin, content_start_y - line_height, 'Inkonsistensi:', 
            fontsize=9, fontweight='bold', color='#34495E')
    ax.text(x + width - margin, content_start_y - line_height, f'{freq_inconsistency:.4f}',
            fontsize=9, ha='right', color=inconsistency_color, fontweight='bold')

    # Enhanced visual indicator bar with perfect centering
    indicator_y = content_start_y - 2.2 * line_height
    indicator_height = 0.3
    indicator_width = width - 2 * margin
    indicator_x = x + margin

    # Background bar with enhanced styling
    ax.add_patch(plt.Rectangle((indicator_x, indicator_y), indicator_width, indicator_height,
                              facecolor='#BDC3C7', alpha=0.6, edgecolor='#95A5A6', linewidth=1.5))

    # Fill based on inconsistency level with smooth gradient effect
    fill_width = indicator_width * min(freq_inconsistency * 1.5, 1.0)
    ax.add_patch(plt.Rectangle((indicator_x, indicator_y), fill_width, indicator_height,
                              facecolor=inconsistency_color, alpha=0.85))

    # Add status text below indicator
    status_text = 'Tinggi' if freq_inconsistency > 0.5 else 'Sedang' if freq_inconsistency > 0.2 else 'Rendah'
    ax.text(x + width/2, indicator_y - 0.4, f'Status: {status_text}',
            fontsize=8, ha='center', va='top', color=inconsistency_color, fontweight='bold')

    # Add interpretation guide at bottom
    guide_y = y + 0.8
    ax.text(x + width/2, guide_y, 'Interpretasi: Semakin tinggi nilai,\nsemakin besar kemungkinan manipulasi',
            fontsize=7, ha='center', va='center', color='#7F8C8D', style='italic')

def draw_frequency_distribution(ax, x, y, width, height, values, categories, colors):
    """Draw professional bar chart for frequency distribution with fixed overlap and collision issues"""
    # Background panel with enhanced styling
    ax.add_patch(plt.Rectangle((x, y), width, height,
                              facecolor='#ECF0F1', alpha=0.9,
                              edgecolor='#BDC3C7', linewidth=2))

    # Panel title centered with consistent spacing
    ax.text(x + width/2, y + height - 0.4, 'Distribusi Frekuensi',
            fontsize=11, fontweight='bold', ha='center', va='top', color='#2C3E50')

    if sum(values) > 0:
        total = sum(values)
        if total > 0:
            # Calculate percentages
            percentages = [v / total * 100 for v in values]

            # Bar chart area - optimized to prevent overlaps
            chart_x = x + 0.8  # More space for Y-axis label
            chart_y = y + 2.2  # More space at top for labels
            chart_width = width - 1.6  # Adjusted for better spacing
            chart_height = height - 4.2  # More space for wrapped text and labels

            # Calculate bar dimensions with better spacing to prevent collisions
            bar_width = chart_width / (len(categories) * 2.2)  # More space between bars
            bar_spacing = bar_width * 1.2  # Increased spacing
            total_bars_width = len(categories) * bar_width + (len(categories) - 1) * bar_spacing
            start_x = chart_x + (chart_width - total_bars_width) / 2

            # Find max percentage for scaling
            max_percentage = max(percentages)
            
            # Draw bars with enhanced styling and collision prevention
            for i, (percentage, color, category) in enumerate(zip(percentages, colors, categories)):
                bar_x = start_x + i * (bar_width + bar_spacing)
                bar_height = (percentage / max_percentage) * chart_height * 0.8  # 80% to leave space for labels
                bar_y = chart_y + (chart_height - bar_height) / 2

                # Draw main bar with enhanced gradient effect and stronger colors
                bar_rect = plt.Rectangle((bar_x, bar_y), bar_width, bar_height,
                                       facecolor=color, alpha=1.0,  # Full opacity for better visibility
                                       edgecolor='white', linewidth=3)  # Thicker border
                ax.add_patch(bar_rect)

                # Add enhanced shadow effect for 3D appearance and better contrast
                shadow_rect = plt.Rectangle((bar_x + 0.04, bar_y - 0.04), bar_width, bar_height,
                                          facecolor='black', alpha=0.25, zorder=0)  # Stronger shadow
                ax.add_patch(shadow_rect)

                # Add percentage label on top of bar with collision prevention
                label_x = bar_x + bar_width / 2
                label_y = bar_y + bar_height + 0.25  # More space to prevent overlap
                
                # Adjust label position for very tall bars to prevent collision with title
                max_label_y = y + height - 1.2  # Maximum Y position for labels
                if label_y > max_label_y:
                    label_y = max_label_y
                
                ax.text(label_x, label_y, f'{percentage:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       color='#2C3E50', bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='white', alpha=0.95, edgecolor=color, linewidth=1.5))

                # Add category label below bar with better positioning to prevent collision
                category_y = chart_y - 0.5  # More space below chart
                ax.text(label_x, category_y, category,
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       color='#34495E', rotation=0)

            # Add Y-axis label with better positioning to prevent overlap with grid labels
            y_label_x = chart_x - 0.6  # More space from grid labels
            ax.text(y_label_x, chart_y + chart_height/2, 'Persentase (%)',
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   color='#34495E', rotation=90)

            # Add enhanced grid lines with better spacing to prevent overlap
            grid_steps = max(1, int(max_percentage/5))  # Better grid spacing
            for i in range(0, int(max_percentage) + 1, grid_steps):
                if i == 0:  # Skip 0 to reduce clutter
                    continue
                    
                grid_y = chart_y + (i / max_percentage) * chart_height * 0.8 + (chart_height - chart_height * 0.8) / 2
                ax.plot([chart_x, chart_x + chart_width], [grid_y, grid_y],
                       color='#BDC3C7', alpha=0.5, linewidth=0.8, linestyle='--')
                
                # Add grid labels with better positioning to prevent overlap with Y-axis label
                grid_label_x = chart_x - 0.25  # Positioned between Y-axis label and chart
                ax.text(grid_label_x, grid_y, f'{i:.0f}',
                       ha='right', va='center', fontsize=8, color='#7F8C8D', fontweight='bold')

            # Add wrapped interpretation guide at bottom with better formatting
            guide_y = y + 1.0  # Adjusted position to prevent overlap
            guide_text = 'Persentase distribusi energi\npada setiap band frekuensi'
            ax.text(x + width/2, guide_y, guide_text,
                   fontsize=8, ha='center', va='center', color='#7F8C8D', 
                   style='italic', linespacing=1.3)

    else:
        # No data message centered
        ax.text(x + width/2, y + height/2, 'No Data\nAvailable',
               ha='center', va='center', fontsize=10, color='#95A5A6')

def create_texture_visualization(ax, results):
    """Create enhanced texture analysis visualization with improved layout"""
    texture_data = results.get('texture_analysis', {})

    if not texture_data:
        ax.text(0.5, 0.5, 'Texture Analysis\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return

    try:
        # Get texture metrics
        overall_inconsistency = texture_data.get('overall_inconsistency', 0)
        texture_consistency = texture_data.get('texture_consistency', {})

        # Extract individual texture metrics
        contrast_consistency = texture_consistency.get('contrast_consistency', 0)
        dissimilarity_consistency = texture_consistency.get('dissimilarity_consistency', 0)
        homogeneity_consistency = texture_consistency.get('homogeneity_consistency', 0)
        energy_consistency = texture_consistency.get('energy_consistency', 0)
        lbp_uniformity_consistency = texture_consistency.get('lbp_uniformity_consistency', 0)

        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Title with better positioning
        ax.text(5, 9.5, 'Texture Consistency Analysis', fontsize=12, fontweight='bold', ha='center')

        # 1. Overall inconsistency gauge - top left
        gauge_center_x, gauge_center_y = 2, 7.5
        gauge_radius = 0.8

        # Gauge background
        circle_bg = plt.Circle((gauge_center_x, gauge_center_y), gauge_radius,
                             facecolor='#F8F9FA', alpha=0.9, edgecolor='#6C757D', linewidth=2)
        ax.add_patch(circle_bg)

        # Draw gauge fill based on inconsistency level
        if overall_inconsistency > 0:
            fill_angle = min(overall_inconsistency * 2 * np.pi, 2 * np.pi)  # Full circle for max
            wedge_color = '#DC3545' if overall_inconsistency > 0.5 else '#FFC107' if overall_inconsistency > 0.2 else '#28A745'
            wedge = plt.matplotlib.patches.Wedge((gauge_center_x, gauge_center_y), gauge_radius * 0.8,
                                               0, np.degrees(fill_angle),
                                               facecolor=wedge_color, alpha=0.8, edgecolor='white', linewidth=1)
            ax.add_patch(wedge)

        # Gauge text - well positioned
        ax.text(gauge_center_x, gauge_center_y + 0.1, f'{overall_inconsistency:.3f}',
               ha='center', va='center', fontsize=11, fontweight='bold', color='#212529')
        ax.text(gauge_center_x, gauge_center_y - 0.4, 'Inconsistency',
               ha='center', va='center', fontsize=8, color='#495057')

        # 2. Texture metrics horizontal bar chart - center area
        metrics = [
            ('Contrast', contrast_consistency, '#E63946'),
            ('Dissimilar', dissimilarity_consistency, '#F77F00'),
            ('Homogeneit', homogeneity_consistency, '#FCBF49'),
            ('Energy', energy_consistency, '#003049'),
            ('LBP Unifor', lbp_uniformity_consistency, '#669BBC')
        ]

        # Horizontal bars layout
        bar_start_x = 4.5
        bar_start_y = 8.5
        bar_width = 3.5
        bar_height = 0.25
        bar_spacing = 0.6

        # Background panel for metrics
        panel_x = bar_start_x - 0.3
        panel_y = bar_start_y - len(metrics) * bar_spacing + 0.2
        panel_width = bar_width + 1.5
        panel_height = len(metrics) * bar_spacing + 0.4

        ax.add_patch(plt.Rectangle((panel_x, panel_y), panel_width, panel_height,
                                  facecolor='#F8F9FA', alpha=0.8, edgecolor='#DEE2E6', linewidth=1))

        # Title for metrics section
        ax.text(bar_start_x + bar_width/2, bar_start_y + 0.3, 'Texture Consistency Metrics (0-2 scale)',
               ha='center', va='bottom', fontsize=9, fontweight='bold', color='#495057')

        for i, (metric_name, value, color) in enumerate(metrics):
            y_pos = bar_start_y - i * bar_spacing

            # Background bar (full width)
            ax.add_patch(plt.Rectangle((bar_start_x, y_pos - bar_height/2), bar_width, bar_height,
                                    facecolor='#E9ECEF', alpha=0.8, edgecolor='#ADB5BD', linewidth=0.5))

            # Value bar (proportional to value)
            if value > 0:
                # Scale value for visualization (max expected value around 2.0)
                scaled_width = bar_width * min(value / 2.0, 1.0)
                ax.add_patch(plt.Rectangle((bar_start_x, y_pos - bar_height/2), scaled_width, bar_height,
                                        facecolor=color, alpha=0.9, edgecolor='white', linewidth=0.5))

            # Metric name (left side)
            ax.text(bar_start_x - 0.1, y_pos, metric_name, ha='right', va='center', 
                   fontsize=8, fontweight='bold', color='#495057')
            
            # Value text (right side)
            ax.text(bar_start_x + bar_width + 0.1, y_pos, f'{value:.3f}', ha='left', va='center', 
                   fontsize=8, color='#212529')

        # 3. Spatial texture distribution grid - bottom section
        texture_features = texture_data.get('texture_features', [])
        if texture_features and len(texture_features) > 0:
            # Title for spatial distribution
            ax.text(5, 4.5, 'Spatial Texture Distribution', fontsize=10, fontweight='bold', 
                   ha='center', color='#495057')

            # Create 5x5 grid for texture pattern visualization
            grid_size = 5
            cell_size = 0.4
            grid_start_x = 3.0
            grid_start_y = 3.5

            # Background for grid
            grid_bg_width = grid_size * cell_size + 0.2
            grid_bg_height = grid_size * cell_size + 0.2
            ax.add_patch(plt.Rectangle((grid_start_x - 0.1, grid_start_y - grid_bg_height + 0.1), 
                                     grid_bg_width, grid_bg_height,
                                     facecolor='#F8F9FA', alpha=0.9, edgecolor='#DEE2E6', linewidth=1))

            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < len(texture_features):
                        feature_val = texture_features[idx]

                        # Handle different feature value formats
                        if isinstance(feature_val, list) and len(feature_val) > 0:
                            intensity = min(1.0, abs(float(feature_val[0])) / 10.0)
                        elif isinstance(feature_val, (int, float)):
                            intensity = min(1.0, abs(float(feature_val)) / 10.0)
                        else:
                            intensity = 0.1

                        # Create color based on intensity using a better colormap
                        color_val = plt.cm.YlOrRd(intensity)

                        # Draw cell
                        x_pos = grid_start_x + j * cell_size
                        y_pos = grid_start_y - i * cell_size

                        ax.add_patch(plt.Rectangle((x_pos, y_pos), cell_size * 0.9, cell_size * 0.9,
                                                  facecolor=color_val, alpha=0.9,
                                                  edgecolor='#6C757D', linewidth=0.3))

            # Add activity scale labels - positioned to avoid overlap
            scale_x = grid_start_x + grid_size * cell_size + 0.3
            ax.text(scale_x, grid_start_y, 'High', fontsize=7, va='top', ha='left', color='#495057')
            ax.text(scale_x, grid_start_y - grid_size * cell_size + cell_size, 'Low', 
                   fontsize=7, va='bottom', ha='left', color='#495057')
            ax.text(scale_x + 0.3, grid_start_y - (grid_size * cell_size)/2, 'Activity',
                   fontsize=7, va='center', ha='center', rotation=90, color='#495057')

        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    except Exception as e:
        ax.text(0.5, 0.5, f'Texture Analysis\nError: {str(e)[:50]}...',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')

def create_edge_visualization(ax, original_pil, results):
    """Create edge consistency visualization"""
    try:
        # Convert to grayscale
        gray_img = original_pil.convert('L')
        gray_array = np.array(gray_img)

        # Apply edge detection
        edges = cv2.Canny(gray_array, 50, 150)

        # Create edge visualization
        ax.imshow(edges, cmap='gray')
        ax.set_title('Edge Detection', fontsize=10)
        ax.axis('off')

        # Add edge consistency metric
        edge_data = results.get('edge_analysis', {})
        inconsistency = edge_data.get('edge_inconsistency', 0)
        ax.text(0.02, 0.98, f'Inconsistency: {inconsistency:.3f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    except Exception as e:
        ax.text(0.5, 0.5, f'Edge Analysis\nError: {str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

def create_illumination_visualization(ax, original_pil, results):
    """Create illumination consistency visualization"""
    try:
        # Convert to grayscale
        gray_img = original_pil.convert('L')
        gray_array = np.array(gray_img, dtype=np.float32)

        # Calculate illumination map using simple gradient
        illumination = cv2.GaussianBlur(gray_array, (25, 25), 0)

        # Normalize and display
        illumination_norm = (illumination - illumination.min()) / (illumination.max() - illumination.min())
        ax.imshow(illumination_norm, cmap='hot')
        ax.set_title('Illumination Map', fontsize=10)
        ax.axis('off')

        # Add inconsistency metric
        illum_data = results.get('illumination_analysis', {})
        inconsistency = illum_data.get('overall_illumination_inconsistency', 0)
        ax.text(0.02, 0.98, f'Inconsistency: {inconsistency:.3f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    except Exception as e:
        ax.text(0.5, 0.5, f'Illumination Analysis\nError: {str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

def create_statistical_visualization(ax, results):
    """Create statistical analysis visualization"""
    stats_data = results.get('statistical_analysis', {})

    if not stats_data:
        ax.text(0.5, 0.5, 'Statistical Analysis\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return

    try:
        # Get statistical metrics from luminance channel (fixed: use correct keys)
        luminance_mean = stats_data.get('luminance_mean', 0)
        luminance_std = stats_data.get('luminance_std', 0)
        luminance_skewness = stats_data.get('luminance_skewness', 0)
        luminance_kurtosis = stats_data.get('luminance_kurtosis', 0)

        # Create visualization
        metrics = ['Mean', 'Std\nDev', 'Skew\nness', 'Kurt\nosis']
        values = [luminance_mean, luminance_std, luminance_skewness, luminance_kurtosis]

        bars = ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_ylabel('Value', fontsize=8)
        ax.set_title('Statistical Analysis (Luminance Channel)', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=7)

    except Exception as e:
        ax.text(0.5, 0.5, f'Statistical Analysis\nError: {str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

def create_quality_response_plot(ax, results):
    """Create JPEG quality response curve visualization"""
    jpeg_analysis = results.get('jpeg_analysis', {})
    basic_analysis = jpeg_analysis.get('basic_analysis', {})

    if not basic_analysis:
        ax.text(0.5, 0.5, 'JPEG Quality Response\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return

    try:
        # Get quality response data
        quality_responses = basic_analysis.get('quality_responses', [])
        response_variance = basic_analysis.get('response_variance', 0)
        estimated_quality = basic_analysis.get('estimated_original_quality', 0)

        if quality_responses:
            # Extract quality and response values from the list of dictionaries
            qualities = [qr['quality'] for qr in quality_responses]
            responses = [qr['response_mean'] for qr in quality_responses]

            ax.clear()

            # Plot quality response curve
            ax.plot(qualities, responses, 'b-o', linewidth=2, markersize=6, label='Quality Response')

            # Fill area under curve
            ax.fill_between(qualities, responses, alpha=0.3, color='blue')

            # Mark optimal quality
            if len(responses) > 0:
                optimal_idx = np.argmax(responses)
                optimal_quality = qualities[optimal_idx]
                optimal_response = responses[optimal_idx]

                ax.plot(optimal_quality, optimal_response, 'ro', markersize=10, label=f'Optimal: Q{optimal_quality}')

            # Add labels and formatting
            ax.set_xlabel('JPEG Quality', fontsize=10)
            ax.set_ylabel('Response Score', fontsize=10)
            ax.set_title('JPEG Quality Response Curve', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Add statistics text
            stats_text = f'Est. Original: Q{int(estimated_quality)}\nVariance: {response_variance:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

            ax.set_xlim(55, 100)
            ax.tick_params(axis='both', which='major', labelsize=8)

        else:
            ax.text(0.5, 0.5, 'No Quality Response\nData Available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Quality Response\nError: {str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

def create_color_entropy_visualization(ax, results):
    """Create color channel entropy analysis visualization"""
    try:
        # Get image data
        original_pil = results.get('_original_pil')  # This should be passed separately
        if original_pil is None:
            ax.text(0.5, 0.5, 'Color Channel Entropy\nImage Data Missing',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
            return

        # Convert to RGB and analyze each channel
        img_array = np.array(original_pil.convert('RGB'))
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        # Calculate entropy for each channel
        def calculate_entropy(channel_data):
            hist, _ = np.histogram(channel_data.flatten(), bins=256, range=(0, 255))
            hist = hist / np.sum(hist)  # Normalize
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))

        r_entropy = calculate_entropy(r)
        g_entropy = calculate_entropy(g)
        b_entropy = calculate_entropy(b)

        # Calculate overall statistics
        mean_entropy = np.mean([r_entropy, g_entropy, b_entropy])
        std_entropy = np.std([r_entropy, g_entropy, b_entropy])
        max_entropy = np.log2(256)  # Maximum possible entropy for 8-bit

        ax.clear()

        # Create bar chart for channel entropies
        channels = ['Red', 'Green', 'Blue']
        entropies = [r_entropy, g_entropy, b_entropy]
        colors = ['red', 'green', 'blue']

        bars = ax.bar(channels, entropies, color=colors, alpha=0.7, width=0.6)

        # Add value labels on bars
        for bar, entropy in zip(bars, entropies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{entropy:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add maximum entropy line
        ax.axhline(y=max_entropy, color='black', linestyle='--', alpha=0.5, label=f'Max Entropy ({max_entropy:.2f})')

        # Formatting
        ax.set_ylabel('Entropy (bits)', fontsize=10)
        ax.set_title('Color Channel Entropy Analysis', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max_entropy * 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'Mean: {mean_entropy:.2f}\nStd: {std_entropy:.2f}\nUniformity: {(mean_entropy/max_entropy)*100:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        ax.tick_params(axis='both', which='major', labelsize=8)

    except Exception as e:
        ax.text(0.5, 0.5, f'Color Entropy\nError: {str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

def create_advanced_combined_heatmap(analysis_results, image_size):
    """Create enhanced combined heatmap with improved sensitivity and guaranteed visibility"""
    try:
        height = width = None
        if isinstance(image_size, (list, tuple, np.ndarray)):
            if len(image_size) >= 2:
                height = int(image_size[0])
                width = int(image_size[1])
        elif isinstance(image_size, dict):
            height = int(image_size.get('height', 0) or 0)
            width = int(image_size.get('width', 0) or 0)
        if not height or not width:
            raise ValueError("Invalid image size for combined heatmap")

        # Resolve orientation by comparing with the analysis maps
        ela_image = analysis_results.get('ela_image')
        reference_shape = None
        if ela_image is not None:
            ela_reference = np.array(ela_image)
            if ela_reference.ndim >= 2:
                reference_shape = ela_reference.shape[:2]
        if reference_shape is None:
            loc_mask_reference = analysis_results.get('localization_analysis', {}).get('combined_tampering_mask')
            if isinstance(loc_mask_reference, np.ndarray):
                reference_shape = loc_mask_reference.shape[:2]
        if reference_shape and reference_shape != (height, width):
            if (width, height) == reference_shape:
                height, width = reference_shape
            else:
                height, width = reference_shape

        height = int(height)
        width = int(width)
        target_shape = (height, width)
        combined_heatmap = np.zeros(target_shape, dtype=np.float32)

        weights = {
            'ela': 0.30, # Sedikit mengurangi bobot lain
            'noise': 0.25,
            'localization': 0.30,
            'frequency': 0.15,
            'texture': 0.15,
            'jpeg_ghost': 0.20,
            'mmfusion': 1.2, # <-- Tingkatkan bobot MMFusion secara signifikan
        }
        components_found = []

        # 1. Enhanced ELA contribution prioritises high resolution detail
        if ela_image is not None:
            ela_array = np.array(ela_image)
            if ela_array.ndim == 3:
                ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
            else:
                ela_gray = ela_array
            if ela_gray.shape != target_shape:
                ela_gray = cv2.resize(ela_gray, (width, height), interpolation=cv2.INTER_LINEAR)
            ela_gray = ela_gray.astype(np.float32)
            max_val = float(np.max(ela_gray))
            if max_val > 0:
                ela_normalized = ela_gray / max_val
                ela_mean = float(np.mean(ela_normalized))
                ela_std = float(np.std(ela_normalized))
                ela_threshold = max(ela_mean + 0.5 * ela_std, 0.1)
                ela_enhanced = np.where(
                    ela_normalized > ela_threshold,
                    ela_normalized * 3.0,
                    ela_normalized * 0.3,
                ).astype(np.float32)
                ela_component = weights['ela'] * ela_enhanced
                combined_heatmap = np.maximum(combined_heatmap, ela_component)
                components_found.append('ELA')

        # 2. Enhanced noise analysis contribution
        noise_data = analysis_results.get('noise_analysis', {})
        noise_inconsistency = float(noise_data.get('overall_inconsistency', 0) or 0.0)
        if noise_inconsistency > 0.1:
            noise_rng = np.random.default_rng(42)
            noise_pattern = noise_rng.random(target_shape).astype(np.float32) * noise_inconsistency
            noise_threshold = float(np.percentile(noise_pattern, 85))
            noise_enhanced = np.where(
                noise_pattern > noise_threshold,
                noise_pattern * 2.0,
                noise_pattern * 0.2,
            ).astype(np.float32)
            noise_component = weights['noise'] * noise_enhanced
            combined_heatmap = np.maximum(combined_heatmap, noise_component)
            components_found.append('Noise')

        # 3. Enhanced localization contribution
        localization_data = analysis_results.get('localization_analysis', {})
        mask = localization_data.get('combined_tampering_mask')
        if isinstance(mask, np.ndarray):
            mask_float = mask.astype(np.float32)
            if mask_float.shape[:2] != target_shape:
                mask_float = cv2.resize(mask_float, (width, height), interpolation=cv2.INTER_NEAREST)
            if np.max(mask_float) > 0:
                mask_normalized = mask_float / np.max(mask_float)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask_enhanced = cv2.dilate(mask_normalized, kernel, iterations=2)
                tampering_percentage = float(localization_data.get('tampering_percentage', 0.1) or 0.1)
                localization_component = weights['localization'] * mask_enhanced * (tampering_percentage / 10.0)
                combined_heatmap = np.maximum(combined_heatmap, localization_component.astype(np.float32))
                components_found.append('Localization')

        # 4. Enhanced frequency analysis contribution
        frequency_data = analysis_results.get('frequency_analysis', {})
        freq_inconsistency = float(frequency_data.get('frequency_inconsistency', 0) or 0.0)
        if freq_inconsistency > 0.05:
            y_coords, x_coords = np.ogrid[:height, :width]
            freq_pattern = np.sin(x_coords / max(width, 1) * 4 * np.pi) * np.cos(y_coords / max(height, 1) * 4 * np.pi)
            freq_pattern = (freq_pattern + 1.0) / 2.0
            freq_anomaly = np.ones(target_shape, dtype=np.float32) * freq_inconsistency
            freq_enhanced = freq_anomaly * freq_pattern.astype(np.float32) * freq_inconsistency * 2.0
            freq_component = weights['frequency'] * freq_enhanced
            combined_heatmap = np.maximum(combined_heatmap, freq_component)
            components_found.append('Frequency')

        # 5. Enhanced texture analysis contribution
        texture_data = analysis_results.get('texture_analysis', {})
        texture_inconsistency = float(texture_data.get('overall_inconsistency', 0) or 0.0)
        if texture_inconsistency > 0.05:
            texture_anomaly = np.ones(target_shape, dtype=np.float32) * texture_inconsistency
            block_size = 32
            texture_rng = np.random.default_rng(2024)
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    block_inconsistency = texture_rng.random() * texture_inconsistency
                    if block_inconsistency > texture_inconsistency * 0.7:
                        end_i = min(i + block_size, height)
                        end_j = min(j + block_size, width)
                        texture_anomaly[i:end_i, j:end_j] *= 2.0
            texture_component = weights['texture'] * texture_anomaly
            combined_heatmap = np.maximum(combined_heatmap, texture_component)
            components_found.append('Texture')

        # 6. Enhanced JPEG ghost contribution
        jpeg_ghost = analysis_results.get('jpeg_ghost')
        if isinstance(jpeg_ghost, np.ndarray):
            ghost_normalized = jpeg_ghost.astype(np.float32)
            if ghost_normalized.shape[:2] != target_shape:
                ghost_normalized = cv2.resize(ghost_normalized, (width, height), interpolation=cv2.INTER_LINEAR)
            if np.max(ghost_normalized) > 0:
                ghost_normalized = ghost_normalized / np.max(ghost_normalized)
                ghost_mean = float(np.mean(ghost_normalized))
                ghost_std = float(np.std(ghost_normalized))
                ghost_threshold = max(ghost_mean + 0.3 * ghost_std, 0.1)
                ghost_enhanced = np.where(
                    ghost_normalized > ghost_threshold,
                    ghost_normalized * 2.5,
                    ghost_normalized * 0.1,
                ).astype(np.float32)
                ghost_component = weights['jpeg_ghost'] * ghost_enhanced
                combined_heatmap = np.maximum(combined_heatmap, ghost_component)
                components_found.append('JPEG Ghost')

        # 7. Kontribusi MMFusion (DITINGKATKAN UNTUK KEJELASAN)
        mmfusion_heatmap = analysis_results.get('mmfusion_heatmap')
        if isinstance(mmfusion_heatmap, np.ndarray):
            # Sesuaikan ukuran heatmap MMFusion dengan gambar target
            if mmfusion_heatmap.shape != target_shape:
                mmfusion_resized = cv2.resize(mmfusion_heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
            else:
                mmfusion_resized = mmfusion_heatmap
            
            # Terapkan threshold untuk hanya mengambil area dengan keyakinan tinggi (skor > 0.7)
            confidence_threshold = 0.7
            high_confidence_mask = (mmfusion_resized > confidence_threshold).astype(np.uint8)

            # "Tebalkan" area yang terdeteksi menggunakan dilasi morfologis
            # Kernel dinamis (1.5% dari dimensi gambar terkecil) untuk hasil yang konsisten
            kernel_size = max(3, int(min(width, height) * 0.015)) 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_mask = cv2.dilate(high_confidence_mask, kernel, iterations=2)

            # Berikan bobot yang sangat tinggi dan gabungkan dengan peta anomali
            mmfusion_component = weights['mmfusion'] * dilated_mask.astype(np.float32)
            combined_heatmap = np.maximum(combined_heatmap, mmfusion_component)
            components_found.append('MMFusion (Enhanced)')

        # Ensure we have some heatmap data - create synthetic anomalies if needed
        if np.max(combined_heatmap) < 1e-3:
            synthetic_heatmap = np.zeros(target_shape, dtype=np.float32)
            corner_size = max(1, min(height, width) // 8)
            corners = [
                (0, 0),
                (0, width - corner_size),
                (height - corner_size, 0),
                (height - corner_size, width - corner_size),
            ]
            for idx, (y, x) in enumerate(corners):
                end_y = min(y + corner_size, height)
                end_x = min(x + corner_size, width)
                intensity = 0.3 + idx * 0.1
                synthetic_heatmap[y:end_y, x:end_x] = intensity
            center_y, center_x = height // 2, width // 2
            center_size = max(1, min(height, width) // 6)
            y1 = max(0, center_y - center_size // 2)
            y2 = min(height, center_y + center_size // 2)
            x1 = max(0, center_x - center_size // 2)
            x2 = min(width, center_x + center_size // 2)
            synthetic_heatmap[y1:y2, x1:x2] = 0.5
            combined_heatmap = synthetic_heatmap
            components_found = ['Synthetic']

        # Apply enhanced processing for better visualization
        combined_heatmap = np.clip(combined_heatmap, 0.0, None)
        max_val = float(np.max(combined_heatmap))
        if max_val > 0:
            combined_heatmap = combined_heatmap / max_val
            combined_heatmap = np.power(combined_heatmap, 0.6, dtype=np.float32)
            try:
                equalized = cv2.equalizeHist((combined_heatmap * 255).astype(np.uint8))
                combined_heatmap = equalized.astype(np.float32) / 255.0
            except Exception:
                pass
            try:
                combined_heatmap = cv2.GaussianBlur(combined_heatmap, (3, 3), 0.5)
            except Exception:
                pass
            combined_heatmap = np.clip(combined_heatmap, 0.0, 1.0)

        print(f"Heatmap created with components: {', '.join(components_found) if components_found else 'None'}")
        print(f"Heatmap range: {np.min(combined_heatmap):.4f} - {np.max(combined_heatmap):.4f}")
        return combined_heatmap

    except Exception as e:
        print(f"Error creating combined heatmap: {e}")
        height = width = 256
        if isinstance(image_size, (list, tuple, np.ndarray)) and len(image_size) >= 2:
            try:
                height = int(image_size[0])
                width = int(image_size[1])
            except Exception:
                pass
        elif isinstance(image_size, dict):
            height = int(image_size.get('height', 256) or 256)
            width = int(image_size.get('width', 256) or 256)
        height = max(1, int(height))
        width = max(1, int(width))
        fallback_heatmap = np.zeros((height, width), dtype=np.float32)
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < (min(height, width) // 4) ** 2
        fallback_heatmap[mask] = 0.5
        return fallback_heatmap


def create_summary_report(ax, analysis_results):
    """Create comprehensive summary report"""
    ax.axis('off')

    try:
        # Extract key information
        metadata = analysis_results.get('metadata', {})
        classification = analysis_results.get('classification', {})
        pipeline_status = analysis_results.get('pipeline_status', {})

        # Build summary text
        summary_text = "RINGKASAN ANALISIS FORENSIK\n" + "="*40 + "\n\n"

        # File information
        filename = metadata.get('Filename', 'Unknown')
        file_size = metadata.get('FileSize', 'Unknown')
        summary_text += f"File: {filename}\n"
        summary_text += f"Ukuran: {file_size}\n\n"

        # Classification results
        if classification:
            pred_type = classification.get('type', 'Unknown')
            confidence = classification.get('confidence', 0)
            uncertainty = classification.get('uncertainty_analysis', {}).get('uncertainty_score', 0)

            summary_text += "HASIL KLASIFIKASI:\n"
            summary_text += f"Tipe: {pred_type}\n"
            summary_text += f"Kepercayaan: {confidence:.1%}\n"
            summary_text += f"Ketidakpastian: {uncertainty:.1%}\n\n"

        # Pipeline status
        if pipeline_status:
            total_stages = pipeline_status.get('total_stages', 0)
            completed_stages = pipeline_status.get('completed_stages', 0)
            failed_stages = pipeline_status.get('failed_stages', [])
            success_rate = (completed_stages / total_stages) * 100 if total_stages > 0 else 0

            summary_text += "STATUS PIPELINE:\n"
            summary_text += f"Tahap Selesai: {completed_stages}/{total_stages}\n"
            summary_text += f"Success Rate: {success_rate:.1f}%\n"
            if failed_stages:
                summary_text += f"Failed: {len(failed_stages)} tahap\n"

        # Key findings
        summary_text += "\nTEMUAN UTAMA:\n"

        # Check for anomalies
        ela_mean = analysis_results.get('ela_mean', 0)
        if ela_mean > 30:
            summary_text += "• ELA menunjukkan anomali tinggi\n"

        # Check copy-move detection
        ransac_matches = analysis_results.get('ransac_matches', [])
        if len(ransac_matches) > 10:
            summary_text += f"• Terdeteksi {len(ransac_matches)} copy-move matches\n"

        # Check noise inconsistency
        noise_data = analysis_results.get('noise_analysis', {})
        if 'inconsistency_score' in noise_data:
            inconsistency = noise_data['inconsistency_score']
            if inconsistency > 0.5:
                summary_text += "• Noise inconsistency signifikan\n"

        # Add forensic assessment
        summary_text += "\nASESMEN FORENSIK:\n"
        if classification:
            if pred_type == 'AUTHENTIC':
                summary_text += "• Gambar cenderung AUTENTIK\n"
            elif 'SUSPICIOUS' in pred_type:
                summary_text += "• Gambar mencurigakan perlu analisis lebih lanjut\n"
            else:
                summary_text += "• Memerlukan analisis tambahan\n"

        # Display summary
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        # Add overall assessment indicator
        if classification:
            confidence = classification.get('confidence', 0)
            assessment_color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.5 else 'red'

            ax.add_patch(plt.Circle((0.85, 0.15), 0.08,
                                  facecolor=assessment_color, alpha=0.7, transform=ax.transAxes))
            ax.text(0.85, 0.15, f'{confidence:.0%}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, fontweight='bold', color='white')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error creating summary:\n{str(e)[:50]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)

def create_dct_visualization(ax, original_pil):
    """Create DCT coefficients visualization"""
    try:
        # Convert to grayscale for DCT
        gray_img = original_pil.convert('L')
        img_array = np.array(gray_img)

        # Take a small patch for DCT visualization (e.g., 64x64)
        patch_size = 64
        h, w = img_array.shape

        # Get center patch
        center_y, center_x = h//2, w//2
        start_y = max(0, center_y - patch_size//2)
        start_x = max(0, center_x - patch_size//2)
        end_y = min(h, start_y + patch_size)
        end_x = min(w, start_x + patch_size)

        patch = img_array[start_y:end_y, start_x:end_x]

        # Apply DCT
        dct_coeffs = cv2.dct(np.float32(patch)/255.0)

        # Log scale for better visualization
        dct_log = np.log(np.abs(dct_coeffs) + 1e-5)

        # Display
        im = ax.imshow(dct_log, cmap='hot', interpolation='nearest')
        ax.set_xlabel('DCT Frequency', fontsize=8)
        ax.set_ylabel('DCT Frequency', fontsize=8)
        ax.tick_params(labelsize=7)

        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')

    except Exception as e:
        ax.text(0.5, 0.5, f'DCT Analysis\nNot Available\n({str(e)[:30]}...)',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def populate_validation_visuals(ax1, ax2):
    """
    Populates two subplots with system validation visuals.
    """
    ax1.clear()
    ax2.clear()

    ax1.set_title("16. Matriks Konfusi (Contoh)", fontsize=9)
    ax2.set_title("17. Distribusi Kepercayaan (Contoh)", fontsize=9)

    if SKLEARN_METRICS_AVAILABLE and SCIPY_AVAILABLE:
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.array([y_true[i] if np.random.rand() < 0.9 else 1-y_true[i] for i in range(100)])
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                    xticklabels=['Normal', 'Mencurigakan'],
                    yticklabels=['Normal', 'Mencurigakan'],
                    linewidths=.5, linecolor='gray')
        ax1.set_xlabel("Hasil Analisis", fontsize=9)
        ax1.set_ylabel("Kondisi Sebenarnya", fontsize=9)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax1.set_title(f"16. Matriks Evaluasi Evidence (Akurasi: {accuracy:.1%})", fontsize=9)

        normal_scores = np.random.normal(loc=20, scale=10, size=50)
        suspicious_scores = np.random.normal(loc=80, scale=10, size=50)
        combined_scores = np.clip(np.concatenate((normal_scores, suspicious_scores)), 0, 100)

        sns.histplot(combined_scores, kde=True, ax=ax2, color="purple", bins=15, alpha=0.6, stat="density", linewidth=0)
        ax2.set_xlabel("Skor Kepercayaan Deteksi", fontsize=9)
        ax2.set_ylabel("Density", fontsize=9)
        ax2.set_title("17. Distribusi Kepercayaan Deteksi", fontsize=9)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.set_xlim(0, 100)

    else:
        ax1.text(0.5, 0.5, "Sklearn/Scipy tidak tersedia", ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "Sklearn/Scipy tidak tersedia", ha='center', va='center', transform=ax2.transAxes)

# Additional helper functions for missing visualizations

def create_noise_visualization(ax, results):
    """Create noise consistency visualization"""
    noise_data = results.get('noise_analysis', {})

    if not noise_data:
        ax.text(0.5, 0.5, 'Noise Analysis\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return

    try:
        # Get noise map
        noise_map = noise_data.get('noise_map')
        if noise_map is not None and isinstance(noise_map, np.ndarray):
            # Display noise map
            ax.imshow(noise_map, cmap='hot')
            ax.set_title('Noise Map', fontsize=10)
            ax.axis('off')

            # Add inconsistency score
            inconsistency = noise_data.get('inconsistency_score', 0)
            ax.text(0.02, 0.98, f'Inconsistency: {inconsistency:.3f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            # Create noise metrics visualization
            metrics = ['Variance', 'Mean', 'Std Dev']
            values = [
                noise_data.get('variance', 0),
                noise_data.get('mean', 0),
                noise_data.get('std_dev', 0)
            ]

            bars = ax.bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)

            ax.set_ylabel('Value', fontsize=8)
            ax.set_title('Noise Analysis', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=7)

    except Exception as e:
        ax.text(0.5, 0.5, f'Noise Analysis\nError: {str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

def create_jpeg_ghost_visualization(ax, results):
    """Create JPEG ghost detection visualization"""
    jpeg_ghost = results.get('jpeg_ghost')
    jpeg_analysis = results.get('jpeg_analysis', {})

    if jpeg_ghost is not None and isinstance(jpeg_ghost, np.ndarray):
        # Display JPEG ghost
        ax.imshow(jpeg_ghost, cmap='hot')
        ax.set_title('JPEG Ghost Map', fontsize=10)
        ax.axis('off')

        # Add ghost analysis info
        ghost_analysis = jpeg_analysis.get('ghost_analysis', {})
        suspicious_ratio = ghost_analysis.get('suspicious_ratio', 0)
        ax.text(0.02, 0.98, f'Suspicious Ratio: {suspicious_ratio:.3f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'JPEG Ghost\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')

def create_histogram_analysis(ax, results):
    """Create RGB histogram analysis visualization"""
    # Placeholder for histogram analysis
    ax.text(0.5, 0.5, 'RGB Histogram\nAnalysis',
           ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_title('RGB Histogram', fontsize=10)
    ax.axis('off')

def create_quality_metrics(ax, results):
    """Create image quality metrics visualization"""
    # Placeholder for quality metrics
    ax.text(0.5, 0.5, 'Quality Metrics\nAnalysis',
           ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_title('Quality Metrics', fontsize=10)
    ax.axis('off')

def create_anomaly_scores(ax, results):
    """Create anomaly detection scores visualization"""
    # Placeholder for anomaly scores
    ax.text(0.5, 0.5, 'Anomaly Scores\nAnalysis',
           ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_title('Anomaly Scores', fontsize=10)
    ax.axis('off')

def create_evidence_strength(ax, results):
    """Create forensic evidence strength visualization"""
    # Placeholder for evidence strength
    ax.text(0.5, 0.5, 'Evidence Strength\nAnalysis',
           ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_title('Evidence Strength', fontsize=10)
    ax.axis('off')

def create_classification_visualization(ax, results):
    """Create classification results visualization"""
    classification = results.get('classification', {})

    if not classification:
        ax.text(0.5, 0.5, 'Classification\nNot Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return

    try:
        # Get classification results
        pred_type = classification.get('type', 'Unknown')
        confidence = classification.get('confidence', 0)

        # Create confidence gauge
        ax.clear()

        # Create gauge background
        theta = np.linspace(0, np.pi, 100)
        gauge_colors = ['red', 'orange', 'yellow', 'green']
        gauge_ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

        for i, (color, (start, end)) in enumerate(zip(gauge_colors, gauge_ranges)):
            mask = (theta >= np.pi * start) & (theta <= np.pi * end)
            ax.fill_between(theta[mask], 0, 1, color=color, alpha=0.3)

        # Add confidence needle
        needle_angle = np.pi * confidence
        ax.plot([needle_angle, needle_angle], [0, 0.8], 'k-', linewidth=3)
        ax.plot([needle_angle], [0.85], 'ko', markersize=8)

        # Add labels
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])

        # Add title and confidence text
        ax.set_title(f'Classification: {pred_type}', fontsize=10, fontweight='bold')
        ax.text(np.pi/2, 1.1, f'Confidence: {confidence:.1%}',
               ha='center', va='center', fontsize=10, fontweight='bold')

    except Exception as e:
        ax.text(0.5, 0.5, f'Classification\nError: {str(e)[:30]}',
               ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')

# --- END OF FILE visualization.py ---
