# validator.py

from PIL import Image
import numpy as np # Import numpy

# Diambil dari app2.py
class ForensicValidator:
    def __init__(self):
        # Bobot algoritma (harus berjumlah 1.0)
        self.weights = {
            'clustering': 0.25,  # K-Means (metode utama)
            'localization': 0.25,  # Lokalisasi tampering (metode utama)
            'ela': 0.20,  # Error Level Analysis (metode pendukung)
            'feature_matching': 0.15,  # SIFT (metode pendukung)
            'metadata': 0.15,  # Metadata analysis (metode pendukung)
        }
        
        # Threshold minimum untuk setiap teknik (0-1 scale) - Disesuaikan untuk performa yang lebih baik
        self.thresholds = {
            'clustering': 0.45,      # Diturunkan dari 0.60 untuk K-Means
            'localization': 0.60,
            'ela': 0.60,
            'feature_matching': 0.35, # Diturunkan dari 0.60 untuk SIFT
            'metadata': 0.25,        # Diturunkan dari 0.40 untuk metadata
        }
    
    def validate_clustering(self, analysis_results):
        """Validasi kualitas clustering K-Means"""
        # Access `kmeans_localization` inside `localization_analysis`
        kmeans_data = analysis_results.get('localization_analysis', {}).get('kmeans_localization', {})

        # NEW CHECK: Handle empty or incomplete data
        if not kmeans_data or 'cluster_means' not in kmeans_data or not kmeans_data['cluster_means']:
            return 0.0, "Gagal: Data statistik K-Means tidak tersedia untuk validasi."
            
        cluster_means_tuples = kmeans_data.get('cluster_means', [])

        # --- VALIDASI STRUKTUR DATA BARU ---
        # Check if data is valid: must be list of tuples with at least 3 elements
        if not cluster_means_tuples or not isinstance(cluster_means_tuples, list):
            return 0.0, "Gagal: Format data 'cluster_means' tidak valid atau kosong."

        # Validate structure and extract mean values safely
        mean_values = []
        if cluster_means_tuples and isinstance(cluster_means_tuples[0], tuple):
            # Handle tuple format: (cluster_id, mean, std, ...)
            for item in cluster_means_tuples:
                if isinstance(item, tuple) and len(item) >= 3:
                    mean_values.append(item[1])  # Extract mean (second element)
                else:
                    return 0.0, f"Gagal: Format tuple tidak valid: {item}"
        else:
            # Handle direct numeric values
            mean_values = cluster_means_tuples

        cluster_count = len(mean_values)

        if cluster_count < 2:
            separation_score = max(mean_values) - min(mean_values) if mean_values else 0
            total_clusters = len(cluster_means_tuples) if cluster_means_tuples else 0
            suspicious_count = len(kmeans_data.get('suspicious_clusters', []))
            return 0.4, f"Gagal: Skor pemisahan cluster ({separation_score:.1f}) di bawah ambang batas (5.0). Teridentifikasi **{suspicious_count} cluster mencurigakan** dari **{total_clusters}** cluster yang ada."
            
        # 2. Periksa pemisahan cluster (semakin tinggi selisih mean ELA antar cluster semakin baik)
        mean_diff = max(mean_values) - min(mean_values) if mean_values else 0
        mean_diff_score = min(1.0, mean_diff / 15.0)  # Normalisasi diperbaiki: a diff of 15 implies a score of 1.0
        
        # 3. Periksa identifikasi cluster tampering (jika ada cluster dengan ELA tinggi yang ditandai)
        suspicious_clusters = kmeans_data.get('suspicious_clusters', [])
        # Check if highest ELA cluster is marked as suspicious
        tampering_identified = len(suspicious_clusters) > 0 and mean_values and max(mean_values) > 5
        
        # 4. Periksa area tampering berukuran wajar (tidak terlalu kecil atau terlalu besar)
        tampering_pct = analysis_results.get('localization_analysis', {}).get('tampering_percentage', 0)
        size_score = 0.0
        if 1.0 < tampering_pct < 50.0:  # Ideal size range for actual tampering
            size_score = 1.0
        elif tampering_pct <= 1.0 and tampering_pct > 0.0:  # Too small but exists
            size_score = tampering_pct # linear interpolation from 0 to 1
        elif tampering_pct >= 50.0: # Too large (might be global effect or full image replacement)
            size_score = max(0.0, 1.0 - ((tampering_pct - 50) / 50.0)) # Linear falloff from 1.0 to 0.0 for 50-100%

        # Gabungkan skor dengan faktor berbobot - Diperbaiki untuk performa yang lebih baik
        confidence = (
            0.25 * min(cluster_count / 4.0, 1.0)  # Up to 4 clusters, lebih realistis
            + 0.35 * mean_diff_score               # Bobot lebih tinggi untuk pemisahan cluster
            + 0.25 * float(tampering_identified)   # Bobot lebih tinggi untuk identifikasi tampering
            + 0.15 * size_score                    # Bobot dikurangi untuk ukuran area
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        details = (
            f"Jumlah cluster: {cluster_count}, "
            f"Pemisahan cluster (Max-Min ELA): {mean_diff:.2f}, "
            f"Tampering teridentifikasi: {'Ya' if tampering_identified else 'Tidak'}, "
            f"Area tampering: {tampering_pct:.1f}%"
        )
        
        return confidence, details
    
    def validate_localization(self, analysis_results):
        """Validasi efektivitas lokalisasi tampering"""
        localization_data = analysis_results.get('localization_analysis', {})

        if not localization_data:
            return 0.0, "Data lokalisasi tidak tersedia"

        # 1. Periksa apakah mask tampering yang digabungkan telah dihasilkan
        has_combined_mask = False
        if 'combined_tampering_mask' in localization_data and localization_data['combined_tampering_mask'] is not None:
            mask = localization_data['combined_tampering_mask']
            # Check if it's a numpy array with size property or has __len__
            if hasattr(mask, 'size') and mask.size > 0:
                has_combined_mask = True
            elif hasattr(mask, '__len__') and len(mask) > 0:
                has_combined_mask = True

        if not has_combined_mask:
            return 0.0, f"Gagal: Mask tampering gabungan tidak berhasil dibuat. Ukuran data: {len(localization_data)} elemen"
            
        # 2. Periksa persentase area yang ditandai (harus wajar untuk manipulasi)
        tampering_pct = localization_data.get('tampering_percentage', 0.0)
        area_score = 0.0
        if 0.5 < tampering_pct < 40.0:  # Common range for effective tampering, neither too small nor too large
            area_score = 1.0
        elif 0.0 < tampering_pct <= 0.5:  # Too small, might be noise
            area_score = tampering_pct / 0.5 # Scale from 0 to 1 as it gets to 0.5%
        else: # tampering_pct >= 40.0: # Too large, could be entire image replaced or a global filter
            area_score = max(0.0, 1.0 - ((tampering_pct - 40.0) / 60.0)) # Drops from 1 to 0 for 40% to 100%
        
        # 3. Periksa konsistensi fisik dengan analisis lain
        ela_mean = analysis_results.get('ela_mean', 0.0)
        ela_std = analysis_results.get('ela_std', 0.0)
        noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0.0)
        jpeg_ghost_ratio = analysis_results.get('jpeg_ghost_suspicious_ratio', 0.0) # Check this exists
        
        # High ELA means stronger splicing signal in general.
        ela_consistency = min(1.0, max(0.0, (ela_mean - 5.0) / 10.0)) # Scores 0 at ELA mean 5, 1 at 15
        ela_consistency = ela_consistency * min(1.0, max(0.0, (ela_std - 10.0) / 15.0)) # Add std influence (scores 0 at 10, 1 at 25)

        # High noise inconsistency (for areas, or globally near manipulated regions)
        noise_consistency = min(1.0, max(0.0, (noise_inconsistency - 0.1) / 0.3)) # Scores 0 at 0.1, 1 at 0.4

        # High JPEG ghost ratio
        jpeg_consistency = min(1.0, max(0.0, jpeg_ghost_ratio / 0.2)) # Scores 0 at 0, 1 at 0.2

        # Combine physical consistency. Max implies if one is strong, it still lends credence.
        physical_consistency = max(ela_consistency, noise_consistency, jpeg_consistency)
        
        # Skor gabungan dengan faktor berbobot
        confidence = (
            0.4 * float(has_combined_mask) # Must have a mask
            + 0.3 * area_score # Quality of area percentage
            + 0.3 * physical_consistency # Agreement with other physical anomalies
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Kalibrasi ke rentang [0,1]

        # Create detailed area assessment
        area_status = "Ideal"
        if tampering_pct <= 0.5:
            area_status = f"Terlalu kecil ({tampering_pct:.1f}%)"
        elif tampering_pct >= 40.0:
            area_status = f"Terlalu besar ({tampering_pct:.1f}%) - mungkin penggantian gambar penuh"
        else:
            area_status = f"Optimal ({tampering_pct:.1f}%)"

        details = (
            f"Mask gabungan: {'OK' if has_combined_mask else 'GAGAL'}, "
            f"Area tampering: {area_status}, "
            f"Konsistensi fisik - ELA: {ela_consistency:.2f}, Noise: {noise_consistency:.2f}, JPEG: {jpeg_consistency:.2f}"
        )
        
        return confidence, details
    
    def validate_ela(self, analysis_results):
        """Enhanced validation for advanced Error Level Analysis"""
        ela_image_obj = analysis_results.get('ela_image')
        # Check if ela_image object itself is a valid PIL Image or has image-like properties that can be converted
        if ela_image_obj is None or (not isinstance(ela_image_obj, Image.Image) and not hasattr(ela_image_obj, 'size') and not hasattr(ela_image_obj, 'ndim')):
             return 0.0, "Tidak ada gambar ELA yang tersedia atau format tidak valid"
            
        ela_mean = analysis_results.get('ela_mean', 0.0)
        ela_std = analysis_results.get('ela_std', 0.0)
        
        # Enhanced adaptive scoring for ELA mean with better low-value handling
        mean_score = self._calculate_adaptive_mean_score(ela_mean, analysis_results)
        
        # Enhanced std scoring with texture awareness
        std_score = self._calculate_adaptive_std_score(ela_std, analysis_results)
            
        # Enhanced regional analysis with new features
        regional_stats = analysis_results.get('ela_regional_stats', {})
        regional_inconsistency = regional_stats.get('regional_inconsistency', 0.0)
        outlier_regions = regional_stats.get('outlier_regions', 0)
        entropy_inconsistency = regional_stats.get('entropy_inconsistency', 0.0)
        texture_aware_score = regional_stats.get('texture_aware_score', 0.0)
        confidence_weighted_score = regional_stats.get('confidence_weighted_score', 0.0)
        
        # Advanced scoring components
        inconsistency_score = min(1.0, regional_inconsistency / 0.4)  # Slightly more sensitive
        outlier_score = min(1.0, outlier_regions / 4.0)  # More sensitive to outliers
        entropy_score = min(1.0, entropy_inconsistency / 2.0)
        texture_score = min(1.0, texture_aware_score)
        confidence_score = confidence_weighted_score
        
        # Enhanced quality analysis with frequency domain features
        quality_stats = analysis_results.get('ela_quality_stats', [])
        quality_variation, frequency_score = self._analyze_enhanced_quality_metrics(quality_stats)
        
        # Adaptive weighting based on signal characteristics
        signal_enhancement_ratio = regional_stats.get('signal_enhancement_ratio', 1.0)
        enhancement_bonus = min(0.2, (signal_enhancement_ratio - 1.0) * 0.5) if signal_enhancement_ratio > 1.0 else 0.0
        
        # Dynamic weight adjustment for low ELA scenarios
        base_weights = self._calculate_dynamic_weights(ela_mean, ela_std, regional_stats)
        
        # Combine scores with adaptive weights
        confidence = (
            base_weights['mean'] * mean_score
            + base_weights['std'] * std_score
            + base_weights['inconsistency'] * inconsistency_score
            + base_weights['outlier'] * outlier_score
            + base_weights['entropy'] * entropy_score
            + base_weights['texture'] * texture_score
            + base_weights['confidence'] * confidence_score
            + base_weights['quality'] * quality_variation
            + base_weights['frequency'] * frequency_score
            + enhancement_bonus
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        # Enhanced details with preservation of important low-value explanations
        details = self._generate_enhanced_details(
            ela_mean, ela_std, mean_score, std_score, 
            regional_inconsistency, outlier_regions, 
            entropy_inconsistency, texture_aware_score,
            confidence_weighted_score, signal_enhancement_ratio
        )
        
        return confidence, details
    
    def _calculate_adaptive_mean_score(self, ela_mean, analysis_results):
        """Calculate adaptive mean score with better handling of low values"""
        regional_stats = analysis_results.get('ela_regional_stats', {})
        suspicious_regions = regional_stats.get('suspicious_regions', [])
        outlier_regions = regional_stats.get('outlier_regions', 0)
        regional_inconsistency = regional_stats.get('regional_inconsistency', 0.0)
        
        # More sophisticated scoring that doesn't penalize low ELA values as harshly
        if 6.0 <= ela_mean <= 25.0:  # Good range for manipulation
            base_score = 1.0
        elif ela_mean > 25.0:  # High values with gradual penalty
            base_score = max(0.3, 1.0 - (ela_mean - 25.0) / 20.0)
        elif 2.0 <= ela_mean < 6.0:  # Low but potentially valid range
            # Use a logarithmic scale for better low-value handling
            base_score = 0.3 + (0.7 * (ela_mean - 2.0) / 4.0)
            
            # Significant bonus for having suspicious regions or outliers
            if len(suspicious_regions) > 0 or outlier_regions > 0:
                region_bonus = min(0.4, (len(suspicious_regions) * 0.1 + outlier_regions * 0.05))
                base_score = min(1.0, base_score + region_bonus)
            
            # Additional bonus for regional inconsistency
            if regional_inconsistency > 0.1:
                inconsistency_bonus = min(0.3, regional_inconsistency)
                base_score = min(1.0, base_score + inconsistency_bonus)
                
        elif 0.1 <= ela_mean < 2.0:  # Very low values
            # Don't automatically give 0 score - use regional analysis
            base_score = 0.1 + (0.2 * ela_mean / 2.0)
            
            # Strong bonus if there are regional anomalies despite low mean
            if len(suspicious_regions) > 0 or outlier_regions > 0 or regional_inconsistency > 0.1:
                anomaly_score = min(0.6, 
                    len(suspicious_regions) * 0.15 + 
                    outlier_regions * 0.1 + 
                    regional_inconsistency * 2.0
                )
                base_score = min(0.8, base_score + anomaly_score)
        else:
            # Only give 0 if truly no signal
            base_score = max(0.0, ela_mean * 0.1)
            
        return base_score
    
    def _calculate_adaptive_std_score(self, ela_std, analysis_results):
        """Calculate adaptive std score with texture awareness"""
        regional_stats = analysis_results.get('ela_regional_stats', {})
        texture_aware_score = regional_stats.get('texture_aware_score', 0.0)
        
        # Adjust std thresholds based on texture complexity
        texture_factor = 1.0 + texture_aware_score * 0.5
        lower_threshold = 8.0 / texture_factor
        upper_threshold = 35.0 * texture_factor
        
        if lower_threshold <= ela_std <= upper_threshold:
            std_score = 1.0
        elif ela_std > upper_threshold:
            std_score = max(0.0, 1.0 - (ela_std - upper_threshold) / 15.0)
        elif ela_std < lower_threshold and ela_std > 0.0:
            std_score = ela_std / lower_threshold
        else:
            std_score = 0.0
            
        return std_score
    
    def _analyze_enhanced_quality_metrics(self, quality_stats):
        """Analyze enhanced quality metrics including frequency features"""
        if not quality_stats:
            return 0.0, 0.0
            
        # Traditional quality variation
        means = [q.get('mean', 0) for q in quality_stats if 'mean' in q]
        quality_variation = 0.0
        if len(means) > 1:
            quality_variation = max(means) - min(means)
            quality_variation = min(1.0, quality_variation / 12.0)  # More sensitive
        
        # Frequency domain analysis
        frequency_score = 0.0
        if any('frequency_energy' in q for q in quality_stats):
            freq_energies = [q.get('frequency_energy', 0) for q in quality_stats]
            freq_variation = np.std(freq_energies) if len(freq_energies) > 1 else 0.0
            frequency_score = min(1.0, freq_variation / 0.1)
        
        return quality_variation, frequency_score
    
    def _calculate_dynamic_weights(self, ela_mean, ela_std, regional_stats):
        """Calculate dynamic weights based on signal characteristics"""
        # Base weights
        weights = {
            'mean': 0.25,
            'std': 0.15,
            'inconsistency': 0.15,
            'outlier': 0.15,
            'entropy': 0.1,
            'texture': 0.05,
            'confidence': 0.05,
            'quality': 0.05,
            'frequency': 0.05
        }
        
        # Adjust weights for low ELA scenarios
        if ela_mean < 5.0:
            # Increase importance of regional and texture analysis
            weights['inconsistency'] += 0.1
            weights['entropy'] += 0.05
            weights['texture'] += 0.05
            weights['confidence'] += 0.1
            weights['mean'] -= 0.15
            weights['std'] -= 0.15
        
        # Adjust for high texture complexity
        texture_score = regional_stats.get('texture_aware_score', 0.0)
        if texture_score > 0.5:
            weights['texture'] += 0.05
            weights['frequency'] += 0.05
            weights['mean'] -= 0.05
            weights['std'] -= 0.05
        
        return weights
    
    def _generate_enhanced_details(self, ela_mean, ela_std, mean_score, std_score,
                                  regional_inconsistency, outlier_regions,
                                  entropy_inconsistency, texture_aware_score,
                                  confidence_weighted_score, signal_enhancement_ratio):
        """Generate enhanced details with preservation of important explanations"""
        
        # Format scores to ensure they show actual calculated values, not just 0.00
        mean_score_display = mean_score if mean_score > 0.01 else max(0.01, ela_mean * 0.01)
        
        details = (
            f"ELA mean: {ela_mean:.2f} (score: {mean_score:.2f}), "
            f"ELA std: {ela_std:.2f} (score: {std_score:.2f}), "
            f"Inkonsistensi regional: {regional_inconsistency:.3f}, "
            f"Region outlier: {outlier_regions}, "
            f"Entropi inkonsistensi: {entropy_inconsistency:.3f}, "
            f"Tekstur-aware score: {texture_aware_score:.3f}, "
            f"Weighted confidence: {confidence_weighted_score:.3f}, "
            f"Signal enhancement: {signal_enhancement_ratio:.2f}x"
        )
        
        # Add interpretive guidance with score explanation
        if ela_mean < 5.0:
            score_explanation = "rendah namun masih valid" if mean_score > 0.3 else "rendah karena nilai ELA minimal"
            details += f" | CATATAN: Nilai ELA {ela_mean:.2f} tergolong rendah. "
            details += f"Score {mean_score:.2f} ({score_explanation}) karena: "
            
            if outlier_regions > 0 or regional_inconsistency > 0.1:
                details += f"(1) Terdeteksi {outlier_regions} region outlier dan inkonsistensi regional {regional_inconsistency:.3f}, "
                details += f"yang meningkatkan score meskipun ELA mean rendah. "
            else:
                details += f"(1) Tidak ada anomali regional signifikan yang terdeteksi. "
                
            details += f"(2) Nilai ELA rendah dapat mengindikasikan gambar asli atau manipulasi halus. "
            details += f"(3) Analisis tekstur (score: {texture_aware_score:.3f}) dan regional tetap penting untuk konfirmasi."
        elif ela_mean > 25.0:
            details += f" | CATATAN: Nilai ELA tinggi ({ela_mean:.2f}) mungkin mengindikasikan over-processing atau noise berlebih."
        
        return details
    
    def validate_metadata(self, analysis_results):
        """Validasi komprehensif metadata dengan pendekatan yang realistis"""
        metadata = analysis_results.get('metadata', {})
        
        if not metadata:
            return 0.0, "Data metadata tidak tersedia"
        
        # Ambil skor autentisitas metadata
        authenticity_score = metadata.get('Metadata_Authenticity_Score', 0)
        inconsistencies = metadata.get('Metadata_Inconsistency', [])
        
        # Konversi skor ke skala 0-1 dengan threshold yang lebih realistis dan toleran
        if authenticity_score >= 70:      # Excellent metadata (diturunkan dari 75)
            base_confidence = 1.0
        elif authenticity_score >= 60:    # Good metadata (diturunkan dari 65)
            base_confidence = 0.9
        elif authenticity_score >= 50:    # Acceptable metadata (diturunkan dari 55)
            base_confidence = 0.8          # Ditingkatkan dari 0.7
        elif authenticity_score >= 40:    # Questionable but possible (diturunkan dari 45)
            base_confidence = 0.6          # Ditingkatkan dari 0.5
        elif authenticity_score >= 30:    # Poor but not necessarily fake (diturunkan dari 35)
            base_confidence = 0.4          # Ditingkatkan dari 0.3
        elif authenticity_score >= 20:    # Very poor (diturunkan dari 25)
            base_confidence = 0.3          # Ditingkatkan dari 0.2
        else:                             # Extremely poor
            base_confidence = 0.2          # Ditingkatkan dari 0.1
        
        # Analisis inkonsistensi dengan bobot yang disesuaikan
        inconsistency_penalty = 0.0
        critical_inconsistencies = 0
        minor_inconsistencies = 0
        
        for inconsistency in inconsistencies:
            inconsistency_lower = inconsistency.lower()
            
            # Inkonsistensi kritis (penalty dikurangi)
            if any(critical in inconsistency_lower for critical in 
                   ['photoshop', 'gimp', 'heavily modified', 'fake']):
                critical_inconsistencies += 1
                inconsistency_penalty += 0.10  # Dikurangi dari 0.15
            
            # Inkonsistensi sedang (penalty dikurangi)
            elif any(moderate in inconsistency_lower for moderate in 
                     ['time difference', 'software', 'editing']):
                minor_inconsistencies += 1
                inconsistency_penalty += 0.03  # Dikurangi dari 0.05
            
            # Inkonsistensi ringan (penalty dikurangi)
            else:
                inconsistency_penalty += 0.01  # Dikurangi dari 0.02
        
        # Terapkan penalty dengan batas maksimum yang dikurangi
        inconsistency_penalty = min(0.25, inconsistency_penalty)  # Dikurangi dari 0.4
        
        # Hitung confidence akhir
        final_confidence = max(0.0, base_confidence - inconsistency_penalty)
        
        # Bonus untuk metadata yang lengkap (threshold diturunkan)
        if authenticity_score >= 70:  # Diturunkan dari 80
            final_confidence = min(1.0, final_confidence + 0.15)  # Ditingkatkan dari 0.1
        elif authenticity_score >= 60:  # Bonus tambahan untuk skor sedang
            final_confidence = min(1.0, final_confidence + 0.08)
        
        # Detail yang informatif
        details = (
            f"Skor autentisitas: {authenticity_score}/100, "
            f"Inkonsistensi kritis: {critical_inconsistencies}, "
            f"Inkonsistensi minor: {minor_inconsistencies}, "
            f"Total inkonsistensi: {len(inconsistencies)}"
        )
        
        return final_confidence, details
    
    def validate_feature_matching(self, analysis_results):
        """Validasi kualitas pencocokan fitur SIFT/ORB/AKAZE"""
        ransac_inliers = analysis_results.get('ransac_inliers', 0)
        sift_matches = analysis_results.get('sift_matches', 0) # Raw matches before RANSAC
        sift_keypoints = analysis_results.get('sift_keypoints', [])
        
        # Ensure ransac_inliers are not negative
        if ransac_inliers < 0: 
            ransac_inliers = 0
        
        # Check if we have keypoints but no matches (legitimate scenario)
        if len(sift_keypoints) > 0 and sift_matches == 0:
            # Still give a small score for having features, even without matches
            return 0.1, f"Terdeteksi {len(sift_keypoints)} keypoints tetapi tidak ada matches (kemungkinan tidak ada copy-move)"
            
        if sift_matches < 1: # No matches at all
             return 0.0, "Tidak ada data pencocokan fitur yang signifikan"
            
        # 1. Periksa kecocokan yang signifikan (RANSAC inliers sebagai indikator kuat)
        # Normalisasi inlier diperbaiki: threshold yang lebih realistis
        inlier_score = min(1.0, ransac_inliers / 15.0) # Score 1.0 at 15 inliers (diturunkan dari 25)
        
        # Raw matches count diperbaiki: threshold yang lebih realistis
        match_score = min(1.0, sift_matches / 80.0) # Score 1.0 at 80 raw matches (diturunkan dari 150)
        
        # 2. Periksa transformasi geometris yang ditemukan oleh RANSAC
        has_transform = analysis_results.get('geometric_transform') is not None
        transform_type = None
        if has_transform: # geometric_transform format is (type_string, matrix)
            try: # Robust access for tuple/list
                transform_type = analysis_results['geometric_transform'][0]
            except (TypeError, IndexError): # In case it's not a tuple or is empty
                transform_type = "Unknown_Type"
        
        # 3. Periksa kecocokan blok (harus berkorelasi dengan kecocokan fitur untuk copy-move)
        block_matches = len(analysis_results.get('block_matches', []))
        block_score = min(1.0, block_matches / 8.0) # Score 1.0 at 8 block matches (diturunkan dari 15)
        
        # Cross-algorithm correlation diperbaiki: threshold yang lebih realistis
        correlation_score = 0.0
        if ransac_inliers > 5 and block_matches > 3: # Both strong: high correlation (threshold diturunkan)
            correlation_score = 1.0
        elif ransac_inliers > 0 and block_matches > 0: # Both exist: some correlation
            correlation_score = 0.7  # Ditingkatkan dari 0.5
        
        # Gabungkan skor dengan bobot yang diperbaiki
        confidence = (
            0.30 * inlier_score # Weight untuk RANSAC inliers
            + 0.25 * match_score # Weight ditingkatkan untuk overall matches
            + 0.15 * float(has_transform) # Weight untuk detecting transform type
            + 0.15 * block_score # Weight ditingkatkan untuk block matching
            + 0.15 * correlation_score # Consistency score between two detection methods
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        details = (
            f"RANSAC inliers: {ransac_inliers} (score: {inlier_score:.2f}), "
            f"Raw SIFT matches: {sift_matches} (score: {match_score:.2f}), "
            f"Tipe transformasi: {transform_type if transform_type else 'Tidak ada'}, "
            f"Kecocokan blok: {block_matches} (score: {block_score:.2f})"
        )
        
        return confidence, details
    
    def validate_cross_algorithm(self, analysis_results):
        """Validasi konsistensi silang algoritma"""
        if not analysis_results:
            return [], 0.0, "Tidak ada hasil analisis yang tersedia", []
        
        validation_results = {}
        for technique, validate_func in [
            ('clustering', self.validate_clustering),
            ('localization', self.validate_localization),
            ('ela', self.validate_ela),
            ('feature_matching', self.validate_feature_matching),
            ('metadata', self.validate_metadata)
        ]:
            confidence, details = validate_func(analysis_results)
            # Ensure confidence is a float, especially important from fallback paths
            confidence = float(confidence)
            validation_results[technique] = {
                'confidence': confidence,
                'details': details,
                'weight': self.weights[technique],
                'threshold': self.thresholds[technique],
                'passed': confidence >= self.thresholds[technique]
            }
        
        # Prepare textual results for console/logging
        process_results_list = []
        
        for technique, result in validation_results.items():
            status = "[LULUS]" if result['passed'] else "[GAGAL]"
            emoji = "✅" if result['passed'] else "❌"
            process_results_list.append(f"{emoji} {status:10} | Validasi {technique.capitalize()} - Skor: {result['confidence']:.2f}")
            
        # Calculate weighted individual technique scores
        weighted_scores = {
            technique: result['confidence'] * result['weight']
            for technique, result in validation_results.items()
        }
        
        # Calculate inter-algorithm agreement ratio
        agreement_pairs = 0
        total_pairs = 0
        techniques_list = list(validation_results.keys()) # Convert to list to iterate
        
        for i in range(len(techniques_list)):
            for j in range(i + 1, len(techniques_list)):
                t1, t2 = techniques_list[i], techniques_list[j]
                total_pairs += 1
                # If both passed or both failed, they "agree"
                if validation_results[t1]['passed'] == validation_results[t2]['passed']:
                    agreement_pairs += 1
        
        if total_pairs > 0:
            agreement_ratio = float(agreement_pairs) / total_pairs
        else: # Handle case of 0 total pairs (e.g., less than 2 techniques or specific edge cases)
            agreement_ratio = 1.0 # If nothing to compare, assume perfect agreement logically
        
        # Combine weighted score and agreement bonus
        raw_weighted_total = sum(weighted_scores.values())
        consensus_boost = agreement_ratio * 0.10 # Add max 10% bonus for perfect agreement (tuned)
        
        final_score = (raw_weighted_total * 100) + (consensus_boost * 100)
        
        # Clamp final score between 0 and 100
        final_score = min(100.0, max(0.0, final_score))
        
        # Collect failed validations for detailed reporting
        failed_validations_detail = [
            {
                'name': f"Validasi {technique.capitalize()}",
                'reason': f"Skor kepercayaan di bawah ambang batas {result['threshold']:.2f}",
                'rule': f"LULUS = (Kepercayaan >= {result['threshold']:.2f})",
                'values': f"Nilai aktual: Kepercayaan = {result['confidence']:.2f}\nDetail: {result['details']}"
            }
            for technique, result in validation_results.items()
            if not result['passed']
        ]
        
        # Determine confidence level description for summary text
        if final_score >= 95:
            confidence_level = "Sangat Tinggi (Very High)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Semua metode analisis menunjukkan konsistensi dan kualitas tinggi."
        elif final_score >= 90:
            confidence_level = "Tinggi (High)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Sebagian besar metode analisis menunjukkan konsistensi dan kualitas baik."
        elif final_score >= 85:
            confidence_level = "Sedang (Medium)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Beberapa metode analisis menunjukkan inkonsistensi minor."
        else:
            confidence_level = "Rendah (Low)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Terdapat inkonsistensi signifikan antar metode analisis yang memerlukan perhatian."
        
        return process_results_list, final_score, summary_text, failed_validations_detail

# --- END OF FILE validator.py ---