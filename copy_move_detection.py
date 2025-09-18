"""
Copy-Move Detection Module
Implements feature-based and block-based copy-move forgery detection
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from config import RATIO_THRESH, MIN_DISTANCE, RANSAC_THRESH, MIN_INLIERS, BLOCK_SIZE


def detect_copy_move_advanced(feature_sets, image_shape):
    """
    Advanced copy-move detection using feature matching with RANSAC verification.
    
    Args:
        feature_sets: Dictionary containing feature keypoints and descriptors
        image_shape: Tuple of (width, height) of the image
    
    Returns:
        Tuple of (ransac_matches, ransac_inliers, geometric_transform, total_matches)
    """
    try:
        # Get SIFT features if available
        sift_features = feature_sets.get('sift', ([], None))
        keypoints = sift_features[0]
        descriptors = sift_features[1]
        
        if descriptors is None or len(keypoints) < 10:
            return [], 0, None, 0
        
        # Create matcher - use FLANN for better performance with SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors with k=3 to ensure we have at least 2 neighbors
        matches = flann.knnMatch(descriptors, descriptors, k=min(3, len(descriptors)))
        
        # Apply ratio test and filter self-matches
        good_matches = []
        unique_pairs = set()  # To avoid duplicate pairs
        
        for match_list in matches:
            if len(match_list) < 2:
                continue
                
            # Find first non-self match
            valid_matches = [m for m in match_list if m.queryIdx != m.trainIdx]
            
            if len(valid_matches) >= 2:
                # Apply Lowe's ratio test on the two best non-self matches
                m, n = valid_matches[0], valid_matches[1]
                
                if m.distance < RATIO_THRESH * n.distance:
                    # Check minimum spatial distance
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    distance = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)
                    
                    if distance > MIN_DISTANCE:
                        # Avoid duplicate pairs (i->j and j->i)
                        pair = tuple(sorted([m.queryIdx, m.trainIdx]))
                        if pair not in unique_pairs:
                            unique_pairs.add(pair)
                            good_matches.append(m)
            elif len(valid_matches) == 1:
                # If we only have one valid match, use a fixed threshold
                m = valid_matches[0]
                if m.distance < 100:  # Reasonable threshold for SIFT
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    distance = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)
                    
                    if distance > MIN_DISTANCE:
                        pair = tuple(sorted([m.queryIdx, m.trainIdx]))
                        if pair not in unique_pairs:
                            unique_pairs.add(pair)
                            good_matches.append(m)
        
        total_matches = len(good_matches)
        
        if len(good_matches) < MIN_INLIERS:
            # Fallback: try to find at least some matches even if below threshold
            if len(good_matches) > 0:
                # Return the matches we have, even if below MIN_INLIERS
                return good_matches, len(good_matches), ('fallback', None), total_matches
            return good_matches, 0, None, total_matches
        
        # Prepare points for RANSAC
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Try different transformation models
        best_inliers = 0
        best_matches = good_matches
        best_transform = None
        
        # Try homography if we have enough points
        if len(src_pts) >= 4:
            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH, maxIters=2000, confidence=0.95)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = int(inliers)
                        matches_mask = mask.ravel().tolist()
                        best_matches = [m for i, m in enumerate(good_matches) if i < len(matches_mask) and matches_mask[i]]
                        best_transform = ('homography', M)
            except Exception:
                pass
        
        # Try affine transformation as fallback
        if best_inliers < MIN_INLIERS and len(src_pts) >= 3:
            try:
                M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH, confidence=0.95, maxIters=2000)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = int(inliers)
                        matches_mask = mask.ravel().tolist()
                        best_matches = [m for i, m in enumerate(good_matches) if i < len(matches_mask) and matches_mask[i]]
                        best_transform = ('affine', M)
            except Exception:
                pass
        
        # Additional fallback: try fundamental matrix estimation
        if best_inliers < MIN_INLIERS and len(src_pts) >= 8:
            try:
                M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, RANSAC_THRESH, 0.95)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = int(inliers)
                        matches_mask = mask.ravel().tolist()
                        best_matches = [m for i, m in enumerate(good_matches) if i < len(matches_mask) and matches_mask[i]]
                        best_transform = ('fundamental', M)
            except Exception:
                pass
        
        return best_matches, best_inliers, best_transform, total_matches
        
    except Exception as e:
        print(f"Error in detect_copy_move_advanced: {e}")
        import traceback
        traceback.print_exc()
        return [], 0, None, 0


def detect_copy_move_blocks(image_pil, block_size=BLOCK_SIZE):
    """
    Detect copy-move forgery using block-based method.
    
    Args:
        image_pil: PIL Image object
        block_size: Size of blocks to compare
    
    Returns:
        List of matched block pairs
    """
    try:
        # Convert to grayscale numpy array
        if image_pil.mode != 'L':
            gray_img = np.array(image_pil.convert('L'))
        else:
            gray_img = np.array(image_pil)
        
        h, w = gray_img.shape
        block_matches = []
        
        # Skip if image is too small
        if h < block_size * 2 or w < block_size * 2:
            return []
        
        # Extract blocks
        blocks = []
        positions = []
        
        # Sliding window with stride
        stride = block_size // 2
        for y in range(0, h - block_size + 1, stride):
            for x in range(0, w - block_size + 1, stride):
                block = gray_img[y:y+block_size, x:x+block_size]
                blocks.append(block.flatten())
                positions.append((x, y))
        
        if len(blocks) < 2:
            return []
        
        blocks_array = np.array(blocks)
        
        # Use DBSCAN clustering to find similar blocks
        if len(blocks_array) > 1000:
            # Sample for performance
            indices = np.random.choice(len(blocks_array), 1000, replace=False)
            sampled_blocks = blocks_array[indices]
            sampled_positions = [positions[i] for i in indices]
        else:
            sampled_blocks = blocks_array
            sampled_positions = positions
        
        # Compute pairwise distances and find matches
        for i in range(len(sampled_blocks)):
            for j in range(i + 1, len(sampled_blocks)):
                # Calculate normalized cross-correlation
                block1 = sampled_blocks[i].reshape(block_size, block_size)
                block2 = sampled_blocks[j].reshape(block_size, block_size)
                
                # Normalize blocks
                block1_norm = (block1 - np.mean(block1)) / (np.std(block1) + 1e-10)
                block2_norm = (block2 - np.mean(block2)) / (np.std(block2) + 1e-10)
                
                # Compute correlation
                correlation = np.sum(block1_norm * block2_norm) / (block_size * block_size)
                
                # Check if blocks are similar enough and far apart
                if correlation > 0.95:  # High correlation threshold
                    pos1 = sampled_positions[i]
                    pos2 = sampled_positions[j]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if distance > MIN_DISTANCE:
                        block_matches.append({
                            'src': pos1,  # Source position
                            'dst': pos2,  # Destination position
                            'correlation': float(correlation),
                            'distance': float(distance),
                            'block_size': block_size
                        })
        
        # Sort by correlation and filter by quality
        block_matches = sorted(block_matches, key=lambda x: x['correlation'], reverse=True)

        # Dynamic limit based on image size and match quality
        max_matches = min(len(block_matches), max(20, int((h * w) / (block_size * block_size * 100))))

        # Only return high-quality matches (correlation > 0.9)
        high_quality_matches = [match for match in block_matches if match['correlation'] > 0.9]

        # Return up to max_matches, prioritizing quality
        final_matches = high_quality_matches[:max_matches]

        print(f"  Block matching found {len(final_matches)} high-quality matches (correlation > 0.9)")

        return final_matches
        
    except Exception as e:
        print(f"Error in detect_copy_move_blocks: {e}")
        return []


def advanced_tampering_localization(image_pil, ela_array, matches=None, keypoints=None, n_clusters=8):
    """
    Advanced tampering localization using multiple clustering approaches and feature-based validation.
    
    Args:
        image_pil: PIL Image object (original image)
        ela_array: Numpy array of ELA values
        matches: Optional list of feature matches for validation
        keypoints: Optional list of keypoints for spatial validation
        n_clusters: Number of clusters for K-means
    
    Returns:
        Dictionary containing comprehensive localization results
    """
    try:
        # Ensure ela_array is 2D grayscale
        if len(ela_array.shape) == 3:
            ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
        else:
            ela_gray = ela_array
        
        height, width = ela_gray.shape
        
        # 1. Multi-feature clustering approach
        # Create feature vectors combining ELA, texture, and spatial information
        feature_vectors = []
        coordinates = []
        
        # Sample points for clustering (to manage computational complexity)
        step_size = max(1, min(width, height) // 100)  # Adaptive sampling
        
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                # ELA value
                ela_val = ela_gray[y, x]
                
                # Local texture features (using local standard deviation)
                window_size = 5
                y_start = max(0, y - window_size//2)
                y_end = min(height, y + window_size//2 + 1)
                x_start = max(0, x - window_size//2)
                x_end = min(width, x + window_size//2 + 1)
                
                local_patch = ela_gray[y_start:y_end, x_start:x_end]
                texture_std = np.std(local_patch)
                texture_mean = np.mean(local_patch)
                
                # Spatial coordinates (normalized)
                norm_x = x / width
                norm_y = y / height
                
                # Gradient information
                if y > 0 and y < height-1 and x > 0 and x < width-1:
                    grad_x = float(ela_gray[y, x+1]) - float(ela_gray[y, x-1])
                    grad_y = float(ela_gray[y+1, x]) - float(ela_gray[y-1, x])
                    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
                else:
                    gradient_mag = 0
                
                feature_vectors.append([
                    ela_val, texture_std, texture_mean, 
                    norm_x, norm_y, gradient_mag
                ])
                coordinates.append((x, y))
        
        feature_vectors = np.array(feature_vectors)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_vectors_normalized = scaler.fit_transform(feature_vectors)
        
        # 2. Apply multiple clustering algorithms
        clustering_results = {}
        
        # K-means clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(feature_vectors_normalized)
            clustering_results['kmeans'] = kmeans_labels
        except:
            clustering_results['kmeans'] = np.zeros(len(feature_vectors))
        
        # DBSCAN clustering for density-based detection
        try:
            # Adaptive eps based on image size
            eps = 0.3 * (min(width, height) / 1000)
            dbscan = DBSCAN(eps=eps, min_samples=5)
            dbscan_labels = dbscan.fit_predict(feature_vectors_normalized)
            clustering_results['dbscan'] = dbscan_labels
        except:
            clustering_results['dbscan'] = np.zeros(len(feature_vectors))
        
        # 3. Create comprehensive localization map
        localization_map = np.zeros((height, width), dtype=np.float32)
        tampering_confidence = np.zeros((height, width), dtype=np.float32)
        
        # Process K-means results
        if 'kmeans' in clustering_results:
            kmeans_labels = clustering_results['kmeans']
            
            # Find suspicious clusters based on ELA values
            cluster_ela_means = []
            for i in range(n_clusters):
                cluster_mask = kmeans_labels == i
                if np.any(cluster_mask):
                    cluster_ela_values = feature_vectors[cluster_mask, 0]  # ELA values
                    cluster_ela_means.append((i, np.mean(cluster_ela_values), np.std(cluster_ela_values)))
            
            # Sort by ELA mean and select suspicious clusters
            cluster_ela_means.sort(key=lambda x: x[1], reverse=True)
            
            # Dynamic threshold based on ELA distribution - diperbaiki untuk lebih selektif
            all_ela_values = feature_vectors[:, 0]
            # Adaptive threshold based on distribution statistics
            ela_mean = np.mean(all_ela_values)
            ela_std = np.std(all_ela_values)
            
            # Threshold lebih selektif: gunakan percentile yang lebih tinggi
            # atau mean + 2*std untuk menghindari over-detection
            percentile_thresh = np.percentile(all_ela_values, 85)  # Top 15% saja
            statistical_thresh = ela_mean + 2.0 * ela_std  # 2 std, bukan 1.5
            
            # Pilih yang lebih tinggi untuk lebih selektif
            ela_threshold = max(percentile_thresh, statistical_thresh)
            
            suspicious_clusters = []
            
            # Hanya ambil cluster yang BENAR-BENAR suspicious
            # Tidak memaksa minimal cluster
            for cluster_id, mean_ela, std_ela in cluster_ela_means:
                # Kriteria lebih ketat: harus melebihi threshold DAN memiliki variasi tinggi
                if mean_ela > ela_threshold and mean_ela > ela_mean + 2.5 * ela_std:
                    suspicious_clusters.append(cluster_id)
                    # Batasi maksimal 2 cluster untuk menghindari over-detection
                    if len(suspicious_clusters) >= 2:
                        break
            
            # Jika tidak ada yang memenuhi kriteria ketat, cek dengan kriteria lebih longgar
            if len(suspicious_clusters) == 0 and len(cluster_ela_means) > 0:
                # Hanya ambil top 1 jika ELA-nya signifikan lebih tinggi
                top_cluster = cluster_ela_means[0]
                if top_cluster[1] > ela_mean + 1.5 * ela_std:
                    suspicious_clusters = [top_cluster[0]]
            
            # Map back to image coordinates
            for i, (x, y) in enumerate(coordinates):
                if kmeans_labels[i] in suspicious_clusters:
                    # Use Gaussian kernel for smooth localization
                    kernel_size = max(3, step_size)
                    y_start = max(0, y - kernel_size)
                    y_end = min(height, y + kernel_size + 1)
                    x_start = max(0, x - kernel_size)
                    x_end = min(width, x + kernel_size + 1)
                    
                    # Hitung confidence berdasarkan seberapa jauh dari mean
                    ela_value = feature_vectors[i, 0]
                    # Confidence tinggi hanya untuk nilai yang signifikan di atas mean
                    if ela_value > ela_mean + 2 * ela_std:
                        confidence = min(1.0, (ela_value - ela_mean) / (3 * ela_std))
                    else:
                        confidence = 0.1  # Confidence rendah untuk nilai normal
                    
                    localization_map[y_start:y_end, x_start:x_end] += confidence * 0.3
                    tampering_confidence[y_start:y_end, x_start:x_end] += confidence * 0.5
        
        # Process DBSCAN results (focus on outliers and dense regions)
        if 'dbscan' in clustering_results:
            dbscan_labels = clustering_results['dbscan']
            
            # Outliers (label = -1) and dense clusters with high ELA
            for i, (x, y) in enumerate(coordinates):
                label = dbscan_labels[i]
                ela_val = feature_vectors[i, 0]
                
                # Kriteria DBSCAN lebih selektif: hanya outlier dengan ELA sangat tinggi
                if label == -1 and ela_val > ela_threshold:  # Outlier DAN high ELA
                    kernel_size = max(3, step_size)
                    y_start = max(0, y - kernel_size)
                    y_end = min(height, y + kernel_size + 1)
                    x_start = max(0, x - kernel_size)
                    x_end = min(width, x + kernel_size + 1)
                    
                    # Confidence berdasarkan deviasi dari mean
                    if ela_val > ela_mean + 2.5 * ela_std:
                        confidence = min(1.0, (ela_val - ela_mean) / (3 * ela_std))
                        localization_map[y_start:y_end, x_start:x_end] += confidence * 0.2
                        tampering_confidence[y_start:y_end, x_start:x_end] += confidence * 0.3
                    else:
                        # Abaikan jika tidak cukup signifikan
                        pass
        
        # 4. Enhanced feature-based validation (if matches are provided)
        if matches and keypoints:
            match_confidence_map = np.zeros((height, width), dtype=np.float32)

            # Create spatial density map of matched keypoints
            keypoint_density = np.zeros((height, width), dtype=np.float32)

            for match in matches:
                kp1 = keypoints[match.queryIdx]
                kp2 = keypoints[match.trainIdx]

                # Add confidence around matched keypoints with adaptive weighting
                for kp in [kp1, kp2]:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    if 0 <= x < width and 0 <= y < height:
                        radius = max(3, int(kp.size / 2))
                        y_start = max(0, y - radius)
                        y_end = min(height, y + radius + 1)
                        x_start = max(0, x - radius)
                        x_end = min(width, x + radius + 1)

                        # Higher weight for stronger matches (lower distance)
                        match_confidence = max(0.1, 1.0 - (match.distance / 150.0))

                        # Weight by keypoint response (stronger features are more reliable)
                        response_weight = min(1.0, kp.response / 100.0) if hasattr(kp, 'response') else 0.5

                        # Apply confidence with spatial spreading
                        for dy in range(y_start, y_end):
                            for dx in range(x_start, x_end):
                                distance = np.sqrt((dx - x)**2 + (dy - y)**2)
                                if distance <= radius:
                                    spatial_weight = 1.0 - (distance / radius)
                                    total_weight = match_confidence * response_weight * spatial_weight
                                    match_confidence_map[dy, dx] += total_weight * 0.5
                                    keypoint_density[dy, dx] += 1.0

            # Normalize and apply advanced fusion with ELA-based clustering
            if np.max(match_confidence_map) > 0:
                match_confidence_map = match_confidence_map / np.max(match_confidence_map)

            # Enhanced fusion: Boost areas where both ELA clusters and feature matches agree
            for i, (x, y) in enumerate(coordinates):
                if kmeans_labels[i] in suspicious_clusters:
                    # Check if this location has supporting feature evidence
                    local_match_confidence = match_confidence_map[y, x] if y < height and x < width else 0
                    local_keypoint_density = keypoint_density[y, x] if y < height and x < width else 0

                    # Boost confidence where both methods agree
                    if local_match_confidence > 0.3 or local_keypoint_density > 1.0:
                        boost_factor = 1.0 + min(0.5, local_match_confidence + local_keypoint_density * 0.1)

                        # Apply enhanced localization with feature support
                        kernel_size = max(3, step_size)
                        y_start = max(0, y - kernel_size)
                        y_end = min(height, y + kernel_size + 1)
                        x_start = max(0, x - kernel_size)
                        x_end = min(width, x + kernel_size + 1)

                        ela_value = feature_vectors[i, 0]
                        if ela_value > ela_mean + 1.5 * ela_std:
                            confidence = min(1.0, (ela_value - ela_mean) / (2.5 * ela_std))
                            enhanced_confidence = confidence * boost_factor
                            localization_map[y_start:y_end, x_start:x_end] += enhanced_confidence * 0.4
                            tampering_confidence[y_start:y_end, x_start:x_end] += enhanced_confidence * 0.6

            # Combine maps with weighted fusion
            localization_map += match_confidence_map * 0.3
            tampering_confidence += match_confidence_map * 0.4
        
        # 5. Post-processing and refinement
        # Normalize confidence maps
        if np.max(localization_map) > 0:
            localization_map = localization_map / np.max(localization_map)
        if np.max(tampering_confidence) > 0:
            tampering_confidence = tampering_confidence / np.max(tampering_confidence)
        
        # Create binary tampering mask dengan threshold lebih selektif
        if np.any(tampering_confidence > 0):
            confidence_values = tampering_confidence[tampering_confidence > 0]
            # Threshold lebih tinggi untuk mengurangi false positive
            # Gunakan 75th percentile atau mean + 0.5*std
            percentile_thresh = np.percentile(confidence_values, 75)
            mean_thresh = np.mean(confidence_values) + 0.5 * np.std(confidence_values)
            
            # Pilih yang lebih tinggi untuk lebih selektif
            adaptive_threshold = max(0.3, max(percentile_thresh, mean_thresh))
        else:
            adaptive_threshold = 0.5  # Default lebih tinggi
        
        tampering_mask = tampering_confidence > adaptive_threshold
        
        # Morphological operations for cleanup
        kernel_size = max(3, min(width, height) // 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        tampering_mask_uint8 = tampering_mask.astype(np.uint8) * 255
        tampering_mask_uint8 = cv2.morphologyEx(tampering_mask_uint8, cv2.MORPH_CLOSE, kernel)
        tampering_mask_uint8 = cv2.morphologyEx(tampering_mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Remove small components (dengan threshold yang lebih kecil)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tampering_mask_uint8)
        min_area = max(100, (width * height) // 5000)  # Minimum 0.02% of image area atau 100 pixels
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                tampering_mask_uint8[labels == i] = 0
        
        final_tampering_mask = tampering_mask_uint8 > 128
        
        return {
            'localization_map': localization_map,
            'tampering_mask': final_tampering_mask,
            'confidence_map': tampering_confidence,
            'adaptive_threshold': adaptive_threshold,
            'suspicious_regions': np.sum(final_tampering_mask),
            'clustering_results': clustering_results,
            'feature_validation': matches is not None and keypoints is not None,
            'cluster_means': cluster_ela_means if 'cluster_ela_means' in locals() else [],
            'suspicious_clusters': suspicious_clusters if 'suspicious_clusters' in locals() else [],
            'ela_threshold': ela_threshold if 'ela_threshold' in locals() else 0.5,
            'n_clusters': n_clusters
        }
        
    except Exception as e:
        print(f"Error in advanced_tampering_localization: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results
        return {
            'localization_map': np.zeros_like(ela_array, dtype=np.float32),
            'tampering_mask': np.zeros_like(ela_array, dtype=bool),
            'confidence_map': np.zeros_like(ela_array, dtype=np.float32),
            'adaptive_threshold': 0.5,
            'suspicious_regions': 0,
            'clustering_results': {},
            'feature_validation': False
        }


def kmeans_tampering_localization(image_pil, ela_array, n_clusters=8, matches=None, keypoints=None):
    """
    K-means clustering function for tampering localization.
    Now properly passes matches and keypoints to the advanced function.
    """
    # Call the advanced function with all parameters
    result = advanced_tampering_localization(image_pil, ela_array, n_clusters=n_clusters,
                                             matches=matches, keypoints=keypoints)

    # CRITICAL FIX: Return ENTIRE result, not just partial data
    # Ensure 'kmeans_result' contains all data needed by validator
    return result


def calculate_detection_confidence_score(matches, keypoints, transform_info, localization_result, image_shape):
    """
    Calculate a comprehensive confidence score for copy-move detection
    
    Args:
        matches: List of feature matches
        keypoints: List of keypoints
        transform_info: Transformation information
        localization_result: Result from tampering localization
        image_shape: Tuple of (width, height)
    
    Returns:
        Dictionary containing detailed confidence metrics
    """
    try:
        width, height = image_shape
        total_pixels = width * height
        
        # Initialize score components
        scores = {
            'feature_quality': 0.0,
            'geometric_consistency': 0.0,
            'spatial_distribution': 0.0,
            'transformation_validity': 0.0,
            'localization_confidence': 0.0,
            'scale_consistency': 0.0,
            'overall_confidence': 0.0
        }
        
        if not matches or len(matches) == 0:
            return scores
        
        # 1. Feature Quality Score
        def calculate_feature_quality(matches, keypoints):
            if not matches:
                return 0.0
            
            # Descriptor distance quality
            distances = [m.distance for m in matches]
            avg_distance = np.mean(distances)
            distance_score = max(0, 1.0 - avg_distance / 150.0)  # Normalize by typical SIFT threshold
            
            # Keypoint response quality
            responses = []
            for m in matches:
                kp1_response = keypoints[m.queryIdx].response
                kp2_response = keypoints[m.trainIdx].response
                responses.extend([kp1_response, kp2_response])
            
            avg_response = np.mean(responses) if responses else 0
            response_score = min(1.0, avg_response * 10)  # Scale typical response values
            
            # Match density (matches per keypoint)
            total_keypoints = len(keypoints)
            density_score = min(1.0, len(matches) / max(total_keypoints * 0.1, 1))
            
            return (distance_score * 0.4 + response_score * 0.3 + density_score * 0.3)
        
        scores['feature_quality'] = calculate_feature_quality(matches, keypoints)
        
        # 2. Geometric Consistency Score
        def calculate_geometric_consistency(matches, keypoints, transform_info):
            if not transform_info or not matches:
                return 0.0
            
            transform_type, transform_matrix = transform_info
            
            # Extract points
            src_pts = np.array([keypoints[m.queryIdx].pt for m in matches])
            dst_pts = np.array([keypoints[m.trainIdx].pt for m in matches])
            
            # Calculate reprojection error
            try:
                if transform_type == 'homography' and transform_matrix is not None:
                    src_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
                    projected = (transform_matrix @ src_homogeneous.T).T
                    projected = projected[:, :2] / projected[:, 2:3]
                elif transform_type in ['affine', 'similarity'] and transform_matrix is not None:
                    src_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
                    projected = (transform_matrix @ src_homogeneous.T).T
                else:
                    return 0.0
                
                errors = np.linalg.norm(projected - dst_pts, axis=1)
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                
                # Lower error = higher score
                error_score = max(0, 1.0 - mean_error / 10.0)
                consistency_score = max(0, 1.0 - std_error / 5.0)
                
                return (error_score * 0.7 + consistency_score * 0.3)
                
            except Exception:
                return 0.0
        
        scores['geometric_consistency'] = calculate_geometric_consistency(matches, keypoints, transform_info)
        
        # 3. Spatial Distribution Score
        def calculate_spatial_distribution(matches, keypoints, image_shape):
            if not matches:
                return 0.0
            
            points = np.array([keypoints[m.queryIdx].pt for m in matches])
            
            # Grid-based coverage analysis
            grid_size = 4
            cell_width = width / grid_size
            cell_height = height / grid_size
            
            occupied_cells = set()
            for pt in points:
                cell_x = min(int(pt[0] / cell_width), grid_size - 1)
                cell_y = min(int(pt[1] / cell_height), grid_size - 1)
                occupied_cells.add((cell_x, cell_y))
            
            coverage_score = len(occupied_cells) / (grid_size * grid_size)
            
            # Distance distribution analysis
            if len(points) > 1:
                center = np.mean(points, axis=0)
                distances = np.linalg.norm(points - center, axis=1)
                distance_std = np.std(distances)
                max_possible_std = np.sqrt((width/2)**2 + (height/2)**2) / 2
                distribution_score = min(1.0, distance_std / max_possible_std)
            else:
                distribution_score = 0.0
            
            return (coverage_score * 0.6 + distribution_score * 0.4)
        
        scores['spatial_distribution'] = calculate_spatial_distribution(matches, keypoints, image_shape)
        
        # 4. Transformation Validity Score
        def calculate_transformation_validity(transform_info):
            if not transform_info:
                return 0.0
            
            transform_type, transform_matrix = transform_info
            
            try:
                if transform_type == 'homography':
                    if transform_matrix is None or transform_matrix.shape != (3, 3):
                        return 0.0
                    
                    # Check determinant
                    det = np.linalg.det(transform_matrix[:2, :2])
                    det_score = 1.0 if 0.1 <= abs(det) <= 10 else 0.0
                    
                    # Check condition number (stability)
                    cond = np.linalg.cond(transform_matrix[:2, :2])
                    cond_score = max(0, 1.0 - np.log10(cond) / 3.0) if cond > 1 else 1.0
                    
                    return (det_score * 0.5 + cond_score * 0.5)
                    
                elif transform_type in ['affine', 'similarity']:
                    if transform_matrix is None or transform_matrix.shape[0] < 2:
                        return 0.0
                    
                    # Check scaling factors
                    scale_x = np.linalg.norm(transform_matrix[0, :2])
                    scale_y = np.linalg.norm(transform_matrix[1, :2])
                    
                    scale_score = 1.0 if 0.3 <= scale_x <= 3.0 and 0.3 <= scale_y <= 3.0 else 0.0
                    
                    # For similarity, check scale consistency
                    if transform_type == 'similarity':
                        scale_ratio = scale_x / scale_y
                        ratio_score = 1.0 if 0.8 <= scale_ratio <= 1.25 else 0.0
                        return (scale_score * 0.7 + ratio_score * 0.3)
                    
                    return scale_score
                
                return 0.0
                
            except Exception:
                return 0.0
        
        scores['transformation_validity'] = calculate_transformation_validity(transform_info)
        
        # 5. Localization Confidence Score
        def calculate_localization_confidence(localization_result):
            if not localization_result:
                return 0.0
            
            confidence_map = localization_result.get('confidence_map')
            tampering_mask = localization_result.get('tampering_mask')
            
            if confidence_map is None or tampering_mask is None:
                return 0.0
            
            # Average confidence in detected regions
            if np.any(tampering_mask):
                avg_confidence = np.mean(confidence_map[tampering_mask])
                
                # Region size relative to image
                region_ratio = np.sum(tampering_mask) / total_pixels
                size_score = min(1.0, region_ratio * 10)  # Prefer reasonable-sized regions
                
                # Confidence distribution
                conf_std = np.std(confidence_map[tampering_mask])
                consistency_score = max(0, 1.0 - conf_std)
                
                return (avg_confidence * 0.5 + size_score * 0.3 + consistency_score * 0.2)
            
            return 0.0
        
        scores['localization_confidence'] = calculate_localization_confidence(localization_result)
        
        # 6. Scale Consistency Score (for multi-scale detection)
        def calculate_scale_consistency(matches, keypoints):
            if not matches:
                return 0.0
            
            # Analyze scale ratios between matched keypoints
            scale_ratios = []
            for m in matches:
                kp1 = keypoints[m.queryIdx]
                kp2 = keypoints[m.trainIdx]
                ratio = kp1.size / (kp2.size + 1e-6)
                scale_ratios.append(ratio)
            
            if not scale_ratios:
                return 0.0
            
            # Consistency of scale ratios
            scale_std = np.std(scale_ratios)
            consistency_score = max(0, 1.0 - scale_std)
            
            # Reasonable scale range
            mean_scale = np.mean(scale_ratios)
            range_score = 1.0 if 0.5 <= mean_scale <= 2.0 else 0.0
            
            return (consistency_score * 0.7 + range_score * 0.3)
        
        scores['scale_consistency'] = calculate_scale_consistency(matches, keypoints)
        
        # 7. Calculate Overall Confidence Score
        # Weighted combination of all scores
        weights = {
            'feature_quality': 0.20,
            'geometric_consistency': 0.25,
            'spatial_distribution': 0.15,
            'transformation_validity': 0.20,
            'localization_confidence': 0.15,
            'scale_consistency': 0.05
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Apply bonus for multiple consistent indicators
        high_scores = sum(1 for score in scores.values() if score > 0.7)
        if high_scores >= 4:
            overall_score *= 1.1  # 10% bonus for multiple strong indicators
        
        # Apply penalty for any very low scores
        low_scores = sum(1 for score in scores.values() if score < 0.3)
        if low_scores >= 2:
            overall_score *= 0.9  # 10% penalty for multiple weak indicators
        
        scores['overall_confidence'] = min(1.0, overall_score)
        
        # Add additional metadata
        scores['num_matches'] = len(matches)
        scores['num_keypoints'] = len(keypoints)
        scores['match_ratio'] = len(matches) / max(len(keypoints), 1)
        
        return scores
        
    except Exception as e:
        print(f"Error calculating confidence score: {e}")
        return scores


def classify_detection_result(confidence_scores, threshold_high=0.75, threshold_medium=0.5):
    """
    Classify detection result based on confidence scores
    
    Args:
        confidence_scores: Dictionary from calculate_detection_confidence_score
        threshold_high: Threshold for high confidence detection
        threshold_medium: Threshold for medium confidence detection
    
    Returns:
        Dictionary containing classification results
    """
    overall_confidence = confidence_scores.get('overall_confidence', 0.0)
    
    if overall_confidence >= threshold_high:
        classification = 'HIGH_CONFIDENCE_FORGERY'
        reliability = 'High'
        recommendation = 'Strong evidence of copy-move forgery detected'
    elif overall_confidence >= threshold_medium:
        classification = 'MEDIUM_CONFIDENCE_FORGERY'
        reliability = 'Medium'
        recommendation = 'Possible copy-move forgery detected, manual review recommended'
    else:
        classification = 'LOW_CONFIDENCE_OR_AUTHENTIC'
        reliability = 'Low'
        recommendation = 'No strong evidence of copy-move forgery'
    
    # Detailed analysis
    strengths = []
    weaknesses = []
    
    for metric, score in confidence_scores.items():
        if metric == 'overall_confidence':
            continue
        if score > 0.7:
            strengths.append(f"{metric}: {score:.2f}")
        elif score < 0.3:
            weaknesses.append(f"{metric}: {score:.2f}")
    
    return {
        'classification': classification,
        'reliability': reliability,
        'confidence_score': overall_confidence,
        'recommendation': recommendation,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'detailed_scores': confidence_scores
    }