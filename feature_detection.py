"""
Feature detection and matching functions
"""

import numpy as np
import cv2
try:
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    def sk_normalize(arr, norm='l2', axis=1):
        denom = np.linalg.norm(arr, ord=2 if norm=='l2' else 1, axis=axis, keepdims=True)
        denom[denom == 0] = 1
        return arr / denom
from config import *

def extract_multi_detector_features(image_pil, ela_image_pil, ela_mean, ela_stddev, sift_nfeatures=SIFT_FEATURES):
    """Extract features using multiple detectors (SIFT, ORB, SURF)"""
    ela_np = np.array(ela_image_pil)
    
    # Adaptive thresholding with multiple methods
    thresholds = [
        ela_mean + 1.2 * ela_stddev,  # Standard threshold
        ela_mean + 1.0 * ela_stddev,  # Lower threshold for more sensitive detection
        np.percentile(ela_np, 75),    # 75th percentile
    ]
    
    # Try multiple thresholds and select the one that gives reasonable ROI size
    best_roi_mask = None
    best_roi_pixels = 0
    
    for thresh_val in thresholds:
        thresh_val = max(min(thresh_val, 200), 20)  # Clamp between 20-200
        temp_mask = (ela_np > thresh_val).astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel)
        
        roi_pixels = np.sum(temp_mask > 0)
        total_pixels = temp_mask.size
        roi_percentage = roi_pixels / total_pixels
        
        # Prefer masks that cover 10-40% of image
        if 0.1 <= roi_percentage <= 0.4 and roi_pixels > best_roi_pixels:
            best_roi_mask = temp_mask
            best_roi_pixels = roi_pixels
    
    # Fallback to original method if no good mask found
    if best_roi_mask is None:
        threshold = ela_mean + 1.5 * ela_stddev
        threshold = max(min(threshold, 180), 30)
        roi_mask = (ela_np > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
    else:
        roi_mask = best_roi_mask
    
    # Convert to grayscale with enhancement
    original_image_np = np.array(image_pil.convert('RGB'))
    gray_original = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2GRAY)
    
    # Multiple enhancement techniques
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray_original)
    
    # Extract features using multiple detectors
    feature_sets = {}
    
    # 1. SIFT with multiple parameter attempts
    sift_kps = []
    sift_descs = None
    
    # Try multiple parameter combinations for better feature detection
    param_combinations = [
        (sift_nfeatures, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD),  # Default
        (sift_nfeatures * 2, SIFT_CONTRAST_THRESHOLD * 0.5, SIFT_EDGE_THRESHOLD),  # More features, lower contrast threshold
        (sift_nfeatures, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD * 2),  # Higher edge threshold
    ]
    
    for nfeat, contrast_thresh, edge_thresh in param_combinations:
        try:
            sift = cv2.SIFT_create(nfeatures=nfeat, 
                                  contrastThreshold=contrast_thresh, 
                                  edgeThreshold=edge_thresh)
            kp, desc = sift.detectAndCompute(gray_enhanced, mask=roi_mask)
            if desc is not None and len(kp) > len(sift_kps):
                sift_kps = kp
                sift_descs = desc
        except Exception:
            continue
    
    # If no features found with mask, try without mask
    if len(sift_kps) < 10:
        try:
            sift = cv2.SIFT_create(nfeatures=sift_nfeatures, 
                                  contrastThreshold=SIFT_CONTRAST_THRESHOLD * 0.3, 
                                  edgeThreshold=SIFT_EDGE_THRESHOLD)
            kp, desc = sift.detectAndCompute(gray_enhanced, None)
            if desc is not None and len(kp) > len(sift_kps):
                sift_kps = kp
                sift_descs = desc
        except Exception:
            pass
    
    feature_sets['sift'] = (sift_kps, sift_descs)
    
    # 2. ORB
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, 
                        scaleFactor=ORB_SCALE_FACTOR, 
                        nlevels=ORB_LEVELS)
    kp_orb, desc_orb = orb.detectAndCompute(gray_enhanced, mask=roi_mask)
    feature_sets['orb'] = (kp_orb, desc_orb)
    
    # 3. AKAZE
    try:
        akaze = cv2.AKAZE_create()
        kp_akaze, desc_akaze = akaze.detectAndCompute(gray_enhanced, mask=roi_mask)
        feature_sets['akaze'] = (kp_akaze, desc_akaze)
    except:
        feature_sets['akaze'] = ([], None)
    
    return feature_sets, roi_mask, gray_enhanced


def extract_multi_scale_features(image_pil, ela_image_pil, ela_mean, ela_stddev, scales=[1.0, 0.8, 1.2]):
    """
    Extract features at multiple scales for robust copy-move detection
    
    Args:
        image_pil: PIL Image object
        ela_image_pil: ELA processed image
        ela_mean: Mean ELA value
        ela_stddev: Standard deviation of ELA values
        scales: List of scale factors to apply
    
    Returns:
        Dictionary containing multi-scale feature sets
    """
    multi_scale_features = {}
    original_size = image_pil.size
    
    for scale in scales:
        try:
            # Calculate new dimensions
            new_width = int(original_size[0] * scale)
            new_height = int(original_size[1] * scale)
            
            # Skip if resulting image is too small
            if new_width < 200 or new_height < 200:
                continue
                
            # Resize images
            scaled_image = image_pil.resize((new_width, new_height), Image.LANCZOS)
            scaled_ela = ela_image_pil.resize((new_width, new_height), Image.LANCZOS)
            
            # Adjust ELA statistics for scaled image
            scaled_ela_array = np.array(scaled_ela)
            scaled_ela_mean = np.mean(scaled_ela_array)
            scaled_ela_std = np.std(scaled_ela_array)
            
            # Extract features at this scale
            feature_sets, roi_mask, gray_enhanced = extract_multi_detector_features(
                scaled_image, scaled_ela, scaled_ela_mean, scaled_ela_std
            )
            
            # Scale keypoints back to original image coordinates
            for detector_name, (keypoints, descriptors) in feature_sets.items():
                if keypoints and len(keypoints) > 0:
                    scaled_keypoints = []
                    for kp in keypoints:
                        # Create new keypoint with scaled coordinates
                        scaled_kp = cv2.KeyPoint(
                            x=kp.pt[0] / scale,
                            y=kp.pt[1] / scale,
                            size=kp.size / scale,
                            angle=kp.angle,
                            response=kp.response,
                            octave=kp.octave,
                            class_id=kp.class_id
                        )
                        scaled_keypoints.append(scaled_kp)
                    
                    feature_sets[detector_name] = (scaled_keypoints, descriptors)
            
            multi_scale_features[f'scale_{scale}'] = {
                'features': feature_sets,
                'roi_mask': roi_mask,
                'scale_factor': scale,
                'image_size': (new_width, new_height)
            }
            
        except Exception as e:
            print(f"Error processing scale {scale}: {e}")
            continue
    
    return multi_scale_features


def optimize_detection_parameters(image_pil, ela_image_pil):
    """
    Optimize detection parameters based on image characteristics
    
    Args:
        image_pil: PIL Image object
        ela_image_pil: ELA processed image
    
    Returns:
        Dictionary containing optimized parameters
    """
    try:
        # Convert images to numpy arrays
        image_np = np.array(image_pil.convert('RGB'))
        ela_np = np.array(ela_image_pil)
        
        if len(ela_np.shape) == 3:
            ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
        else:
            ela_gray = ela_np
        
        height, width = ela_gray.shape
        image_size = width * height
        
        # 1. Analyze image characteristics
        characteristics = analyze_image_characteristics(image_np, ela_gray)
        
        # 2. Optimize SIFT parameters
        sift_params = optimize_sift_parameters(characteristics, image_size)
        
        # 3. Optimize matching parameters
        matching_params = optimize_matching_parameters(characteristics)
        
        # 4. Optimize clustering parameters
        clustering_params = optimize_clustering_parameters(characteristics, image_size)
        
        # 5. Optimize detection thresholds
        threshold_params = optimize_threshold_parameters(characteristics)
        
        return {
            'image_characteristics': characteristics,
            'sift_parameters': sift_params,
            'matching_parameters': matching_params,
            'clustering_parameters': clustering_params,
            'threshold_parameters': threshold_params
        }
        
    except Exception as e:
        print(f"Error optimizing parameters: {e}")
        # Return default parameters
        return get_default_parameters()


def analyze_image_characteristics(image_np, ela_gray):
    """
    Analyze image characteristics to guide parameter optimization
    """
    height, width = ela_gray.shape
    
    # Image size category
    total_pixels = width * height
    if total_pixels < 500000:  # < 0.5MP
        size_category = 'small'
    elif total_pixels < 2000000:  # < 2MP
        size_category = 'medium'
    else:
        size_category = 'large'
    
    # Texture analysis
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Calculate local standard deviation (texture measure)
    kernel = np.ones((9, 9), np.float32) / 81
    mean_img = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
    sqr_img = cv2.filter2D((gray_image.astype(np.float32))**2, -1, kernel)
    texture_map = np.sqrt(sqr_img - mean_img**2)
    avg_texture = np.mean(texture_map)
    
    if avg_texture < 10:
        texture_category = 'smooth'
    elif avg_texture < 25:
        texture_category = 'moderate'
    else:
        texture_category = 'textured'
    
    # ELA characteristics
    ela_mean = np.mean(ela_gray)
    ela_std = np.std(ela_gray)
    ela_max = np.max(ela_gray)
    
    # ELA distribution analysis
    ela_hist, _ = np.histogram(ela_gray, bins=50, range=(0, 255))
    ela_entropy = -np.sum((ela_hist + 1e-10) * np.log2(ela_hist + 1e-10))
    
    if ela_std < 15:
        ela_category = 'uniform'
    elif ela_std < 30:
        ela_category = 'moderate'
    else:
        ela_category = 'varied'
    
    # Noise level estimation
    noise_estimate = estimate_noise_level(gray_image)
    
    if noise_estimate < 5:
        noise_category = 'low'
    elif noise_estimate < 15:
        noise_category = 'moderate'
    else:
        noise_category = 'high'
    
    # Edge density
    edges = cv2.Canny(gray_image, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels
    
    if edge_density < 0.05:
        edge_category = 'sparse'
    elif edge_density < 0.15:
        edge_category = 'moderate'
    else:
        edge_category = 'dense'
    
    return {
        'size_category': size_category,
        'texture_category': texture_category,
        'ela_category': ela_category,
        'noise_category': noise_category,
        'edge_category': edge_category,
        'dimensions': (width, height),
        'total_pixels': total_pixels,
        'avg_texture': avg_texture,
        'ela_stats': {'mean': ela_mean, 'std': ela_std, 'max': ela_max},
        'ela_entropy': ela_entropy,
        'noise_estimate': noise_estimate,
        'edge_density': edge_density
    }


def estimate_noise_level(gray_image):
    """
    Estimate noise level using Laplacian variance method
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.var()


def optimize_sift_parameters(characteristics, image_size):
    """
    Optimize SIFT parameters based on image characteristics
    """
    from config import SIFT_FEATURES, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD
    
    # Base parameters
    nfeatures = SIFT_FEATURES
    contrast_threshold = SIFT_CONTRAST_THRESHOLD
    edge_threshold = SIFT_EDGE_THRESHOLD
    
    # Adjust based on image size
    if characteristics['size_category'] == 'large':
        nfeatures = int(nfeatures * 1.5)
    elif characteristics['size_category'] == 'small':
        nfeatures = int(nfeatures * 0.7)
    
    # Adjust based on texture
    if characteristics['texture_category'] == 'smooth':
        contrast_threshold *= 0.5  # Lower threshold for smooth images
        nfeatures = int(nfeatures * 1.2)  # More features needed
    elif characteristics['texture_category'] == 'textured':
        contrast_threshold *= 1.5  # Higher threshold for textured images
    
    # Adjust based on noise
    if characteristics['noise_category'] == 'high':
        contrast_threshold *= 1.3
        edge_threshold *= 1.2
    elif characteristics['noise_category'] == 'low':
        contrast_threshold *= 0.8
    
    # Adjust based on edge density
    if characteristics['edge_category'] == 'dense':
        edge_threshold *= 1.3
    elif characteristics['edge_category'] == 'sparse':
        edge_threshold *= 0.8
    
    return {
        'nfeatures': min(nfeatures, 5000),  # Cap at reasonable limit
        'contrast_threshold': max(0.01, min(0.1, contrast_threshold)),
        'edge_threshold': max(5, min(20, edge_threshold))
    }


def optimize_matching_parameters(characteristics):
    """
    Optimize matching parameters based on image characteristics
    """
    from config import RATIO_THRESH, MIN_DISTANCE, RANSAC_THRESH, MIN_INLIERS
    
    ratio_thresh = RATIO_THRESH
    min_distance = MIN_DISTANCE
    ransac_thresh = RANSAC_THRESH
    min_inliers = MIN_INLIERS
    
    # Adjust based on texture
    if characteristics['texture_category'] == 'smooth':
        ratio_thresh *= 0.9  # Stricter ratio test
        min_distance *= 0.8  # Allow closer matches
    elif characteristics['texture_category'] == 'textured':
        ratio_thresh *= 1.1  # More lenient ratio test
        min_distance *= 1.2  # Require farther matches
    
    # Adjust based on noise
    if characteristics['noise_category'] == 'high':
        ransac_thresh *= 1.5  # More tolerant RANSAC
        min_inliers = max(min_inliers, 8)  # Require more inliers
    elif characteristics['noise_category'] == 'low':
        ransac_thresh *= 0.8  # Stricter RANSAC
        min_inliers = max(4, min_inliers - 2)  # Can work with fewer inliers
    
    # Adjust based on image size
    if characteristics['size_category'] == 'large':
        min_distance *= 1.3
        min_inliers = max(min_inliers, 8)
    elif characteristics['size_category'] == 'small':
        min_distance *= 0.7
        min_inliers = max(4, min_inliers - 2)
    
    return {
        'ratio_thresh': max(0.6, min(0.9, ratio_thresh)),
        'min_distance': max(15, min(60, min_distance)),
        'ransac_thresh': max(3.0, min(15.0, ransac_thresh)),
        'min_inliers': max(4, min(15, min_inliers))
    }


def optimize_clustering_parameters(characteristics, image_size):
    """
    Optimize clustering parameters based on image characteristics
    """
    # Base number of clusters
    n_clusters = 8
    
    # Adjust based on ELA characteristics
    if characteristics['ela_category'] == 'uniform':
        n_clusters = 6  # Fewer clusters for uniform ELA
    elif characteristics['ela_category'] == 'varied':
        n_clusters = 10  # More clusters for varied ELA
    
    # Adjust based on image size
    if characteristics['size_category'] == 'large':
        n_clusters = min(12, n_clusters + 2)
    elif characteristics['size_category'] == 'small':
        n_clusters = max(4, n_clusters - 2)
    
    # DBSCAN parameters
    base_eps = 0.3
    min_samples = 5
    
    if characteristics['texture_category'] == 'smooth':
        base_eps *= 0.8
        min_samples = 3
    elif characteristics['texture_category'] == 'textured':
        base_eps *= 1.2
        min_samples = 7
    
    return {
        'n_clusters': n_clusters,
        'dbscan_eps': base_eps,
        'dbscan_min_samples': min_samples
    }


def optimize_threshold_parameters(characteristics):
    """
    Optimize detection thresholds based on image characteristics
    """
    # Base confidence thresholds
    high_confidence_thresh = 0.75
    medium_confidence_thresh = 0.5
    
    # Adjust based on noise level
    if characteristics['noise_category'] == 'high':
        high_confidence_thresh *= 1.1  # Require higher confidence
        medium_confidence_thresh *= 1.1
    elif characteristics['noise_category'] == 'low':
        high_confidence_thresh *= 0.95  # Can be slightly more lenient
        medium_confidence_thresh *= 0.95
    
    # Adjust based on texture
    if characteristics['texture_category'] == 'smooth':
        # Smooth images may have fewer but more reliable features
        high_confidence_thresh *= 0.9
        medium_confidence_thresh *= 0.9
    elif characteristics['texture_category'] == 'textured':
        # Textured images may have more false positives
        high_confidence_thresh *= 1.05
        medium_confidence_thresh *= 1.05
    
    return {
        'high_confidence_threshold': min(0.9, max(0.6, high_confidence_thresh)),
        'medium_confidence_threshold': min(0.7, max(0.3, medium_confidence_thresh))
    }


def get_default_parameters():
    """
    Return default parameters when optimization fails
    """
    from config import (SIFT_FEATURES, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD,
                       RATIO_THRESH, MIN_DISTANCE, RANSAC_THRESH, MIN_INLIERS)
    
    return {
        'image_characteristics': {},
        'sift_parameters': {
            'nfeatures': SIFT_FEATURES,
            'contrast_threshold': SIFT_CONTRAST_THRESHOLD,
            'edge_threshold': SIFT_EDGE_THRESHOLD
        },
        'matching_parameters': {
            'ratio_thresh': RATIO_THRESH,
            'min_distance': MIN_DISTANCE,
            'ransac_thresh': RANSAC_THRESH,
            'min_inliers': MIN_INLIERS
        },
        'clustering_parameters': {
            'n_clusters': 8,
            'dbscan_eps': 0.3,
            'dbscan_min_samples': 5
        },
        'threshold_parameters': {
            'high_confidence_threshold': 0.75,
            'medium_confidence_threshold': 0.5
        }
    }


def match_multi_scale_features(multi_scale_features, ratio_thresh, min_distance, ransac_thresh, min_inliers):
    """
    Perform feature matching across multiple scales
    
    Args:
        multi_scale_features: Dictionary of features at different scales
        ratio_thresh: Ratio threshold for Lowe's test
        min_distance: Minimum spatial distance between matches
        ransac_thresh: RANSAC threshold
        min_inliers: Minimum number of inliers
    
    Returns:
        Dictionary containing best matches across scales
    """
    best_results = {
        'matches': [],
        'inliers': 0,
        'transform': None,
        'scale': 1.0,
        'confidence': 0.0
    }
    
    scale_results = []
    
    # Test each scale
    for scale_key, scale_data in multi_scale_features.items():
        try:
            feature_sets = scale_data['features']
            scale_factor = scale_data['scale_factor']
            
            # Get SIFT features for this scale
            sift_features = feature_sets.get('sift', ([], None))
            keypoints, descriptors = sift_features
            
            if descriptors is None or len(keypoints) < min_inliers:
                continue
            
            # Perform matching at this scale
            matches, inliers, transform = match_sift_features(
                keypoints, descriptors, ratio_thresh, 
                min_distance * scale_factor,  # Scale the minimum distance
                ransac_thresh, min_inliers
            )
            
            if inliers > 0:
                # Calculate confidence score based on multiple factors
                match_density = inliers / len(keypoints) if len(keypoints) > 0 else 0
                scale_penalty = abs(1.0 - scale_factor) * 0.1  # Prefer scales closer to 1.0
                confidence = (inliers / max(min_inliers, 1)) * match_density * (1.0 - scale_penalty)
                
                scale_results.append({
                    'matches': matches,
                    'inliers': inliers,
                    'transform': transform,
                    'scale': scale_factor,
                    'confidence': confidence,
                    'keypoints': keypoints
                })
        
        except Exception as e:
            print(f"Error matching features at {scale_key}: {e}")
            continue
    
    # Select best result based on confidence score
    if scale_results:
        best_result = max(scale_results, key=lambda x: x['confidence'])
        
        # Additional validation: check if multiple scales agree
        consistent_scales = [r for r in scale_results if r['inliers'] >= min_inliers]
        
        if len(consistent_scales) >= 2:
            # Multiple scales found matches - increase confidence
            best_result['confidence'] *= 1.2
            best_result['multi_scale_consensus'] = True
        else:
            best_result['multi_scale_consensus'] = False
        
        best_results.update(best_result)
    
    return best_results


def validate_geometric_consistency(matches, keypoints, transform_info, image_shape):
    """
    Validate geometric consistency of matches to reduce false positives
    
    Args:
        matches: List of cv2.DMatch objects
        keypoints: List of keypoints
        transform_info: Tuple of (transform_type, transformation_matrix)
        image_shape: Tuple of (width, height)
    
    Returns:
        Dictionary containing validation results and filtered matches
    """
    if not matches or not transform_info:
        return {
            'valid_matches': [],
            'consistency_score': 0.0,
            'geometric_valid': False,
            'spatial_distribution_valid': False
        }
    
    transform_type, transform_matrix = transform_info
    width, height = image_shape
    
    # Extract match points
    src_pts = np.array([keypoints[m.queryIdx].pt for m in matches])
    dst_pts = np.array([keypoints[m.trainIdx].pt for m in matches])
    
    # 1. Spatial Distribution Validation
    def validate_spatial_distribution(points, image_shape):
        """Check if points are well distributed across the image"""
        if len(points) < 4:
            return False
            
        # Divide image into grid and check coverage
        grid_size = 3
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        occupied_cells = set()
        for pt in points:
            cell_x = min(int(pt[0] / cell_width), grid_size - 1)
            cell_y = min(int(pt[1] / cell_height), grid_size - 1)
            occupied_cells.add((cell_x, cell_y))
        
        # Require at least 40% of cells to be occupied
        coverage = len(occupied_cells) / (grid_size * grid_size)
        return coverage >= 0.4
    
    spatial_valid = validate_spatial_distribution(src_pts, image_shape)
    
    # 2. Transformation Consistency Validation
    def validate_transformation_consistency(src_pts, dst_pts, transform_type, transform_matrix):
        """Validate transformation consistency"""
        if transform_matrix is None:
            return False, 0.0
        
        try:
            if transform_type == 'homography':
                # Check homography properties
                if transform_matrix.shape != (3, 3):
                    return False, 0.0
                
                # Check determinant (should not be too close to zero)
                det = np.linalg.det(transform_matrix[:2, :2])
                if abs(det) < 0.1 or abs(det) > 10:
                    return False, 0.0
                
                # Check if transformation preserves reasonable aspect ratios
                corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]).T
                transformed_corners = transform_matrix @ corners
                transformed_corners = transformed_corners[:2] / transformed_corners[2]
                
                # Check if transformed corners are reasonable
                if np.any(transformed_corners < -width) or np.any(transformed_corners > 2*width):
                    return False, 0.0
                
            elif transform_type in ['affine', 'similarity']:
                # Check affine transformation properties
                if transform_matrix.shape[0] < 2 or transform_matrix.shape[1] < 3:
                    return False, 0.0
                
                # Check scaling factors
                scale_x = np.linalg.norm(transform_matrix[0, :2])
                scale_y = np.linalg.norm(transform_matrix[1, :2])
                
                if scale_x < 0.3 or scale_x > 3.0 or scale_y < 0.3 or scale_y > 3.0:
                    return False, 0.0
                
                # For similarity transform, check if scales are approximately equal
                if transform_type == 'similarity':
                    scale_ratio = scale_x / scale_y
                    if scale_ratio < 0.8 or scale_ratio > 1.25:
                        return False, 0.0
            
            # Calculate reprojection error
            if transform_type == 'homography':
                src_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
                projected = (transform_matrix @ src_homogeneous.T).T
                projected = projected[:, :2] / projected[:, 2:3]
            else:
                src_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
                projected = (transform_matrix @ src_homogeneous.T).T
            
            errors = np.linalg.norm(projected - dst_pts, axis=1)
            mean_error = np.mean(errors)
            
            # Good transformation should have low reprojection error
            consistency_score = max(0, 1.0 - mean_error / 10.0)  # Normalize by expected error
            
            return mean_error < 5.0, consistency_score
            
        except Exception as e:
            return False, 0.0
    
    geometric_valid, consistency_score = validate_transformation_consistency(
        src_pts, dst_pts, transform_type, transform_matrix
    )
    
    # 3. Match Quality Validation
    def validate_match_quality(matches, keypoints):
        """Validate individual match quality"""
        valid_matches = []
        
        for match in matches:
            kp1 = keypoints[match.queryIdx]
            kp2 = keypoints[match.trainIdx]
            
            # Check keypoint response (strength)
            if kp1.response < 0.01 or kp2.response < 0.01:
                continue
            
            # Check descriptor distance
            if match.distance > 150:  # Threshold for SIFT descriptors
                continue
            
            # Check scale consistency
            scale_ratio = kp1.size / (kp2.size + 1e-6)
            if scale_ratio < 0.5 or scale_ratio > 2.0:
                continue
            
            valid_matches.append(match)
        
        return valid_matches
    
    quality_filtered_matches = validate_match_quality(matches, keypoints)
    
    # 4. Cluster Analysis for Copy-Move Validation
    def validate_copy_move_pattern(src_pts, dst_pts):
        """Check if matches form a coherent copy-move pattern"""
        if len(src_pts) < 6:
            return False
        
        # Calculate displacement vectors
        displacements = dst_pts - src_pts
        
        # Check if displacements are consistent (similar direction and magnitude)
        mean_displacement = np.mean(displacements, axis=0)
        displacement_std = np.std(displacements, axis=0)
        
        # Displacements should be consistent for copy-move
        consistency_threshold = 20  # pixels
        displacement_consistent = np.all(displacement_std < consistency_threshold)
        
        # Check minimum displacement (avoid self-matches)
        min_displacement = np.min(np.linalg.norm(displacements, axis=1))
        sufficient_displacement = min_displacement > 30
        
        return displacement_consistent and sufficient_displacement
    
    copy_move_valid = validate_copy_move_pattern(src_pts, dst_pts)
    
    # Combine all validation results
    overall_valid = (geometric_valid and spatial_valid and 
                    copy_move_valid and len(quality_filtered_matches) >= 6)
    
    # Calculate final consistency score
    final_score = consistency_score
    if spatial_valid:
        final_score *= 1.1
    if copy_move_valid:
        final_score *= 1.2
    if len(quality_filtered_matches) >= len(matches) * 0.8:
        final_score *= 1.1
    
    return {
        'valid_matches': quality_filtered_matches,
        'consistency_score': min(final_score, 1.0),
        'geometric_valid': geometric_valid,
        'spatial_distribution_valid': spatial_valid,
        'copy_move_pattern_valid': copy_move_valid,
        'overall_valid': overall_valid,
        'reprojection_error': consistency_score,
        'quality_retention': len(quality_filtered_matches) / max(len(matches), 1)
    }

def match_sift_features(keypoints, descriptors, ratio_thresh, min_distance, ransac_thresh, min_inliers):
    """Enhanced SIFT matching with robust cross-checking and geometric validation"""
    # Handle empty descriptors
    if descriptors is None or len(descriptors) == 0:
        return [], 0, None
        
    descriptors_norm = sk_normalize(descriptors, norm='l2', axis=1)
    
    # FLANN matcher with optimized parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)  # More trees for better accuracy
    search_params = dict(checks=100)  # More checks for better matching
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Forward and backward matching for cross-checking
    matches_forward = flann.knnMatch(descriptors_norm, descriptors_norm, k=min(8, len(descriptors)))
    matches_backward = flann.knnMatch(descriptors_norm, descriptors_norm, k=min(8, len(descriptors)))
    
    # Apply Lowe's ratio test and cross-checking
    good_matches = []
    match_pairs = []
    unique_pairs = set()
    
    for i, match_list in enumerate(matches_forward):
        if len(match_list) < 2:
            continue
            
        # Find valid non-self matches
        valid_matches = [m for m in match_list if m.queryIdx != m.trainIdx]
        
        if len(valid_matches) >= 2:
            m, n = valid_matches[0], valid_matches[1]
            
            # Lowe's ratio test with adaptive threshold - diperbaiki untuk matching yang lebih baik
            adaptive_ratio = ratio_thresh * (1.0 + 0.05 * np.log(len(descriptors) / 800.0)) if len(descriptors) > 800 else ratio_thresh * 0.9  # Threshold lebih ketat
            
            if m.distance < adaptive_ratio * n.distance:
                # Cross-checking: verify backward match
                backward_matches = [bm for bm in matches_backward[m.trainIdx] if bm.trainIdx == i]
                
                if backward_matches and backward_matches[0].distance < adaptive_ratio * (backward_matches[1].distance if len(backward_matches) > 1 else backward_matches[0].distance * 2):
                    # Spatial distance validation
                    pt1 = keypoints[i].pt
                    pt2 = keypoints[m.trainIdx].pt
                    spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    
                    # Scale consistency check - diperbaiki untuk toleransi yang lebih baik
                    scale_ratio = keypoints[i].size / (keypoints[m.trainIdx].size + 1e-6)
                    scale_consistent = 0.4 <= scale_ratio <= 2.5  # Range diperluas dari 0.5-2.0 ke 0.4-2.5
                    
                    # Orientation consistency check - diperbaiki untuk toleransi yang lebih baik
                    angle_diff = abs(keypoints[i].angle - keypoints[m.trainIdx].angle)
                    angle_diff = min(angle_diff, 360 - angle_diff)  # Handle circular nature
                    orientation_consistent = angle_diff < 60  # Diperluas dari 45 ke 60 derajat
                    
                    if (spatial_dist > min_distance and scale_consistent and orientation_consistent):
                        # Avoid duplicate pairs
                        pair = tuple(sorted([i, m.trainIdx]))
                        if pair not in unique_pairs:
                            unique_pairs.add(pair)
                            good_matches.append(m)
                            match_pairs.append((i, m.trainIdx))
    
    if len(match_pairs) < min_inliers:
        return good_matches, 0, None
    
    # Enhanced RANSAC verification with multiple transformation models
    src_pts = np.float32([keypoints[i].pt for i, _ in match_pairs]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[j].pt for _, j in match_pairs]).reshape(-1, 1, 2)
    
    best_inliers = 0
    best_transform = None
    best_mask = None
    best_matches = good_matches
    
    # Try different transformation models with improved parameters - diperbaiki untuk deteksi yang lebih baik
    transformation_configs = [
        ('similarity', lambda: cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, 
                                                          ransacReprojThreshold=ransac_thresh * 1.2,  # Threshold diperlonggar
                                                          maxIters=2000, confidence=0.95)),           # Iterasi dikurangi, confidence diperlonggar
        ('affine', lambda: cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                               ransacReprojThreshold=ransac_thresh * 1.2,    # Threshold diperlonggar
                                               maxIters=2000, confidence=0.95)),             # Iterasi dikurangi, confidence diperlonggar
        ('homography', lambda: cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                                 ransac_thresh * 1.2, maxIters=2000, confidence=0.95) if len(src_pts) >= 4 else (None, None))  # Parameter diperlonggar
    ]
    
    for transform_type, transform_func in transformation_configs:
        try:
            result = transform_func()
            if result[0] is not None and result[1] is not None:
                M, mask = result
                inliers = np.sum(mask)
                
                # Additional geometric validation
                if inliers >= min_inliers:
                    # Check transformation quality
                    if transform_type == 'homography' and M is not None:
                        # Check if homography is reasonable (not too distorted)
                        det = np.linalg.det(M[:2, :2])
                        if abs(det) < 0.1 or abs(det) > 10:  # Avoid extreme scaling
                            continue
                    
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_transform = (transform_type, M)
                        best_mask = mask
                        
        except Exception as e:
            continue
    
    # Final validation and filtering
    if best_mask is not None and best_inliers >= min_inliers:
        # Filter matches based on RANSAC inliers
        ransac_matches = [good_matches[i] for i in range(len(good_matches))
                         if i < len(best_mask) and best_mask[i][0] == 1]
        
        # Additional consistency check on inlier matches
        if len(ransac_matches) >= min_inliers:
            # Check spatial distribution of matches
            inlier_pts = [keypoints[m.queryIdx].pt for m in ransac_matches]
            if len(set(inlier_pts)) >= min_inliers * 0.8:  # Ensure spatial diversity
                return ransac_matches, best_inliers, best_transform
    
    return good_matches, 0, None

def match_orb_features(keypoints, descriptors, min_distance, ransac_thresh, min_inliers):
    """ORB feature matching"""
    # Hamming distance matcher for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=6)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        for m in match_list[1:]:  # Skip self-match
            pt1 = keypoints[i].pt
            pt2 = keypoints[m.trainIdx].pt
            
            spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            if spatial_dist > min_distance and m.distance < 80:  # Hamming distance threshold
                good_matches.append(m)
                match_pairs.append((i, m.trainIdx))
    
    if len(match_pairs) < min_inliers:
        return good_matches, 0, None
    
    # Simple geometric verification
    return good_matches, len(match_pairs), ('orb_matches', None)

def match_akaze_features(keypoints, descriptors, min_distance, ransac_thresh, min_inliers):
    """AKAZE feature matching"""
    if descriptors is None:
        return [], 0, None
    
    # Hamming distance for AKAZE
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=6)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        if len(match_list) > 1:
            for m in match_list[1:]:  # Skip self-match
                pt1 = keypoints[i].pt
                pt2 = keypoints[m.trainIdx].pt
                
                spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                
                if spatial_dist > min_distance and m.distance < 100:
                    good_matches.append(m)
                    match_pairs.append((i, m.trainIdx))
    
    return good_matches, len(match_pairs), ('akaze_matches', None)