"""Explicit pixel and object metrics for diagnostic checkpoint evaluation."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops


class DetectionMetrics:
    def __init__(self, match_distance=3.0):
        if match_distance <= 0:
            raise ValueError("match_distance must be positive")
        self.match_distance = match_distance
        self.intersection = self.union = self.images = self.total_pixels = 0
        self.sample_iou_sum = 0.0
        self.gt_objects = self.matched_objects = self.unmatched_prediction_pixels = 0
        self.pixel_false_positives = 0

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("Metrics require equal HxW binary arrays")
        intersection = int(np.logical_and(prediction, target).sum())
        union = int(np.logical_or(prediction, target).sum())
        self.intersection += intersection
        self.union += union
        self.sample_iou_sum += intersection / union if union else 0.0
        self.images += 1
        self.total_pixels += prediction.size
        self.pixel_false_positives += int(np.logical_and(prediction, ~target).sum())
        predicted = regionprops(label(prediction, connectivity=2))
        truth = regionprops(label(target, connectivity=2))
        self.gt_objects += len(truth)
        matched_predictions = set()
        if predicted and truth:
            pc = np.asarray([obj.centroid for obj in predicted])
            tc = np.asarray([obj.centroid for obj in truth])
            distances = np.linalg.norm(tc[:, None, :] - pc[None, :, :], axis=-1)
            penalty = (max(distances.shape) + 1) * (self.match_distance + 1)
            cost = np.where(distances < self.match_distance, distances, penalty)
            rows, cols = linear_sum_assignment(cost)
            matched_predictions = {int(j) for i, j in zip(rows, cols) if distances[i, j] < self.match_distance}
        self.matched_objects += len(matched_predictions)
        self.unmatched_prediction_pixels += sum(int(obj.area) for index, obj in enumerate(predicted) if index not in matched_predictions)

    def result(self):
        return {
            "images": self.images, "IoU": self.intersection / self.union if self.union else 0.0,
            "nIoU": self.sample_iou_sum / self.images if self.images else 0.0,
            "Pd": self.matched_objects / self.gt_objects if self.gt_objects else None,
            "Fa_fraction": self.unmatched_prediction_pixels / self.total_pixels if self.total_pixels else 0.0,
            "Fa_per_million_pixels": self.unmatched_prediction_pixels / self.total_pixels * 1e6 if self.total_pixels else 0.0,
            "pixel_false_positive_fraction": self.pixel_false_positives / self.total_pixels if self.total_pixels else 0.0,
            "ground_truth_objects": self.gt_objects, "matched_objects": self.matched_objects,
            "unmatched_prediction_pixels": self.unmatched_prediction_pixels,
            "protocol": {"connectivity": 8, "centroid_match_distance_strictly_less_than": self.match_distance,
                         "matching": "maximum-cardinality one-to-one matching, then minimum centroid distance",
                         "Fa": "Area of unmatched predicted connected components divided by all evaluated pixels.",
                         "empty_union_sample_iou": 0.0,
                         "paper_protocol_verified": False},
        }
