import json
import numpy as np


class PurePythonLGBM:

    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.model_data = json.load(f)
        self.tree_info = self.model_data["tree_info"]

    def _predict_tree(self, node: dict, feature_values: np.ndarray) -> float:
        if "leaf_value" in node:
            return node["leaf_value"]

        feat_idx = node["split_feature"]

        # Safe index check: if feature index is missing, treat as NaN
        if feat_idx < len(feature_values):
            val = feature_values[feat_idx]
        else:
            val = np.nan

        threshold = node["threshold"]
        default_left = node.get("default_left", True)

        if np.isnan(val):
            next_node = (
                node["left_child"] if default_left else node["right_child"]
            )
        else:
            decision_type = node.get("decision_type", "<=")
            if decision_type == "<=":
                is_left = val <= threshold
            else:
                is_left = val == threshold

            next_node = (
                node["left_child"] if is_left else node["right_child"]
            )

        return self._predict_tree(next_node, feature_values)

    def predict_one(self, feature_vector: np.ndarray) -> float:
        if hasattr(feature_vector, "toarray"):
            feature_vector = feature_vector.toarray().ravel()

        total_score = 0.0
        for tree in self.tree_info:
            total_score += self._predict_tree(
                tree["tree_structure"], feature_vector
            )

        return float(np.clip(total_score, 0.0, 1.0))
