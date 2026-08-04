import joblib
import numpy as np


class PurePythonLGBM:

    def __init__(self, model_path: str, is_joblib: bool = True):
        if is_joblib:
            # Load scikit-learn model or pipeline components using joblib
            self.model_data = joblib.load(model_path)
        else:
            import json
            with open(model_path, "r") as f:
                self.model_data = json.load(f)
        
        # Adjust extraction based on whether it's a raw tree text format or a joblib/json object
        if isinstance(self.model_data, dict) and "tree_info" in self.model_data:
            self.tree_info = self.model_data["tree_info"]
        else:
            # Handle parsed structure or fallback pipeline elements if applicable
            self.tree_info = getattr(self.model_data, "tree_info", [])

    def _predict_tree(self, node: dict, feature_values: np.ndarray) -> float:
        if "leaf_value" in node:
            return node["leaf_value"]

        feat_idx = node.get("split_feature", 0)

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
            if decision_type == "<=" or decision_type == 2:
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
            tree_struct = tree["tree_structure"] if isinstance(tree, dict) else tree
            total_score += self._predict_tree(tree_struct, feature_vector)

        return float(np.clip(total_score, 0.0, 1.0))
