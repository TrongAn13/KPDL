# Cài đặt thuật toán ID3: entropy, information gain, TreeNode, ID3DecisionTree

import pandas as pd
import numpy as np
from math import log2
from collections import Counter


# 1. HÀM TÍNH ENTROPY 
def entropy(labels):
    """
    labels: mảng hoặc Series nhãn (0/1)
    """
    counts = Counter(labels)
    total = len(labels)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * log2(p)
    return ent


# 2. HÀM TÍNH INFORMATION GAIN
def info_gain_categorical(X_col, y):
    """
    Gain của 1 thuộc tính phân loại (chia theo từng giá trị)
    """
    base_ent = entropy(y)
    total = len(y)
    values = X_col.unique()

    cond_ent = 0.0
    for v in values:
        mask = (X_col == v)
        y_sub = y[mask]
        cond_ent += len(y_sub) / total * entropy(y_sub)

    gain = base_ent - cond_ent
    return gain


def best_threshold_numeric(X_col, y):
    """
    Tìm ngưỡng chia tốt nhất cho thuộc tính số:
    thử các ngưỡng là trung điểm giữa các giá trị đã sắp xếp
    Trả về: (best_gain, best_threshold)
    """
    base_ent = entropy(y)
    sorted_vals = np.sort(X_col.unique())

    if len(sorted_vals) <= 1:
        return 0.0, None

    best_gain = -1.0
    best_thr = None
    total = len(y)

    # Các ngưỡng thử: trung điểm giữa 2 giá trị liên tiếp
    for i in range(len(sorted_vals) - 1):
        thr = (sorted_vals[i] + sorted_vals[i + 1]) / 2
        left_mask = (X_col <= thr)
        right_mask = (X_col > thr)

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        y_left = y[left_mask]
        y_right = y[right_mask]

        ent_left = entropy(y_left)
        ent_right = entropy(y_right)

        cond_ent = (len(y_left) / total) * ent_left + \
                   (len(y_right) / total) * ent_right

        gain = base_ent - cond_ent

        if gain > best_gain:
            best_gain = gain
            best_thr = thr

    if best_gain < 0:
        best_gain = 0.0
        best_thr = None

    return best_gain, best_thr


# 3. CẤU TRÚC NODE CỦA CÂY 
class TreeNode:
    def __init__(self, *, is_leaf=False, prediction=None,
                 feature=None, threshold=None, children=None):
        """
        is_leaf: True nếu là lá
        prediction: nhãn dự đoán tại lá
        feature: tên thuộc tính dùng để chia tại node này
        threshold: nếu thuộc tính số -> chia <= threshold / > threshold
        children:
          - nếu thuộc tính phân loại: dict {giá trị: TreeNode}
          - nếu thuộc tính số: dict {"<=": node_trai, ">": node_phai}
        """
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.children = children or {}


# 4. LỚP CÂY QUYẾT ĐỊNH ID3 
class ID3DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2):
        """
        feature_types: dict {tên_cột: "categorical" hoặc "numeric"}
        max_depth: độ sâu tối đa (None = không giới hạn)
        min_samples_split: số mẫu tối thiểu để tiếp tục tách
        """
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.classes_ = None

    # HÀM TRAIN 
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = sorted(y.unique())
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # 1. Nếu tất cả nhãn giống nhau -> lá
        if len(set(y)) == 1:
            return TreeNode(is_leaf=True, prediction=y.iloc[0])

        # 2. Nếu hết thuộc tính hoặc đạt giới hạn độ sâu / số mẫu -> lá với nhãn majority
        if len(X.columns) == 0 or \
           (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split:
            majority_label = y.value_counts().idxmax()
            return TreeNode(is_leaf=True, prediction=majority_label)

        # 3. Chọn thuộc tính cho Information Gain lớn nhất
        best_feature = None
        best_gain = 0.0
        best_thr = None

        for col in X.columns:
            ftype = self.feature_types[col]
            if ftype == "categorical":
                gain = info_gain_categorical(X[col], y)
                thr = None
            else:
                gain, thr = best_threshold_numeric(X[col], y)

            if gain > best_gain and (ftype == "categorical" or thr is not None):
                best_gain = gain
                best_feature = col
                best_thr = thr

        # Nếu Gain = 0 (không cải thiện) -> lá
        if best_feature is None or best_gain <= 0:
            majority_label = y.value_counts().idxmax()
            return TreeNode(is_leaf=True, prediction=majority_label)

        # 4. Tạo node phân chia
        node = TreeNode(
            is_leaf=False,
            feature=best_feature,
            threshold=best_thr,
            children={}
        )

        ftype = self.feature_types[best_feature]

        if ftype == "categorical":
            # 1 nhánh cho mỗi giá trị
            for v in X[best_feature].unique():
                mask = (X[best_feature] == v)
                X_sub = X[mask].drop(columns=[best_feature])
                y_sub = y[mask]
                child = self._build_tree(X_sub, y_sub, depth + 1)
                node.children[v] = child
        else:
            # numeric: 2 nhánh <= threshold và > threshold
            mask_left = X[best_feature] <= best_thr
            mask_right = X[best_feature] > best_thr

            X_left = X[mask_left].copy()
            X_right = X[mask_right].copy()
            y_left = y[mask_left]
            y_right = y[mask_right]

            child_left = self._build_tree(X_left, y_left, depth + 1)
            child_right = self._build_tree(X_right, y_right, depth + 1)

            node.children["<="] = child_left
            node.children[">"] = child_right

        return node

    #  HÀM DỰ ĐOÁN 1 DÒNG 
    def predict_row(self, row):
        node = self.root
        while not node.is_leaf:
            value = row[node.feature]
            ftype = self.feature_types[node.feature]

            if ftype == "categorical":
                if value in node.children:
                    node = node.children[value]
                else:
                    # giá trị chưa gặp -> rơi về nhánh con đầu tiên
                    node = list(node.children.values())[0]
            else:  # numeric
                if value <= node.threshold:
                    node = node.children["<="]
                else:
                    node = node.children[">"]
        return node.prediction

    #  HÀM DỰ ĐOÁN NHIỀU DÒNG 
    def predict(self, X: pd.DataFrame):
        return X.apply(self.predict_row, axis=1)
    
    # HÀM IN CẤU TRÚC CÂY
    def print_tree(self, node=None, indent="", prefix=""):
        """In cấu trúc cây ra màn hình (Hỗ trợ cả số và chữ)"""
        if node is None:
            node = self.root

        # 1. Nếu là lá -> In kết quả
        if node.is_leaf:
            # Tô màu hoặc làm nổi bật kết quả
            res_str = "CÓ BỆNH (1)" if node.prediction == 1 else "KHÔNG BỆNH (0)"
            print(f"{indent}{prefix} --> {res_str}")
            return

        # 2. In tên thuộc tính đang xét
        print(f"{indent}{prefix} [Hỏi: {node.feature} ?]")

        # 3. Duyệt qua các nhánh con
        ftype = self.feature_types.get(node.feature, "categorical")
        
        if ftype == "categorical":
            for val, child in node.children.items():
                self.print_tree(child, indent + "    ", f"== {val}")
        else:
            # Nếu là số thực, in ngưỡng so sánh
            self.print_tree(node.children["<="], indent + "    ", f"<= {node.threshold:.2f}")
            self.print_tree(node.children[">"], indent + "    ", f">  {node.threshold:.2f}")
        
    # HÀM GIẢI THÍCH LÝ DO DỰ ĐOÁN
    def predict_one_with_reason(self, row):
        """Dự đoán 1 dòng và trả về kèm đường dẫn lý do"""
        node = self.root
        path = [] # Lưu lại lịch sử đi
        
        while not node.is_leaf:
            val = row[node.feature]
            ftype = self.feature_types.get(node.feature, "categorical")
            
            # Ghi lại bước đi
            if ftype == "categorical":
                decision_step = f"{node.feature} == '{val}'"
                path.append(decision_step)
                
                if val in node.children:
                    node = node.children[val]
                else:
                    # Gặp giá trị lạ -> đi nhánh đầu tiên
                    node = list(node.children.values())[0]
                    path.append("(Giá trị lạ -> Dùng nhánh mặc định)")
            else:
                # Logic cho số thực
                if val <= node.threshold:
                    path.append(f"{node.feature} ({val}) <= {node.threshold:.2f}")
                    node = node.children["<="]
                else:
                    path.append(f"{node.feature} ({val}) > {node.threshold:.2f}")
                    node = node.children[">"]
        
        return node.prediction, path
