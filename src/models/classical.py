"""Mô hình cơ sở: HOG + (SVM | Random Forest | KNN)."""
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.config import HOG_SIZE, SEED

HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)


def hog_from_pil(img, size=HOG_SIZE):
    """Ảnh xám, chuẩn hóa về [0, 1], resize về size x size -> vector HOG (8100 chiều với 128x128)."""
    gray = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32) / 255.0
    return hog(gray, **HOG_PARAMS).astype(np.float32)


def hog_from_path(path, size=HOG_SIZE):
    return hog_from_pil(Image.open(path), size)


def extract_hog_features(paths, n_jobs=-1):
    feats = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(hog_from_path)(p) for p in tqdm(paths, desc="HOG"))
    return np.stack(feats)


def build_pipeline(name, pca_components=200):
    steps = [("scaler", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, random_state=SEED)))
    if name == "svm":
        clf = SVC(kernel="rbf", random_state=SEED)
    elif name == "rf":
        clf = RandomForestClassifier(random_state=SEED, n_jobs=1)
    elif name == "knn":
        clf = KNeighborsClassifier(n_jobs=1)
    else:
        raise ValueError(name)
    steps.append(("clf", clf))
    return Pipeline(steps)


PARAM_GRIDS = {
    "svm": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 1e-3, 1e-2]},
    "rf": {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 30], "clf__max_features": ["sqrt", "log2"]},
    "knn": {"clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "cosine"]},
}
