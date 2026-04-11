import os
import math
import pandas as pd
import numpy as np
import copy
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import cv2
from tqdm import tqdm
import pickle
from torchvision import transforms

def build_global_encoders(root_dir, num_age_bins=5):
    """users.csv 全体からジャンル非依存のエンコーダとage bin境界を構築する."""
    maked_dir = os.path.join(root_dir, 'maked')
    users = pd.read_csv(os.path.join(maked_dir, 'users.csv'))

    encoded_trait_columns = ['age', 'gender', 'edu', 'nationality', 'art_learn', 'fashion_learn', 'photoVideo_learn']

    # Age bins: 全ユーザーの min/max から算出
    min_age, max_age = users['age'].min(), users['age'].max()
    global_age_bins = np.linspace(min_age, max_age, num=num_age_bins + 1)
    age_labels = [f'{int(global_age_bins[i])}-{int(global_age_bins[i+1])-1}' for i in range(num_age_bins)]

    # Trait encoders: 全ユーザーのカテゴリをソート順で構築
    global_trait_encoders = []
    for attr in encoded_trait_columns:
        if attr == 'age':
            encoder = {label: idx for idx, label in enumerate(age_labels)}
        else:
            unique_vals = sorted(users[attr].dropna().unique(), key=str)
            encoder = {val: idx for idx, val in enumerate(unique_vals)}
        global_trait_encoders.append(encoder)

    return global_trait_encoders, global_age_bins


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, genre=None, backbone=None, max_frames=None, is_train=False,
                 global_trait_encoders=None, global_age_bins=None):
        """
        Args:
            root_dir (string): Directory with all the images and CSVs.
            genre (string, optional): Genre of the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.genre = genre
        self.backbone = backbone
        self.use_image = (self.backbone != "i3d")
        self.root_dir = root_dir
        self.transform = transform
        # max_frames: if not None, enforce fixed temporal length for videos.
        # is_train: if True, sample a random contiguous window when T > max_frames.
        self.max_frames = max_frames
        self.is_train = is_train
        self.maked_dir = os.path.join(root_dir, 'maked')

        self.ratings_data = pd.read_csv(os.path.join(self.maked_dir, 'ratings.csv'))
        self.ratings_data = self.ratings_data[self.ratings_data['genre'] == self.genre]
        if {'user_id', 'sample_file'}.issubset(self.ratings_data.columns):
            self.ratings_data = self.ratings_data.drop_duplicates(subset=['user_id', 'sample_file'], keep='first')
        self.users_data = pd.read_csv(os.path.join(self.maked_dir, 'users.csv'))
        self.users_data = self.users_data.set_index("user_id")
        self.data = pd.merge(self.ratings_data, self.users_data, on="user_id", how="left")

        # sceneryの場合、use_videoに応じてQIPファイルを切り替え
        if self.genre == 'scenery' and self.backbone == 'i3d':
            # use_video=True case: scenery with i3d backbone uses QIP_scenery_video.csv
            qip_filename = f'QIP_{self.genre}_video.csv'
        elif self.genre == 'scenery':
            # use_video=False case: scenery with resnet50 backbone uses QIP_scenery_image.csv
            qip_filename = f'QIP_{self.genre}_image.csv'
        else:
            # Other genres (art, fashion) use default QIP_{genre}.csv
            qip_filename = f'QIP_{self.genre}.csv'
        self.qip_df = pd.read_csv(os.path.join(self.maked_dir, qip_filename))
        self.qip_df = self.qip_df.set_index("img_file")
        # self.samples_dir = os.path.join(root_dir, 'samples')
        # Determine samples directory based on genre and backbone (video vs image)
        if self.genre == 'scenery' and self.backbone == 'i3d':
            # use_video=True case: scenery with i3d backbone uses scenery_video
            self.samples_dir = "/home/hayashi0884/proj-xpass/data/samples/scenery_video"
        elif self.genre == 'scenery':
            # use_video=False case: scenery with resnet50 backbone uses scenery_image
            self.samples_dir = "/home/hayashi0884/proj-xpass/data/samples/scenery_image"
        else:
            # Other genres (art, fashion) use default structure
            self.samples_dir = "/home/hayashi0884/proj-xpass/data/samples"

        # Load and preprocess data
        # Encode text columns
        self.score_fields  = [f'Q{i}' for i in range(1, 11)] + ['art_interest', 'fashion_interest', 'photoVideo_interest']
        self.metadata = ['Aesthetic', 'fold', 'user_id']
        self.encoded_trait_columns = ['age', 'gender', 'edu', 'nationality', 'art_learn', 'fashion_learn', 'photoVideo_learn']

        # Categorize ages into 5 bins
        if global_age_bins is not None:
            interval_edges = global_age_bins
        else:
            min_age, max_age = self.data['age'].min(), self.data['age'].max()
            interval_edges = np.linspace(min_age, max_age, num=6)
        interval_labels = [f'{int(interval_edges[i])}-{int(interval_edges[i+1])-1}' for i in range(len(interval_edges)-1)]
        age_intervals = pd.cut(self.data['age'], bins=interval_edges, labels=interval_labels, include_lowest=True)
        self.data['age'] = age_intervals

        if global_trait_encoders is not None:
            self.trait_encoders = global_trait_encoders
        else:
            self.trait_encoders = [
                {group: idx for idx, group in enumerate(sorted(self.data[attribute].dropna().unique(), key=str))}
                for attribute in self.encoded_trait_columns
            ]

        # qip_dataを列ごと（特徴量ごと）にドメイン全体の統計量でminmax正規化
        qip_min = self.qip_df.min()
        qip_max = self.qip_df.max()
        qip_range = qip_max - qip_min
        qip_range[qip_range == 0] = 1  # ゼロ除算回避（全サンプルで同一値の列はそのまま）
        self.qip_df = (self.qip_df - qip_min) / qip_range
        self.qip_data = self.qip_df.to_dict(orient="index")

    def one_hot_personality(self, trait_value):
        # Enforce valid integer inputs; raise on None/NaN/non-convertible values
        if trait_value is None:
            raise ValueError("one_hot_personality: trait_value is None")
        if isinstance(trait_value, float) and math.isnan(trait_value):
            raise ValueError("one_hot_personality: trait_value is NaN")
        try:
            idx = int(trait_value)
        except Exception as e:
            raise ValueError(f"one_hot_personality: trait_value '{trait_value}' is not convertible to int") from e

        # Validate range [0, 6] for 7 classes
        if not (0 <= idx <= 6):
            raise ValueError(f"one_hot_personality: trait_value index out of range [0,6]: {idx}")

        return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=7)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = {
            attribute: torch.tensor(self.data.iloc[idx][attribute], dtype=torch.int) for attribute in self.metadata
        }

        sample.update({
            f'{attribute}_onehot': self.one_hot_personality(self.data.iloc[idx][attribute]) for attribute in self.score_fields
        })

        sample.update({
            trait: torch.tensor(encoder[self.data.iloc[idx][trait]], dtype=torch.int)
            for trait, encoder in zip(self.encoded_trait_columns, self.trait_encoders)
        })

        for trait, encoder in zip(self.encoded_trait_columns, self.trait_encoders):
            original_val = sample[trait]
            onehot_val = F.one_hot(original_val.long(), num_classes=len(encoder))
            sample[f'{trait}_onehot'] = onehot_val

        # del non-one hot trait
        for trait in self.encoded_trait_columns:
            del sample[trait]

        # For scenery, samples_dir already includes genre info (scenery_video/scenery_image)
        # For other genres, append genre to samples_dir
        sample_file = self.data.iloc[idx]['sample_file']
        if self.genre == 'scenery':
            # If using image (resnet50) for scenery, convert .mp4 to .jpg
            if self.use_image and sample_file.endswith('.mp4'):
                sample_file = sample_file.replace('.mp4', '.jpg')
            sample_path = os.path.join(self.samples_dir, sample_file)
        else:
            sample_path = os.path.join(self.samples_dir, self.genre, sample_file)
        sample['sample_path'] = sample_path
        sample['sample_file'] = sample_file

        # 統合: image / video を同一メソッドで扱う。
        if self.use_image:
            # 画像読み込み（従来通り）
            sample['image'] = Image.open(sample_path).convert('RGB')
            if self.transform:
                sample['image'] = self.transform(sample['image'])
        else:
            # 動画読み込み（VideoDataset の処理を統合）
            cap = cv2.VideoCapture(sample_path)
            frames = []
            # Read all frames sequentially. FPS reduction is handled in
            # preprocessing (user chose to lower raw video FPS beforehand).

            # If a transform is provided, sample a single random seed for this video
            # and re-seed RNGs before applying transform to each sampled frame so that
            # stochastic transforms produce the same result across frames.
            video_seed = None
            if self.transform:
                video_seed = int(np.random.randint(0, 2**31 - 1))
            frame_idx = 0
            while True:
                ret, frm = cap.read()
                if not ret:
                    break

                # Keep every frame (no on-the-fly downsampling).

                # OpenCV は BGR を返すので RGB に変換
                try:
                    frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
                img = Image.fromarray(frm)

                if self.transform:
                    # Re-seed Python, NumPy and PyTorch RNGs so that the
                    # transform's internal randomness is identical for every sampled
                    # frame of this video (video_seed is sampled once above).
                    random.seed(video_seed)
                    np.random.seed(video_seed)
                    torch.manual_seed(video_seed)
                    img = self.transform(img)

                if isinstance(img, torch.Tensor):
                    frames.append(img)
                else:
                    arr = np.array(img)
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    frames.append(torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32))

                frame_idx += 1
            cap.release()

            if len(frames) == 0:
                sample['image'] = torch.empty(0)
            else:
                # frames: list of [C, H, W] tensors -> stacked shape [T, C, H, W]
                video_tensor = torch.stack(frames, dim=0).float()
                # Convert to [C, T, H, W] so that after DataLoader batching
                # we'll have [B, C, T, H, W], which matches 3D backbones' expectation.
                video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()

                # If requested, enforce fixed temporal length `max_frames`.
                # Training: take a random contiguous window of length max_frames.
                # Eval: take a center contiguous window. If shorter, pad with zeros on the temporal axis.
                if (self.max_frames is not None) and (self.max_frames > 0):
                    C, T, H, W = video_tensor.shape
                    mf = int(self.max_frames)
                    if T > mf:
                        if self.is_train:
                            start = random.randint(0, T - mf)
                        else:
                            start = max(0, (T - mf) // 2)
                        video_tensor = video_tensor[:, start:start + mf, :, :]
                    elif T < mf:
                        pad_len = mf - T
                        pad = torch.zeros((C, pad_len, H, W), dtype=video_tensor.dtype)
                        video_tensor = torch.cat([video_tensor, pad], dim=1)

                sample['image'] = video_tensor
            # # check video shape
            # print(sample['video'].shape)

        sample['QIP'] = self.qip_data[sample['sample_file']]
        sample['user_id'] = self.data.iloc[idx]['user_id']
        return sample

def create_GIAA_split_dataset(dataset, fold_id):
    root_dir = dataset.root_dir
    genre = dataset.genre

    # fold_id is repurposed as dataset version
    version = str(getattr(dataset, 'dataset_ver', fold_id))
    train_file_path = os.path.join(root_dir, 'split', version, genre, 'train_images_GIAA.txt')
    val_file_path = os.path.join(root_dir, 'split', version, genre, 'val_images_GIAA.txt')
    # PIAA split files (used to exclude overlapping user_ids)
    piaa_train_file_path = os.path.join(root_dir, 'split', version, genre, 'train_PIAA.txt')
    piaa_val_file_path = os.path.join(root_dir, 'split', version, genre, 'val_PIAA.txt')
    piaa_test_file_path = os.path.join(root_dir, 'split', version, genre, 'test_PIAA.txt')

    print('Read Image Set')
    with open(train_file_path, "r") as train_file:
        train_image_names = train_file.read().splitlines()
    with open(val_file_path, "r") as val_file:
        val_image_names = val_file.read().splitlines()

    # Collect user_ids present in any PIAA split (<train,val,test>_PIAA.txt)
    piaa_user_ids = set()
    def _parse_user_ids(lines):
        ids = []
        for ln in lines:
            ln = ln.strip()
            if ln == '':
                continue
            parts = ln.split()
            if len(parts) < 2:
                # Skip malformed lines silently; PIAA files are expected as 'user_id filename'
                continue
            ids.append(parts[0])  # keep as string for robust matching
        return ids

    try:
        if os.path.exists(piaa_train_file_path):
            with open(piaa_train_file_path, "r") as f:
                piaa_user_ids.update(_parse_user_ids(f.read().splitlines()))
        if os.path.exists(piaa_val_file_path):
            with open(piaa_val_file_path, "r") as f:
                piaa_user_ids.update(_parse_user_ids(f.read().splitlines()))
        if os.path.exists(piaa_test_file_path):
            with open(piaa_test_file_path, "r") as f:
                piaa_user_ids.update(_parse_user_ids(f.read().splitlines()))
    except Exception:
        # If any error occurs reading PIAA files, proceed without exclusion.
        piaa_user_ids = set()

    train_dataset_GIAA, val_dataset_GIAA = copy.deepcopy(dataset), copy.deepcopy(dataset)
    # First filter by image names
    train_dataset_GIAA.data = train_dataset_GIAA.data[train_dataset_GIAA.data['sample_file'].isin(train_image_names)]
    val_dataset_GIAA.data = val_dataset_GIAA.data[val_dataset_GIAA.data['sample_file'].isin(val_image_names)]

    # Then exclude any rows whose user_id is present in PIAA splits
    if len(piaa_user_ids) > 0:
        train_dataset_GIAA.data = train_dataset_GIAA.data.assign(user_id_str=train_dataset_GIAA.data['user_id'].astype(str))
        val_dataset_GIAA.data   = val_dataset_GIAA.data.assign(user_id_str=val_dataset_GIAA.data['user_id'].astype(str))
        train_dataset_GIAA.data = train_dataset_GIAA.data[~train_dataset_GIAA.data['user_id_str'].isin(piaa_user_ids)].drop(columns=['user_id_str'])
        val_dataset_GIAA.data   = val_dataset_GIAA.data[~val_dataset_GIAA.data['user_id_str'].isin(piaa_user_ids)].drop(columns=['user_id_str'])

    return train_dataset_GIAA, val_dataset_GIAA

def create_GIAA_test_dataset(dataset, fold_id):
    """Load GIAA test split from test_images_GIAA.txt."""
    root_dir = dataset.root_dir
    genre = dataset.genre
    version = str(getattr(dataset, 'dataset_ver', fold_id))
    test_file_path = os.path.join(root_dir, 'split', version, genre, 'test_images_GIAA.txt')

    with open(test_file_path, "r") as f:
        test_image_names = f.read().splitlines()

    test_dataset_GIAA = copy.deepcopy(dataset)
    test_dataset_GIAA.data = test_dataset_GIAA.data[test_dataset_GIAA.data['sample_file'].isin(test_image_names)]
    return test_dataset_GIAA

def create_GIAA_user_split_dataset(dataset, fold_id):
    """Split GIAA dataset by user_id using train_users_GIAA.txt / val_users_GIAA.txt."""
    root_dir = dataset.root_dir
    genre = dataset.genre
    version = str(getattr(dataset, 'dataset_ver', fold_id))

    train_user_file = os.path.join(root_dir, 'split', version, genre, 'train_users_GIAA.txt')
    val_user_file = os.path.join(root_dir, 'split', version, genre, 'val_users_GIAA.txt')

    with open(train_user_file, "r") as f:
        train_user_ids = set(line.strip() for line in f if line.strip())
    with open(val_user_file, "r") as f:
        val_user_ids = set(line.strip() for line in f if line.strip())

    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)

    train_dataset.data = train_dataset.data[
        train_dataset.data['user_id'].astype(str).isin(train_user_ids)
    ]
    val_dataset.data = val_dataset.data[
        val_dataset.data['user_id'].astype(str).isin(val_user_ids)
    ]

    return train_dataset, val_dataset

def create_PIAA_split_dataset(dataset, fold_id):
    root_dir = dataset.root_dir
    genre = dataset.genre

    version = str(getattr(dataset, 'dataset_ver', fold_id))
    train_file_path = os.path.join(root_dir, 'split', version, genre, 'train_PIAA.txt')
    val_file_path = os.path.join(root_dir, 'split', version, genre, 'val_PIAA.txt')
    test_file_path = os.path.join(root_dir, 'split', version, genre, 'test_PIAA.txt')

    with open(train_file_path, "r") as train_file:
        train_lines = train_file.read().splitlines()
    with open(val_file_path, "r") as val_file:
        val_lines = val_file.read().splitlines()
    with open(test_file_path, "r") as test_file:
        test_lines = test_file.read().splitlines()

    # Parse lines expecting 'user_id\tfilename' (or whitespace-separated). Return list of (user_id, filename).
    def _parse_userfile_pairs(lines):
        pairs = []
        for ln in lines:
            ln = ln.strip()
            if ln == '':
                continue
            parts = ln.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed split line (expected 'user_id filename'): '{ln}'")
            uid = str(parts[0])
            fname = parts[-1]
            pairs.append((uid, fname))
        return pairs

    train_pairs = pd.DataFrame(_parse_userfile_pairs(train_lines), columns=['user_id', 'sample_file'])
    val_pairs = pd.DataFrame(_parse_userfile_pairs(val_lines), columns=['user_id', 'sample_file'])
    test_pairs = pd.DataFrame(_parse_userfile_pairs(test_lines), columns=['user_id', 'sample_file'])

    train_dataset_PIAA, val_dataset_PIAA, test_dataset_PIAA = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)

    # Filter datasets by exact (user_id, sample_file) pairs using an inner merge.
    train_dataset_PIAA.data = train_dataset_PIAA.data.assign(user_id=train_dataset_PIAA.data['user_id'].astype(str))
    train_dataset_PIAA.data = train_dataset_PIAA.data.merge(train_pairs, on=['user_id', 'sample_file'], how='inner')

    val_dataset_PIAA.data = val_dataset_PIAA.data.assign(user_id=val_dataset_PIAA.data['user_id'].astype(str))
    val_dataset_PIAA.data = val_dataset_PIAA.data.merge(val_pairs, on=['user_id', 'sample_file'], how='inner')

    test_dataset_PIAA.data = test_dataset_PIAA.data.assign(user_id=test_dataset_PIAA.data['user_id'].astype(str))
    test_dataset_PIAA.data = test_dataset_PIAA.data.merge(test_pairs, on=['user_id', 'sample_file'], how='inner')

    train_dataset_PIAA.data['user_id'] = pd.to_numeric(train_dataset_PIAA.data['user_id'], errors='coerce').astype('Int64')
    val_dataset_PIAA.data['user_id']   = pd.to_numeric(val_dataset_PIAA.data['user_id'], errors='coerce').astype('Int64')
    test_dataset_PIAA.data['user_id']  = pd.to_numeric(test_dataset_PIAA.data['user_id'], errors='coerce').astype('Int64')

    return train_dataset_PIAA, val_dataset_PIAA, test_dataset_PIAA


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(backbone=None):
    """Return (train_transform, test_transform) with backbone-appropriate normalization."""
    if backbone in ('clip_rn50', 'clip_vit_b16'):
        normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    elif backbone in ('resnet50', 'vit_b_16', 'i3d'):
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        normalize = None

    train_ops = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
    ]
    test_ops = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        train_ops.append(normalize)
        test_ops.append(normalize)

    return transforms.Compose(train_ops), transforms.Compose(test_ops)

def load_data(args, global_trait_encoders=None, global_age_bins=None):
    # Dataset transformations
    root_dir = args.root_dir
    genre = args.genre
    backbone = getattr(args, 'backbone', None)
    dataset_ver = getattr(args, 'dataset_ver', 'v1')

    train_transform, test_transform = get_transforms(backbone)

    # Create datasets with the appropriate transformations
    all_dataset = ImageDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone,
                               global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
    # Attach dataset version to the dataset instance for downstream path resolution
    setattr(all_dataset, 'dataset_ver', dataset_ver)
    train_giaa_raw, val_giaa_raw = create_GIAA_split_dataset(all_dataset, dataset_ver)
    train_piaa_dataset, val_piaa_dataset, test_piaa_dataset = create_PIAA_split_dataset(all_dataset, dataset_ver)

    # User-based GIAA split for PIAA pretraining (prevents user leakage)
    train_giaa_usersplit, val_giaa_usersplit = create_GIAA_user_split_dataset(all_dataset, dataset_ver)
    pretrain_train_data = train_giaa_usersplit.data
    pretrain_val_data = val_giaa_usersplit.data

    """Precompute"""
    # Store precomputed cache under a versioned subfolder: data/cash/<version>/<genre>_dataset_pkl/
    version = str(dataset_ver) if dataset_ver is not None else 'v1'
    pkl_dir_base = os.path.join(root_dir, 'cash', version)
    pkl_dir = os.path.join(pkl_dir_base, f'{genre}_dataset_pkl')
    ensure_dir_exists(pkl_dir)
    map_file = os.path.join(pkl_dir, 'trainset_image_dct.pkl')
    # For training datasets enforce a fixed temporal length by default
    enc_kwargs = dict(global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
    train_giaa_hist_dataset = Image_GIAA_HistogramDataset(root_dir, transform=train_transform,
        genre=genre, backbone=backbone, data=train_giaa_raw.data, map_file=map_file,
    precompute_file=os.path.join(pkl_dir, 'trainset_GIAA_dct.pkl'),
        max_frames=16, is_train=True, **enc_kwargs)

    val_mapfile = os.path.join(pkl_dir, 'valset_image_dct.pkl')
    val_precompute_file = os.path.join(pkl_dir, 'valset_GIAA_dct.pkl')
    val_giaa_hist_dataset = Image_GIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, \
        backbone=backbone, data=val_giaa_raw.data, map_file=val_mapfile, precompute_file=val_precompute_file,
        max_frames=None, is_train=False, **enc_kwargs)

    train_giaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone, data=pretrain_train_data, max_frames=16, is_train=True, **enc_kwargs)
    val_giaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, backbone=backbone, data=pretrain_val_data, max_frames=None, is_train=False, **enc_kwargs)
    train_piaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone, data=train_piaa_dataset.data, max_frames=16, is_train=True, **enc_kwargs)
    val_piaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, backbone=backbone, data=val_piaa_dataset.data, max_frames=None, is_train=False, **enc_kwargs)
    test_piaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, backbone=backbone, data=test_piaa_dataset.data, max_frames=None, is_train=False, **enc_kwargs)

    # print num of datasets + unique user counts (if available)
    def _unique_users(ds):
        try:
            df = getattr(ds, 'data', None)
            if df is not None and 'user_id' in df.columns:
                return int(df['user_id'].astype(str).nunique())
        except Exception:
            pass
        return None

    ug_train = _unique_users(train_giaa_hist_dataset)
    ug_val = _unique_users(val_giaa_hist_dataset)
    upre_train = _unique_users(train_giaa_dataset)
    upre_val = _unique_users(val_giaa_dataset)
    ufine_train = _unique_users(train_piaa_dataset)
    ufine_val = _unique_users(val_piaa_dataset)
    ufine_test = _unique_users(test_piaa_dataset)

    def _fmt_users(n):
        return f', unique_users={n}' if n is not None else ''

    print(f'Train size GIAA: {len(train_giaa_hist_dataset)}{_fmt_users(ug_train)}, Val size GIAA: {len(val_giaa_hist_dataset)}{_fmt_users(ug_val)}')
    print(f'Train size PIAA-Pre: {len(train_giaa_dataset)}{_fmt_users(upre_train)}, Val size PIAA-Pre: {len(val_giaa_dataset)}{_fmt_users(upre_val)}')
    print(f'Train size PIAA-Fine: {len(train_piaa_dataset)}{_fmt_users(ufine_train)}, Val size PIAA-Fine: {len(val_piaa_dataset)}{_fmt_users(ufine_val)}, Test size PIAA-Fine: {len(test_piaa_dataset)}{_fmt_users(ufine_test)}')
    return train_giaa_hist_dataset, train_piaa_dataset, train_giaa_dataset, val_giaa_hist_dataset, val_piaa_dataset, val_giaa_dataset, test_piaa_dataset

def load_data_giaa_only(args):
    """Load GIAA train/val/test datasets only (no PIAA). Used for backbone validation."""
    root_dir = args.root_dir
    genre = args.genre
    backbone = getattr(args, 'backbone', None)
    dataset_ver = getattr(args, 'dataset_ver', 'v1')

    train_transform, test_transform = get_transforms(backbone)

    all_dataset = ImageDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone)
    setattr(all_dataset, 'dataset_ver', dataset_ver)

    train_giaa_raw, val_giaa_raw = create_GIAA_split_dataset(all_dataset, dataset_ver)
    test_giaa_raw = create_GIAA_test_dataset(all_dataset, dataset_ver)

    # Precompute cache directory
    version = str(dataset_ver) if dataset_ver is not None else 'v1'
    pkl_dir = os.path.join(root_dir, 'cash', version, f'{genre}_dataset_pkl')
    ensure_dir_exists(pkl_dir)

    train_giaa_dataset = Image_GIAA_HistogramDataset(
        root_dir, transform=train_transform, genre=genre, backbone=backbone,
        data=train_giaa_raw.data,
        map_file=os.path.join(pkl_dir, 'trainset_image_dct.pkl'),
        precompute_file=os.path.join(pkl_dir, 'trainset_GIAA_dct.pkl'),
        max_frames=16, is_train=True,
    )
    val_giaa_dataset = Image_GIAA_HistogramDataset(
        root_dir, transform=test_transform, genre=genre, backbone=backbone,
        data=val_giaa_raw.data,
        map_file=os.path.join(pkl_dir, 'valset_image_dct.pkl'),
        precompute_file=os.path.join(pkl_dir, 'valset_GIAA_dct.pkl'),
        max_frames=None, is_train=False,
    )
    test_giaa_dataset = Image_GIAA_HistogramDataset(
        root_dir, transform=test_transform, genre=genre, backbone=backbone,
        data=test_giaa_raw.data,
        map_file=os.path.join(pkl_dir, 'testset_image_dct.pkl'),
        precompute_file=os.path.join(pkl_dir, 'testset_GIAA_dct.pkl'),
        max_frames=None, is_train=False,
    )

    print(f'[val_backbone] Train GIAA: {len(train_giaa_dataset)}, Val GIAA: {len(val_giaa_dataset)}, Test GIAA: {len(test_giaa_dataset)}')
    return train_giaa_dataset, val_giaa_dataset, test_giaa_dataset

class Image_GIAA_HistogramDataset(ImageDataset):
    def __init__(self, root_dir, transform=None, genre=None, backbone=None, data=None, map_file=None, precompute_file=None, max_frames=None, is_train=False,
                 global_trait_encoders=None, global_age_bins=None):
        # pass max_frames/is_train through to the underlying ImageDataset
        super().__init__(root_dir, transform, genre, backbone=backbone, max_frames=max_frames, is_train=is_train,
                         global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
        if data is not None:
            self.data = data

        if map_file and os.path.exists(map_file):
            print('Loading image to indices map from file...')
            self.image_to_indices_map = self._load_map(map_file)
            self.unique_images = [img for img in self.image_to_indices_map.keys()]
        else:
            self.image_to_indices_map = dict()
            for image in tqdm(self.data['sample_file'].unique(), desc='Processing images'):
                indices_for_image = [i for i, img in enumerate(self.data['sample_file']) if img == image]
                if any(not idx < len(self.data) for idx in indices_for_image):
                    print(indices_for_image)
                    raise Exception('Index out of bounds for the data.')
                if len(indices_for_image) > 0:  # Only add if there are indices for the image
                    self.image_to_indices_map[image] = indices_for_image

            self.unique_images = [img for img in self.image_to_indices_map.keys()]  # Filtered unique_images

            if map_file:
                print(f"Saving image to indices map to {map_file}")
                self._save_map(map_file)

        # If precompute_file is given and exists, load it, otherwise recompute the data
        if precompute_file and os.path.exists(precompute_file):
            print(f'Loading precomputed data from {precompute_file}...')
            self.load(precompute_file)
        else:
            self.precompute_data()
            if precompute_file:
                print(f"Saving precomputed data to {precompute_file}")
                self.save(precompute_file)

    def precompute_data(self):
        self.precomputed_data = []
        for idx in tqdm(range(len(self)), desc='Precompute images'):
            self.precomputed_data.append(self._compute_item(idx))

    def _compute_item(self, idx):
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]

        # Assuming maximum scores for response and VAIAK/2VAIAK ratings for proper one-hot encoding
        max_response_score = 7  # Adjust based on your data

        # Initialize accumulators for one-hot encoded vectors
        accumulated_response = torch.zeros(max_response_score, dtype=torch.float32)

        for ai in associated_indices:
            # Avoid calling super().__getitem__ (which may load images/videos)
            # for every associated rating. Instead, read the required fields
            # directly from the underlying dataframe for efficiency.
            row = self.data.iloc[ai]
            round_score = int(row['Aesthetic'])
            response_one_hot = F.one_hot(torch.tensor(round_score), num_classes=max_response_score)
            accumulated_response += response_one_hot

        # Prepare the final accumulated histogram for the image
        accumulated_histogram = {'Aesthetic': accumulated_response}

        # Average out histograms over the number of samples
        total_samples = len(associated_indices)
        accumulated_histogram['Aesthetic'] /= total_samples
        accumulated_histogram['n_samples'] = total_samples
        # Use the image filename (unique_images) as the representative sample_file
        accumulated_histogram['sample_file'] = self.unique_images[idx]

        return accumulated_histogram

    def __getitem__(self, idx):
        item_data = copy.deepcopy(self.precomputed_data[idx])
        img_sample = super().__getitem__(self.image_to_indices_map[self.unique_images[idx]][0])
        inherit_list = ['image', 'sample_file']
        for item in inherit_list:
            item_data[item] = img_sample[item]
        item_data['traits'] = torch.tensor([i for i in range(10)])  # Dummy placeholder for traits
        item_data['QIP'] = torch.tensor([i for i in range(10)])  # Dummy placeholder for QIP

        return item_data

    def _save_map(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.image_to_indices_map, f)

    def _load_map(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.image_to_indices_map)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.precomputed_data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.precomputed_data = pickle.load(f)

def collate_fn(batch):
    # print(len(batch[0]['traits']))
    # print(len(batch[0]['QIP']))
    batch_dict = {
        'image': torch.stack([item['image'] for item in batch]),
        'Aesthetic': torch.stack([item['Aesthetic'] for item in batch]),
        'traits': torch.stack([item['traits'] for item in batch]),
        'QIP': torch.stack([item['QIP'] for item in batch]),
    }
    # include optional fields if present in samples
    optional_keys = ['user_id']
    for k in optional_keys:
        if k in batch[0]:
            try:
                batch_dict[k] = torch.stack([item[k] for item in batch])
            except Exception:
                # fallback: collect as list (for non-tensor items)
                batch_dict[k] = [item[k] for item in batch]
    return batch_dict

class Image_PIAA_HistogramDataset(ImageDataset):
    def __init__(self, root_dir, transform=None, data=None, genre=None, backbone=None, max_frames=None, is_train=False,
                 global_trait_encoders=None, global_age_bins=None):
        # pass max_frames/is_train through to the underlying ImageDataset
        super().__init__(root_dir, transform, genre=genre, backbone=backbone, max_frames=max_frames, is_train=is_train,
                         global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
        if data is not None:
            self.data = data

    def __getitem__(self, idx):
        # Assuming maximum scores for response and VAIAK/2VAIAK ratings for proper one-hot encoding
        max_response_score = 7  # Adjust based on your data

        # Initialize accumulators for one-hot encoded vectors
        sample = super().__getitem__(idx)

        # One-hot encode 'response' and accumulate
        # round_score = int(sample['Aesthetic']) / (max_response_score - 1)
        round_score = int(sample['Aesthetic'])
        # Prepare the final accumulated histogram for the image
        accumulated_histogram = {
            'Aesthetic': round_score,
        }

        # Build combined trait vector from score_fields (Q*_onehot) and encoded traits (e.g. age_onehot)
        # Keep it simple: concatenate available one-hot tensors in sample
        score_vecs = []
        trait_vecs = []
        for k, v in sample.items():
            if k.endswith('_onehot') and k.startswith('Q'):
                score_vecs.append(v.float())
            elif k.endswith('_onehot') and not k.startswith('Q'):
                trait_vecs.append(v.float())

        try:
            combined = torch.cat(score_vecs + trait_vecs) if (len(score_vecs) + len(trait_vecs)) > 0 else torch.tensor([])
        except Exception:
            # Fallback: empty tensor on failure
            combined = torch.tensor([], dtype=torch.float32)

        accumulated_histogram['traits'] = combined
        accumulated_histogram['n_samples'] = 1

        # --- Min-max normalize Aesthetic to [0,1] (min=0, max=6) ---
        # round_score は int で与えられる想定なので直接正規化する（安全策としてフォールバックあり）
        scalar = float(round_score)
        accumulated_histogram['Aesthetic'] = torch.tensor([scalar / (max_response_score - 1)], dtype=torch.float32)

        # --- Flatten QIP dict into single 1-D tensor ---
        # Keys are assumed common and present for all samples (per user's note).
        qip = sample.get('QIP', {})
        parts = []
        # Use sorted keys for deterministic ordering
        for k in sorted(qip.keys()):
            v = qip[k]
            if isinstance(v, torch.Tensor):
                parts.append(v.view(-1).float())
            else:
                # torch.tensor will handle scalars, lists, numpy arrays, etc.
                parts.append(torch.tensor(v, dtype=torch.float32).view(-1))

        qip_tensor = torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)
        accumulated_histogram['QIP'] = qip_tensor

        # inherit a minimal set of fields (avoid copying the dict 'QIP')
        inherit_list = ['image', 'sample_file', 'user_id']
        for item in inherit_list:
            if item in sample:
                accumulated_histogram[item] = sample[item]

        accumulated_histogram['genre'] = self.genre

        return accumulated_histogram
