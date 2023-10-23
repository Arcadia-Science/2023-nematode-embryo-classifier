from pathlib import Path
import cv2
import numpy as np


def compute_morphodynamics(movie, features=None, t_window=5, normalize_features=True):
    if features is None:
        features = [
            "moving_std",
            "moving_mean",
            "optical_flow",
            "raw",
        ]
    # Assuming movie is a T*XY numpy array
    T, _, _, Y, X = movie.shape
    edge_frames = int(t_window // 2)

    # Parameters for calculating optical flow.
    pyr_scale = 0.5
    pyr_levels = 1
    win_size = Y // 10  # We want to measure large scale changes in the embryo shape.
    iterations = 5
    poly_n = 7
    poly_sigma = 1.5
    options = 0

    feature_imgs = np.random.random((T, len(features), 1, Y, X)).astype(np.float32)
    # Store data in the usual TCZYX format.

    for t_idx in range(edge_frames, T - edge_frames):
        for c_idx, channel in enumerate(features):
            if channel == "moving_std":
                feature_imgs[t_idx, c_idx, 0] = np.std(
                    movie[t_idx - edge_frames : t_idx + edge_frames],
                    axis=0,
                )
            elif channel == "moving_mean":
                feature_imgs[t_idx, c_idx, 0] = np.mean(
                    movie[t_idx - edge_frames : t_idx + edge_frames],
                    axis=0,
                )
            elif channel == "optical_flow":
                # We measure optical flow between frames neighboring current frame.

                prev_frame = movie[t_idx - 1].squeeze().astype(np.float32)
                curr_frame = movie[t_idx + 1].squeeze().astype(np.float32)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame,
                    curr_frame,
                    None,
                    pyr_scale,
                    pyr_levels,
                    win_size,
                    iterations,
                    poly_n,
                    poly_sigma,
                    options,
                )
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                feature_imgs[t_idx, c_idx, 0] = magnitude
            elif channel == "raw":
                feature_imgs[t_idx, c_idx, 0] = movie[t_idx]

    if normalize_features:
        t_range = range(edge_frames, T - edge_frames)

        for c_idx, channel in enumerate(features):
            flattened_features = feature_imgs[t_range, c_idx].flatten()
            if channel == "optical_flow":
                # Optical flow max is sensitive to other embryos moving into scene.
                # We use the 99th percentile instead.
                max_val = np.percentile(flattened_features, 99)
                min_val = np.min(flattened_features.flatten())
            else:
                max_val = np.max(flattened_features)
                min_val = np.min(flattened_features)

            feature_imgs[t_range, c_idx, 0] = (feature_imgs[t_range, c_idx, 0] - min_val) / (
                max_val - min_val
            )

    return feature_imgs, features


def get_movie_paths(
    database_path: Path,
    strain: str,
    perturbation: str,
    date_stamp: str,
    FOVs: list[int] = [1],
):
    movie_paths = []
    # List all FOVs.
    all_fovs = Path(
        database_path,
        f"{date_stamp}_{strain}_{perturbation}",
    ).expanduser()

    # Concatenate paths to all embryos within each FOV.
    for fov in FOVs:
        movie_paths_fov = list(all_fovs.glob(f"{date_stamp}_{fov}/embryo*"))
        if not movie_paths_fov:
            print(
                "No movie found at "
                f"{database_path}/{date_stamp}_{strain}_{perturbation}/{date_stamp}_{fov}."
                "Check the date stamp and FOV numbers."
            )
        elif movie_paths:
            movie_paths = movie_paths + movie_paths_fov
        else:
            movie_paths = movie_paths_fov

    return movie_paths
