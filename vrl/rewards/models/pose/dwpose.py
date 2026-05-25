"""DWPose ONNX whole-body pose model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)


class DWPoseModel:
    """Whole-body DWPose model returning OpenPose-ordered keypoints.

    We intentionally do not run pose on the full image when YOLOX finds no
    person: for reward training, a detector miss must stay distinguishable from
    a weak pose.
    """

    def __init__(
        self,
        *,
        model_repo: str,
        detector_file: str,
        pose_file: str,
        cache_dir: str | None,
        local_files_only: bool,
        device: str,
        detect_resolution: int,
    ) -> None:
        try:
            import cv2
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "DWPoseModel requires opencv-python, onnxruntime, and huggingface_hub",
            ) from exc

        self._cv2 = cv2
        self._detect_resolution = detect_resolution
        providers, provider_options = _onnx_providers(ort, device)
        detector_path = hf_hub_download(
            repo_id=model_repo,
            filename=detector_file,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        pose_path = hf_hub_download(
            repo_id=model_repo,
            filename=pose_file,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._det_session = ort.InferenceSession(
            detector_path,
            providers=providers,
            provider_options=provider_options,
        )
        self._pose_session = ort.InferenceSession(
            pose_path,
            providers=providers,
            provider_options=provider_options,
        )
        self._provider = _active_onnx_provider(self._pose_session)
        _warn_if_cuda_fallback(
            device=device,
            detector_session=self._det_session,
            pose_session=self._pose_session,
        )

    def predict_batch(self, images: list[Any]) -> list[Mapping[str, Any]]:
        """Predict whole-body pose for a batch of images."""

        return [self.predict(image) for image in images]

    def predict(self, image: Any) -> Mapping[str, Any]:
        """Predict whole-body pose for one image."""

        from PIL import Image

        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.array(image, dtype=np.uint8)
        height, width, _ = arr.shape
        scale = float(self._detect_resolution) / min(height, width)
        target_width = int(np.round(width * scale / 64.0)) * 64
        target_height = int(np.round(height * scale / 64.0)) * 64
        interpolation = self._cv2.INTER_LANCZOS4 if scale > 1 else self._cv2.INTER_AREA
        arr = self._cv2.resize(arr, (target_width, target_height), interpolation=interpolation)
        height, width = arr.shape[:2]
        boxes = _onnx_inference_detector(self._det_session, self._cv2, arr)
        if len(boxes) == 0:
            return {
                "keypoints": np.empty((0, 134, 2), dtype=float),
                "scores": np.empty((0, 134), dtype=float),
                "detector_boxes": boxes,
                "provider": self._provider,
            }
        keypoints, scores = _onnx_inference_pose(self._pose_session, self._cv2, boxes, arr)
        keypoints, scores = _onnx_openpose_order(keypoints, scores)
        keypoints = keypoints.astype(float)
        keypoints[..., 0] /= float(width)
        keypoints[..., 1] /= float(height)
        return {
            "keypoints": keypoints,
            "scores": scores,
            "detector_boxes": boxes,
            "provider": self._provider,
        }

    def __call__(self, image: Any) -> Mapping[str, Any]:
        return self.predict(image)


def _onnx_providers(
    ort: Any,
    device: str,
) -> tuple[list[str], list[dict[str, int] | dict[str, Any]] | None]:
    device = str(device)
    if device == "cpu":
        return ["CPUExecutionProvider"], None
    _preload_onnx_cuda_libraries(ort)
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"], None
    gpu_id = 0
    if ":" in device:
        try:
            gpu_id = int(device.rsplit(":", 1)[1])
        except ValueError:
            gpu_id = 0
    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": gpu_id}, {}]


def _preload_onnx_cuda_libraries(ort: Any) -> None:
    preload_dlls = getattr(ort, "preload_dlls", None)
    if not callable(preload_dlls):
        return
    try:
        preload_dlls(cuda=True, cudnn=True, msvc=False)
    except Exception as exc:
        _LOGGER.warning("Failed to preload ONNX Runtime CUDA libraries: %s", exc)


def _active_onnx_provider(session: Any) -> str:
    providers = list(session.get_providers())
    return providers[0] if providers else "unknown"


def _warn_if_cuda_fallback(*, device: str, detector_session: Any, pose_session: Any) -> None:
    if str(device) == "cpu":
        return
    detector_providers = list(detector_session.get_providers())
    pose_providers = list(pose_session.get_providers())
    if "CUDAExecutionProvider" in detector_providers and "CUDAExecutionProvider" in pose_providers:
        return
    _LOGGER.warning(
        "DWPose requested device=%s but ONNX Runtime fell back to CPU "
        "(detector_providers=%s, pose_providers=%s).",
        device,
        detector_providers,
        pose_providers,
    )


def _onnx_inference_detector(session: Any, cv2: Any, image: np.ndarray) -> np.ndarray:
    input_shape = (640, 640)
    padded = np.ones((input_shape[0], input_shape[1], 3), dtype=np.uint8) * 114
    ratio = min(input_shape[0] / image.shape[0], input_shape[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded[: int(image.shape[0] * ratio), : int(image.shape[1] * ratio)] = resized
    processed = np.ascontiguousarray(padded.transpose((2, 0, 1)), dtype=np.float32)
    ort_inputs = {session.get_inputs()[0].name: processed[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = _onnx_detector_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    detections = _onnx_multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if detections is None:
        return np.array([])
    final_boxes = detections[:, :4]
    final_scores = detections[:, 4]
    final_classes = detections[:, 5]
    keep = np.logical_and(final_scores > 0.3, final_classes == 0)
    return final_boxes[keep]


def _onnx_detector_postprocess(
    outputs: np.ndarray,
    image_size: tuple[int, int],
    *,
    p6: bool = False,
) -> np.ndarray:
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    heights = [image_size[0] // stride for stride in strides]
    widths = [image_size[1] // stride for stride in strides]
    for height, width, stride in zip(heights, widths, strides, strict=True):
        xv, yv = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grids_arr = np.concatenate(grids, 1)
    strides_arr = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids_arr) * strides_arr
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides_arr
    return outputs


def _onnx_multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    nms_thr: float,
    score_thr: float,
) -> np.ndarray | None:
    detections = []
    for class_idx in range(scores.shape[1]):
        class_scores = scores[:, class_idx]
        valid = class_scores > score_thr
        if valid.sum() == 0:
            continue
        valid_scores = class_scores[valid]
        valid_boxes = boxes[valid]
        keep = _onnx_nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            class_ids = np.ones((len(keep), 1)) * class_idx
            detections.append(
                np.concatenate([valid_boxes[keep], valid_scores[keep, None], class_ids], 1),
            )
    return np.concatenate(detections, 0) if detections else None


def _onnx_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[current] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def _onnx_inference_pose(
    session: Any,
    cv2: Any,
    boxes: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = session.get_inputs()[0].shape[2:]
    input_size = (width, height)
    images, centers, scales = _onnx_preprocess_pose(cv2, image, boxes, input_size)
    outputs = []
    output_names = [out.name for out in session.get_outputs()]
    input_name = session.get_inputs()[0].name
    for processed in images:
        outputs.append(session.run(output_names, {input_name: [processed.transpose(2, 0, 1)]}))
    return _onnx_postprocess_pose(outputs, input_size, centers, scales)


def _onnx_preprocess_pose(
    cv2: Any,
    image: np.ndarray,
    boxes: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    image_shape = image.shape[:2]
    if len(boxes) == 0:
        boxes = np.array([[0, 0, image_shape[1], image_shape[0]]])
    out_images, out_centers, out_scales = [], [], []
    for box in boxes:
        x1, y1, x2, y2 = np.hsplit(np.asarray(box)[None, :], [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2])[0] * 0.5
        scale = np.hstack([x2 - x1, y2 - y1])[0] * 1.25
        resized, scale = _onnx_top_down_affine(cv2, input_size, scale, center, image)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        out_images.append((resized - mean) / std)
        out_centers.append(center)
        out_scales.append(scale)
    return out_images, out_centers, out_scales


def _onnx_postprocess_pose(
    outputs: list[np.ndarray],
    model_input_size: tuple[int, int],
    centers: list[np.ndarray],
    scales: list[np.ndarray],
    *,
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    all_keypoints, all_scores = [], []
    for idx, output in enumerate(outputs):
        simcc_x, simcc_y = output
        keypoints, scores = _onnx_simcc_maximum(simcc_x, simcc_y)
        keypoints /= simcc_split_ratio
        keypoints = keypoints / model_input_size * scales[idx] + centers[idx] - scales[idx] / 2
        all_keypoints.append(keypoints[0])
        all_scores.append(scores[0])
    return np.asarray(all_keypoints), np.asarray(all_scores)


def _onnx_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, keypoints, width_x = simcc_x.shape
    _, _, width_y = simcc_y.shape
    x_scores = simcc_x.reshape(count * keypoints, width_x)
    y_scores = simcc_y.reshape(count * keypoints, width_y)
    x_locs = np.argmax(x_scores, axis=1)
    y_locs = np.argmax(y_scores, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    values = np.minimum(np.amax(x_scores, axis=1), np.amax(y_scores, axis=1))
    locs[values <= 0.0] = -1
    return locs.reshape(count, keypoints, 2), values.reshape(count, keypoints)


def _onnx_top_down_affine(
    cv2: Any,
    input_size: tuple[int, int],
    box_scale: np.ndarray,
    box_center: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = input_size
    aspect_ratio = width / height
    box_width, box_height = np.hsplit(box_scale[None, :], [1])
    box_scale = np.where(
        box_width > box_height * aspect_ratio,
        np.hstack([box_width, box_width / aspect_ratio]),
        np.hstack([box_height * aspect_ratio, box_height]),
    )[0]
    warp_matrix = _onnx_warp_matrix(box_center, box_scale, output_size=(width, height))
    return (
        cv2.warpAffine(image, warp_matrix, (int(width), int(height)), flags=cv2.INTER_LINEAR),
        box_scale,
    )


def _onnx_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    import cv2

    src_width = scale[0]
    dst_width, dst_height = output_size
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array([0.0, src_width * -0.5])
    direction = src[0, :] - src[1, :]
    src[2, :] = src[1, :] + np.r_[-direction[1], direction[0]]
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_width * 0.5, dst_height * 0.5]
    dst[1, :] = [dst_width * 0.5, dst_height * 0.5 - dst_width * 0.5]
    direction = dst[0, :] - dst[1, :]
    dst[2, :] = dst[1, :] + np.r_[-direction[1], direction[0]]
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _onnx_openpose_order(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keypoint_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(keypoint_info[:, [5, 6]], axis=1)
    neck[:, 2] = np.logical_and(keypoint_info[:, 5, 2] > 0.3, keypoint_info[:, 6, 2] > 0.3)
    reordered = np.insert(keypoint_info, 17, neck, axis=1)
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    reordered[:, openpose_idx] = reordered[:, mmpose_idx]
    return reordered[..., :2], reordered[..., 2]


__all__ = ["DWPoseModel", "_onnx_providers"]
