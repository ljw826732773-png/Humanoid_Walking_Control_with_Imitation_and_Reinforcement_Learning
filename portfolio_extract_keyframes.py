import argparse
import csv
import os

import cv2
import numpy as np


def choose_frame_indices(frame_count, num_frames):
    if frame_count <= 0:
        raise ValueError("Video has no readable frames.")
    if num_frames <= 1:
        return [frame_count // 2]
    start = int(frame_count * 0.05)
    end = int(frame_count * 0.95)
    return np.linspace(start, end, num_frames, dtype=int).tolist()


def put_label(image, text):
    labeled = image.copy()
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (labeled.shape[1], 34), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, labeled, 0.45, 0, labeled)
    cv2.putText(labeled, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def extract_keyframes(video_path, output_dir, num_frames, columns):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = choose_frame_indices(frame_count, num_frames)

    frames = []
    rows = []
    for order, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        time_sec = frame_index / fps
        label = f"Frame {frame_index} | {time_sec:.1f}s"
        frame_path = os.path.join(output_dir, f"bc_improved_keyframe_{order:02d}.png")
        labeled = put_label(frame, label)
        cv2.imwrite(frame_path, labeled)
        frames.append(labeled)
        rows.append(
            {
                "order": order,
                "frame_index": frame_index,
                "time_sec": f"{time_sec:.3f}",
                "path": frame_path,
            }
        )

    cap.release()
    if not frames:
        raise RuntimeError("No frames could be extracted.")

    thumb_w = 320
    thumb_h = int(height * (thumb_w / max(width, 1)))
    thumbs = [cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for frame in frames]
    columns = max(1, min(columns, len(thumbs)))
    rows_count = int(np.ceil(len(thumbs) / columns))
    pad = 10
    canvas_h = rows_count * thumb_h + (rows_count + 1) * pad
    canvas_w = columns * thumb_w + (columns + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        y = pad + row * (thumb_h + pad)
        x = pad + col * (thumb_w + pad)
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb

    sheet_path = os.path.join(output_dir, "bc_improved_keyframes.png")
    cv2.imwrite(sheet_path, canvas)

    csv_path = os.path.join(output_dir, "bc_improved_keyframes.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"video frames: {frame_count}, fps: {fps:.2f}")
    print(f"wrote {sheet_path}")
    print(f"wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="portfolio_retrain_bc_improved/bc_improved_demo.mp4")
    parser.add_argument("--output-dir", default="portfolio_retrain_bc_improved/keyframes")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    extract_keyframes(args.video, args.output_dir, args.num_frames, args.columns)


if __name__ == "__main__":
    main()
