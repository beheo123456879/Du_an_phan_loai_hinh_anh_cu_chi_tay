def square_box(x1, y1, x2, y2, img_w, img_h, margin=0.15):
    """Biến khung bao (pixel) thành khung vuông, mở rộng mỗi phía `margin` và cắt theo biên ảnh."""
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(w, h) * (1 + 2 * margin)
    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(img_w, cx + side / 2))
    ny2 = int(min(img_h, cy + side / 2))
    return nx1, ny1, nx2, ny2
