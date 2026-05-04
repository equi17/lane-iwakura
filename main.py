import cv2
import numpy as np


# edge detection
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 70, 140)


# bird view
def perspective_transform(frame):
    height, width = frame.shape[:2]

    src = np.float32(
        [
            [width * 0.40, height * 0.63],
            [width * 0.60, height * 0.63],
            [width * 0.9, height * 0.9],
            [width * 0.1, height * 0.9],
        ]
    )

    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    return warped, matrix


def inverse_perspective(frame, matrix, shape):
    height, width = shape[:2]
    inv_matrix = np.linalg.inv(matrix)
    return cv2.warpPerspective(frame, inv_matrix, (width, height))


# only look at the road area - helps ignore shadows on the sides
def road_mask(binary):
    h, w = binary.shape[:2]
    mask = np.zeros_like(binary)

    poly = np.array(
        [
            [
                (int(w * 0.1), h),
                (int(w * 0.45), int(h * 0.6)),
                (int(w * 0.55), int(h * 0.6)),
                (int(w * 0.9), h),
            ]
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(binary, mask)


# finds left and right lane pixels
def get_lane_pixels(binary_warped):
    y, x = np.where(binary_warped > 0)
    mid = binary_warped.shape[1] // 2

    left = x < mid
    right = x >= mid

    return (x[left], y[left]), (x[right], y[right])


# curve fitting
def fit_poly(x, y):
    if len(x) < 80:
        return None
    return np.polyfit(y, x, 2)


# smooths out jitter
class LaneSmoother:
    def __init__(self, max_len=5):
        self.left = []
        self.right = []
        self.max_len = max_len

    def update(self, left, right):
        if left is not None:
            self.left.append(left)
            self.left = self.left[-self.max_len :]

        if right is not None:
            self.right.append(right)
            self.right = self.right[-self.max_len :]

        left_avg = np.mean(self.left, axis=0) if self.left else None
        right_avg = np.mean(self.right, axis=0) if self.right else None

        return left_avg, right_avg


# draws the lane lines
def draw_poly(frame, poly, color):
    if poly is None:
        return

    y_vals = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
    x_vals = poly[0] * y_vals**2 + poly[1] * y_vals + poly[2]

    pts = np.array([np.transpose(np.vstack([x_vals, y_vals]))], dtype=np.int32)
    cv2.polylines(frame, pts, False, color, 5)


# main loop
cap = cv2.VideoCapture("road.mp4")
smoother = LaneSmoother()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    edges = canny(frame)
    masked = road_mask(edges)
    warped, matrix = perspective_transform(masked)

    left_pts, right_pts = get_lane_pixels(warped)

    left_poly = fit_poly(*left_pts)
    right_poly = fit_poly(*right_pts)

    left_poly, right_poly = smoother.update(left_poly, right_poly)

    lane = np.zeros((h, w, 3), dtype=np.uint8)

    draw_poly(lane, left_poly, (0, 255, 0))
    draw_poly(lane, right_poly, (0, 255, 0))

    lane_back = inverse_perspective(lane, matrix, frame.shape)

    result = cv2.addWeighted(frame, 0.8, lane_back, 1, 1)

    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
