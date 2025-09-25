#!/usr/bin/env python3
"""
Original Board Detection Algorithm
Based on the original wechat_jump_auto.py algorithm that handles rectangles well
"""

import math
from PIL import Image

def find_platform_boundaries(image, piece_x, piece_y):
    """
    Find platform center using the ORIGINAL algorithm from wechat_jump_auto.py
    This algorithm handles rectangular platforms correctly using geometric perspective
    """
    print(f"🎯 Using ORIGINAL algorithm for board detection...")
    print(f"   Piece at: ({piece_x:.1f}, {piece_y:.1f})")

    w, h = image.size
    im_pixel = image.load()

    board_x = 0
    board_y = 0
    piece_body_width = 80  # Width around piece to avoid

    # Limit board scanning area to avoid piece interference (original algorithm)
    if piece_x < w / 2:
        board_x_start = piece_x
        board_x_end = w
    else:
        board_x_start = 0
        board_x_end = piece_x

    print(f"   Scanning board area: x=({board_x_start:.0f}-{board_x_end:.0f})")

    # Scan for board edges using color difference detection (original algorithm)
    for i in range(int(h / 3), int(h * 2 / 3)):
        last_pixel = im_pixel[0, i]
        if board_x or board_y:
            break

        board_x_sum = 0
        board_x_c = 0

        for j in range(int(board_x_start), int(board_x_end)):
            pixel = im_pixel[j, i]

            # Skip area around piece (original algorithm)
            if abs(j - piece_x) < piece_body_width:
                continue

            # Check Y axis below 5 pixels to avoid interference (original algorithm)
            if i + 5 < h:
                ver_pixel = im_pixel[j, i + 5]

                # Color difference detection (original algorithm)
                if (abs(pixel[0] - last_pixel[0]) +
                    abs(pixel[1] - last_pixel[1]) +
                    abs(pixel[2] - last_pixel[2]) > 10 and
                    abs(ver_pixel[0] - last_pixel[0]) +
                    abs(ver_pixel[1] - last_pixel[1]) +
                    abs(ver_pixel[2] - last_pixel[2]) > 10):

                    board_x_sum += j
                    board_x_c += 1

        if board_x_sum:
            board_x = board_x_sum / board_x_c
            print(f"   Found board_x: {board_x:.1f} at y={i}")
            break

    if not board_x:
        print("❌ No board_x found")
        return None, None

    # CRITICAL: Use original geometric center calculation for board_y
    # This handles rectangular platforms correctly using perspective projection
    center_x = w / 2 + (24 / 1080) * w
    center_y = h / 2 + (17 / 1920) * h

    if piece_x > center_x:
        board_y = round((25.5 / 43.5) * (board_x - center_x) + center_y)
    else:
        board_y = round(-(25.5 / 43.5) * (board_x - center_x) + center_y)

    print(f"   Geometric center: ({center_x:.1f}, {center_y:.1f})")
    print(f"   Calculated board_y: {board_y} using perspective projection")
    print(f"✅ ORIGINAL algorithm result: ({board_x:.1f}, {board_y})")

    return float(board_x), float(board_y)


def main():
    """Test the original board detection"""
    try:
        image = Image.open('current_screenshot.png')
        width, height = image.size
        print(f"📱 Image size: {width} x {height}")

        # Test with current piece position
        piece_x, piece_y = 365.8, 1393.0

        center_x, center_y = find_platform_boundaries(image, piece_x, piece_y)

        if center_x is not None:
            print(f"🎯 ORIGINAL DETECTION SUCCESS!")
            print(f"   Board center: ({center_x:.1f}, {center_y:.1f})")

            # Create debug image
            import cv2
            import numpy as np

            img_array = np.array(image)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Draw piece (red)
            cv2.circle(img, (int(piece_x), int(piece_y)), 25, (0, 0, 255), -1)
            cv2.putText(img, 'PIECE', (int(piece_x-40), int(piece_y-40)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Draw detected center (green)
            cv2.circle(img, (int(center_x), int(center_y)), 25, (0, 255, 0), -1)
            cv2.circle(img, (int(center_x), int(center_y)), 35, (0, 255, 0), 3)
            cv2.line(img, (int(center_x-20), int(center_y)), (int(center_x+20), int(center_y)), (255, 255, 255), 3)
            cv2.line(img, (int(center_x), int(center_y-20)), (int(center_x), int(center_y+20)), (255, 255, 255), 3)
            cv2.putText(img, 'ORIGINAL TARGET', (int(center_x-80), int(center_y-50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw connection line
            cv2.line(img, (int(piece_x), int(piece_y)), (int(center_x), int(center_y)), (255, 255, 0), 3)

            cv2.imwrite('debug_original_detection.png', img)
            print(f"💾 Original debug image saved as: debug_original_detection.png")
        else:
            print("❌ Original detection failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()