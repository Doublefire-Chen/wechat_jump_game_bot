#!/usr/bin/env python3
"""
Modern WeChat Jump Game Bot
Complete bot with manual and auto modes
"""

import math
import time
import threading
import sys
import select
import subprocess
from PIL import Image


class JumpBot:
    def __init__(self, debug_mode=False):
        self.piece_color_range = {
            'r_min': 50, 'r_max': 60,
            'g_min': 53, 'g_max': 63,
            'b_min': 95, 'b_max': 110
        }
        self.piece_base_height_half = 13
        self.piece_body_width = 80  # Increased for better exclusion
        self.stop_auto = False
        self.debug_mode = debug_mode

    def take_screenshot(self, filename='current_screenshot.png'):
        """Take screenshot using ADB"""
        try:
            # Take screenshot on device
            subprocess.run(['adb', 'shell', 'screencap', '-p', f'/sdcard/{filename}'],
                           check=True, capture_output=True)

            # Pull to local
            subprocess.run(['adb', 'pull', f'/sdcard/{filename}', f'./{filename}'],
                           check=True, capture_output=True)

            print(f"✅ Screenshot saved as {filename}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to take screenshot: {e}")
            return False

    def find_piece_position(self, image):
        """Find the game piece position"""
        width, height = image.size
        pixels = image.load()

        coord_y_start_scan = height // 4
        border_x_scan = width // 8

        piece_x_sum = 0
        piece_x_counter = 0
        piece_y_max = 0

        for y in range(coord_y_start_scan, height * 2 // 3):
            for x in range(border_x_scan, width - border_x_scan):
                pixel = pixels[x, y]
                r, g, b = pixel[:3] if len(pixel) > 3 else pixel

                if (self.piece_color_range['r_min'] < r < self.piece_color_range['r_max'] and
                    self.piece_color_range['g_min'] < g < self.piece_color_range['g_max'] and
                        self.piece_color_range['b_min'] < b < self.piece_color_range['b_max']):
                    piece_x_sum += x
                    piece_x_counter += 1
                    piece_y_max = max(y, piece_y_max)

        if piece_x_counter == 0:
            return None, None

        piece_x = piece_x_sum / piece_x_counter
        piece_y = piece_y_max - self.piece_base_height_half

        return piece_x, piece_y

    def find_platform_boundaries(self, image, piece_x, piece_y):
        """
        Find platform boundaries and calculate true centers
        UPDATED: Now uses improved circular platform detection algorithm
        """
        # Import the original algorithm from the repo
        try:
            from original_board_detection import find_platform_boundaries as original_detection
            result = original_detection(image, piece_x, piece_y)
            print(f"✅ Using ORIGINAL algorithm: {result}")
            return result
        except ImportError as e:
            print(f"⚠️  Original algorithm not available: {e}")
            print("   Using fallback method")
        except Exception as e:
            print(f"❌ Original algorithm failed: {e}")
            print("   Falling back to original method")

        # Fallback to original method
        width, height = image.size
        pixels = image.load()
        piece_body_width = 80

        if self.debug_mode:
            print(f"🔍 Finding platform boundaries for center calculation...")

        # Search in upper area where target platforms are likely
        upper_zone_start = height // 6
        upper_zone_end = int(piece_y - 100)

        platform_groups = {}
        y_step = 5

        for y in range(upper_zone_start, upper_zone_end, y_step):
            # Find all continuous platform segments at this y level
            platform_segments = []
            current_segment_start = None
            current_segment_end = None

            last_pixel = pixels[0, y]
            background_color = last_pixel  # Assume first pixel is background

            for x in range(1, width):
                # Skip piece area
                if abs(x - piece_x) < piece_body_width:
                    if current_segment_start is not None:
                        # End current segment before piece area
                        current_segment_end = x - 1
                        if current_segment_end - current_segment_start > 30:  # Minimum platform width
                            platform_segments.append(
                                (current_segment_start, current_segment_end))
                        current_segment_start = None
                    continue

                pixel = pixels[x, y]

                # Calculate color difference from background
                bg_diff = (abs(pixel[0] - background_color[0]) +
                           abs(pixel[1] - background_color[1]) +
                           abs(pixel[2] - background_color[2]))

                # If this pixel is significantly different from background (part of platform)
                if bg_diff > 20:
                    if current_segment_start is None:
                        current_segment_start = x
                    current_segment_end = x
                else:
                    # Back to background, end current segment
                    if current_segment_start is not None:
                        if current_segment_end - current_segment_start > 30:
                            platform_segments.append(
                                (current_segment_start, current_segment_end))
                        current_segment_start = None

            # Add final segment if exists
            if current_segment_start is not None and current_segment_end is not None:
                if current_segment_end - current_segment_start > 30:
                    platform_segments.append(
                        (current_segment_start, current_segment_end))

            # Group segments by their center position
            for start_x, end_x in platform_segments:
                center_x = (start_x + end_x) / 2
                width_px = end_x - start_x

                # Find which platform group this belongs to
                found_group = False
                for group_id, group_data in platform_groups.items():
                    if abs(group_data['center_x'] - center_x) < 50:  # Same platform
                        # Update group with new data
                        platform_groups[group_id]['segments'].append({
                            'y': y, 'start_x': start_x, 'end_x': end_x,
                            'center_x': center_x, 'width': width_px
                        })
                        found_group = True
                        break

                if not found_group:
                    # Create new group
                    group_id = len(platform_groups)
                    platform_groups[group_id] = {
                        'center_x': center_x,
                        'segments': [{
                            'y': y, 'start_x': start_x, 'end_x': end_x,
                            'center_x': center_x, 'width': width_px
                        }]
                    }

        # Calculate true centers for each platform
        platforms = []
        for group_id, group_data in platform_groups.items():
            segments = group_data['segments']

            if len(segments) < 3:  # Need multiple segments to be a valid platform
                continue

            # Calculate weighted center based on all segments
            total_weighted_x = 0
            total_weight = 0
            min_y = float('inf')
            max_y = 0

            for segment in segments:
                weight = segment['width']  # Width as weight
                total_weighted_x += segment['center_x'] * weight
                total_weight += weight
                min_y = min(min_y, segment['y'])
                max_y = max(max_y, segment['y'])

            if total_weight == 0:
                continue

            true_center_x = total_weighted_x / total_weight
            # Closer to top of platform
            center_y = min_y + (max_y - min_y) * 0.3

            distance_from_piece = math.sqrt(
                (true_center_x - piece_x)**2 + (center_y - piece_y)**2)

            # Filter out platforms too close to piece
            if distance_from_piece < 120:
                continue

            platforms.append({
                'center_x': true_center_x,
                'center_y': center_y,
                'min_y': min_y,
                'max_y': max_y,
                'segment_count': len(segments),
                'avg_width': total_weight / len(segments),
                'distance': distance_from_piece
            })

            if self.debug_mode:
                print(f"Platform {group_id}: Center=({true_center_x:.1f}, {center_y:.1f}), "
                      f"Segments={len(segments)}, AvgWidth={total_weight/len(segments):.1f}, "
                      f"Distance={distance_from_piece:.1f}")

        if not platforms:
            if self.debug_mode:
                print("❌ No valid platforms found")
            return None, None

        # Score platforms
        for platform in platforms:
            score = 0

            # Prefer platforms with more segments (more solid detection)
            score += platform['segment_count'] * 15

            # Prefer wider platforms
            score += min(platform['avg_width'] / 10, 25)

            # Prefer platforms at reasonable distance
            if 150 < platform['distance'] < 600:
                score += 40

            # Prefer platforms above the piece
            if platform['center_y'] < piece_y - 50:
                score += 30

            platform['score'] = score

        # Sort by score
        platforms.sort(key=lambda x: x['score'], reverse=True)

        if self.debug_mode:
            print(f"\n🏆 Top platform candidates:")
            for i, platform in enumerate(platforms[:3]):
                print(f"  {i+1}. Score={platform['score']:.1f}, Center=({platform['center_x']:.1f}, {platform['center_y']:.1f}), "
                      f"Segments={platform['segment_count']}, Distance={platform['distance']:.1f}")

        best_platform = platforms[0]
        if self.debug_mode:
            print(
                f"✅ Selected TRUE center: ({best_platform['center_x']:.1f}, {best_platform['center_y']:.1f})")

        return best_platform['center_x'], best_platform['center_y']

    def calculate_press_time(self, distance):
        """Calculate press time based on distance using user's calibration"""
        # User specified: 565.5 pixels = 684 ms
        # So time_coefficient = 684 / 565.5 = 1.2097
        time_coefficient = 1.252
        press_time = int(distance * time_coefficient)

        # Ensure minimum press time
        press_time = max(press_time, 320)

        return press_time

    def execute_jump(self, press_time):
        """Execute jump using ADB swipe command"""
        cmd = ['adb', 'shell', 'input', 'swipe', '500',
               '1500', '500', '500', str(press_time)]
        cmd_str = ' '.join(cmd)

        try:
            print(f"🎯 Executing: {cmd_str}")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Jump executed with {press_time}ms duration")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to execute jump: {e}")
            return False

    def save_debug_image(self, image_path, piece_x, piece_y, board_x, board_y):
        """Save debug image without showing visualization (always executed)"""
        try:
            import cv2
            import numpy as np

            # Load and prepare image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if piece_x and piece_y:
                # Draw piece (red)
                cv2.circle(img_rgb, (int(piece_x), int(piece_y)),
                           20, (255, 0, 0), -1)
                cv2.circle(img_rgb, (int(piece_x), int(piece_y)),
                           30, (255, 0, 0), 3)
                cv2.putText(img_rgb, 'PIECE', (int(piece_x-40), int(piece_y-40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if board_x and board_y:
                # Draw target center (bright green)
                cv2.circle(img_rgb, (int(board_x), int(board_y)),
                           20, (0, 255, 0), -1)
                cv2.circle(img_rgb, (int(board_x), int(board_y)),
                           30, (0, 255, 0), 3)

                # White crosshair for precise center
                cv2.line(img_rgb, (int(board_x-15), int(board_y)),
                         (int(board_x+15), int(board_y)), (255, 255, 255), 2)
                cv2.line(img_rgb, (int(board_x), int(board_y-15)),
                         (int(board_x), int(board_y+15)), (255, 255, 255), 2)

                cv2.putText(img_rgb, 'TARGET', (int(board_x-35), int(board_y-40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Draw connection line
                if piece_x and piece_y:
                    cv2.line(img_rgb, (int(piece_x), int(piece_y)),
                             (int(board_x), int(board_y)), (255, 255, 0), 3)

            # Save debug image (overwrite policy)
            debug_filename = 'debug_detection.png'
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(debug_filename, img_bgr)
            print(f"💾 Debug image saved as: {debug_filename}")

        except ImportError:
            print("⚠️  Debug image save requires opencv-python")
        except Exception as e:
            print(f"⚠️  Could not save debug image: {e}")

    def show_debug_visualization(self, image_path, piece_x, piece_y, board_x, board_y):
        """Show debug visualization in debug mode"""
        if not self.debug_mode:
            return

        try:
            import cv2
            import matplotlib.pyplot as plt

            # Load and prepare image (reuse the same image processing)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if piece_x and piece_y:
                # Draw piece (red)
                cv2.circle(img_rgb, (int(piece_x), int(piece_y)),
                           20, (255, 0, 0), -1)
                cv2.circle(img_rgb, (int(piece_x), int(piece_y)),
                           30, (255, 0, 0), 3)
                cv2.putText(img_rgb, 'PIECE', (int(piece_x-40), int(piece_y-40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if board_x and board_y:
                # Draw target center (bright green)
                cv2.circle(img_rgb, (int(board_x), int(board_y)),
                           20, (0, 255, 0), -1)
                cv2.circle(img_rgb, (int(board_x), int(board_y)),
                           30, (0, 255, 0), 3)

                # White crosshair for precise center
                cv2.line(img_rgb, (int(board_x-15), int(board_y)),
                         (int(board_x+15), int(board_y)), (255, 255, 255), 2)
                cv2.line(img_rgb, (int(board_x), int(board_y-15)),
                         (int(board_x), int(board_y+15)), (255, 255, 255), 2)

                cv2.putText(img_rgb, 'TARGET', (int(board_x-35), int(board_y-40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Draw connection line
                if piece_x and piece_y:
                    cv2.line(img_rgb, (int(piece_x), int(piece_y)),
                             (int(board_x), int(board_y)), (255, 255, 0), 3)

            # Show visualization
            plt.figure(figsize=(8, 14))
            plt.imshow(img_rgb)
            plt.title(
                'DEBUG MODE - Detection Verification\n(Red=Piece, Green+White=Target Center)')
            plt.axis('off')

            if piece_x and board_x:
                distance = math.sqrt((board_x - piece_x) **
                                     2 + (board_y - piece_y)**2)
                press_time = self.calculate_press_time(distance)

                info_text = f'Distance: {distance:.1f}px\nPress Time: {press_time}ms\nAre the positions correct?'
                plt.figtext(0.02, 0.02, info_text, fontsize=11,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9))

            plt.tight_layout()
            plt.show()

            print(f"🖼️  Debug visualization shown - close the window to continue")

        except ImportError:
            print("⚠️  Debug visualization requires opencv-python and matplotlib")
        except Exception as e:
            print(f"⚠️  Debug visualization failed: {e}")

    def analyze_and_jump(self, step_num=0):
        """Analyze screenshot and execute jump"""
        print(f"\n=== Step {step_num} ===")

        # Take screenshot
        if not self.take_screenshot():
            return False

        # Load and analyze image
        try:
            image = Image.open('current_screenshot.png')
            print(f"📱 Image size: {image.size}")

            # Find piece
            piece_x, piece_y = self.find_piece_position(image)
            if piece_x is None:
                print("❌ Could not find game piece")
                return False

            print(f"🔵 Piece at: ({piece_x:.1f}, {piece_y:.1f})")

            # Find board
            board_x, board_y = self.find_platform_boundaries(
                image, piece_x, piece_y)
            if board_x is None:
                print("❌ Could not find target board")
                return False

            print(f"🎯 Target at: ({board_x:.1f}, {board_y:.1f})")

            # Calculate distance and timing
            distance = math.sqrt((board_x - piece_x) **
                                 2 + (board_y - piece_y) ** 2)
            press_time = self.calculate_press_time(distance)

            print(f"📏 Distance: {distance:.1f} pixels")
            print(f"⏱️  Press time: {press_time} ms")

            # Always save debug image (overwrite policy)
            self.save_debug_image('current_screenshot.png',
                                  piece_x, piece_y, board_x, board_y)

            # Show debug visualization if in debug mode
            if self.debug_mode:
                self.show_debug_visualization(
                    'current_screenshot.png', piece_x, piece_y, board_x, board_y)

            return press_time

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False

    def manual_mode(self):
        """Manual mode - requires user confirmation for each jump"""
        print("🎮 Starting MANUAL mode")
        print("Press Enter to take screenshot and calculate jump, or 'q' to quit")

        step = 1

        while True:
            try:
                user_input = input(
                    f"\n[Step {step}] Press Enter to continue (or 'q' to quit): ").strip().lower()

                if user_input == 'q':
                    print("👋 Manual mode stopped by user")
                    break

                if user_input == '':
                    press_time = self.analyze_and_jump(step)
                    if press_time:
                        # Ask for jump confirmation
                        confirm = input(
                            f"Execute jump with {press_time}ms? (Enter/y to confirm, 'n' to skip): ").strip().lower()

                        if confirm in ['', 'y', 'yes']:
                            if self.execute_jump(press_time):
                                step += 1
                                print(
                                    "✅ Jump completed! Wait for the piece to land before next step.")
                            else:
                                print("❌ Jump failed")
                        else:
                            print("⏭️  Jump skipped")
                    else:
                        print("❌ Could not analyze screenshot")
                else:
                    print("Invalid input. Press Enter to continue or 'q' to quit.")

            except KeyboardInterrupt:
                print("\n👋 Manual mode stopped")
                break

    def check_for_stop_input(self):
        """Check for keyboard input to stop auto mode"""
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line:
                self.stop_auto = True
                return True
        return False

    def auto_mode(self, max_steps=50):
        """Auto mode - runs automatically with 1s delays"""
        print("🤖 Starting AUTO mode")
        print("Press Enter at any time to stop auto mode")
        print(f"Max steps: {max_steps}")

        self.stop_auto = False
        step = 1

        while step <= max_steps and not self.stop_auto:
            try:
                print(f"\n⏳ Auto step {step}/{max_steps}")

                # Check for user input to stop
                if self.check_for_stop_input():
                    print("👋 Auto mode stopped by user input")
                    break

                press_time = self.analyze_and_jump(step)
                if press_time:
                    if self.execute_jump(press_time):
                        print(f"✅ Step {step} completed")
                        step += 1

                        # Wait 1 second before next step
                        print("⏱️  Waiting 1 second...")
                        time.sleep(1.0)
                    else:
                        print("❌ Jump failed, stopping auto mode")
                        break
                else:
                    print("❌ Analysis failed, stopping auto mode")
                    break

            except KeyboardInterrupt:
                print("\n👋 Auto mode stopped by Ctrl+C")
                break

        if step > max_steps:
            print(f"🏁 Auto mode completed {max_steps} steps")


def main():
    """Main function"""
    print("=" * 50)
    print("🎮 WeChat Jump Game Bot")
    print("=" * 50)

    # Check ADB connection
    try:
        result = subprocess.run(
            ['adb', 'devices'], capture_output=True, text=True, check=True)
        if 'device' not in result.stdout:
            print("❌ No Android device connected via ADB")
            print("Please connect your device and enable USB debugging")
            return
        print("✅ ADB device connected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ADB not found or failed to run")
        print("Please install ADB and add it to your PATH")
        return

    bot = JumpBot()

    while True:
        print("\n📋 Select mode:")
        print("1. Manual mode (with confirmations)")
        print("2. Auto mode (1s delays, press Enter to stop)")
        print("3. Debug mode (manual + visual verification)")
        print("4. Single test (analyze current screen)")
        print("5. Quit")

        try:
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == '1':
                bot.manual_mode()
            elif choice == '2':
                try:
                    max_steps = int(
                        input("Enter max steps (default 50): ") or "50")
                except ValueError:
                    max_steps = 50
                bot.auto_mode(max_steps)
            elif choice == '3':
                print(
                    "🐛 Starting DEBUG mode - you'll see visual verification for each detection")
                debug_bot = JumpBot(debug_mode=True)
                debug_bot.manual_mode()
            elif choice == '4':
                press_time = bot.analyze_and_jump(0)
                if press_time:
                    print(f"🎯 Recommended press time: {press_time}ms")
                    print(
                        f"🔧 Command would be: adb shell input swipe 500 1500 500 500 {press_time}")
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")

        except KeyboardInterrupt:
            print("\n👋 Program stopped")
            break


if __name__ == '__main__':
    main()
