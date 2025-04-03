#!/usr/bin/env python3
"""
Chop That Image - A tool to cut images into tiles
"""
import os
import sys
import argparse
import random # <-- Import random
from pathlib import Path
from typing import Tuple, List, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ttkthemes import ThemedTk


class ImageCutter:
    """Core functionality for cutting images into tiles"""
    
    def __init__(self):
        self.image_path = None
        self.output_dir = None
        self.image = None
        self.preview_image = None
        self.crop_box = None  # Store crop region (left, top, right, bottom)
    
    def load_image(self, image_path: str) -> bool:
        """Load an image from the specified path"""
        try:
            self.image_path = image_path
            self.image = Image.open(image_path)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def cut_image(self, rows: int, cols: int, output_dir: str,
                  handle_uneven: str = 'scale', trim_borders: bool = False,
                  trim_threshold: int = 75) -> List[str]:
        """
        Cut the loaded image into a grid of tiles
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            output_dir: Directory to save the output tiles
            handle_uneven: How to handle images that don't divide evenly
                           ('scale', 'pad', 'crop')
            trim_borders: Whether to automatically trim empty space around each tile
            trim_threshold: Color difference threshold for trimming (0-255)
        
        Returns:
            List of paths to the saved tile images
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Keep a copy of the original image in case the object is reused (e.g., in GUI)
        # This prevents modifications like scaling/padding/cropping from persisting
        original_image_ref = self.image.copy()

        # Use a working copy for processing within this function call
        current_image_for_processing = self.image.copy()

        # --- (Initial image processing: uneven handling, etc.) ---
        # Get image dimensions before potential modification by handle_uneven
        img_width, img_height = current_image_for_processing.size
        tile_width_calc = img_width / cols
        tile_height_calc = img_height / rows

        # Check if the image divides evenly
        is_even_division = (img_width % cols == 0) and (img_height % rows == 0)

        # If not even division, handle according to strategy
        if not is_even_division:
            if handle_uneven == 'scale':
                # Scale the image to make it divisible
                new_width = cols * int(tile_width_calc)
                new_height = rows * int(tile_height_calc)
                current_image_for_processing = current_image_for_processing.resize((new_width, new_height), Image.LANCZOS)
            elif handle_uneven == 'pad':
                # Pad the image to make it divisible
                new_width = cols * int(tile_width_calc + 0.5)
                new_height = rows * int(tile_height_calc + 0.5)
                # Ensure new image uses mode compatible with original (handle L, P etc.)
                mode = current_image_for_processing.mode
                if mode == 'P': # Use RGBA for palette if padding
                     mode = 'RGBA'
                     bg_color_pad = (255, 255, 255, 0) # Transparent padding
                elif mode == 'L':
                     bg_color_pad = 255 # White padding for grayscale
                elif 'A' in mode:
                     bg_color_pad = (255, 255, 255, 0) # Transparent padding for RGBA/LA
                else:
                     bg_color_pad = (255, 255, 255) # White padding for RGB

                new_img = Image.new(mode, (new_width, new_height), bg_color_pad)

                # Convert original image if necessary before pasting
                paste_img = current_image_for_processing
                if paste_img.mode != new_img.mode:
                     # Try converting to the target mode
                     try:
                          paste_img = paste_img.convert(new_img.mode)
                     except ValueError:
                          # Fallback: Convert both to RGBA if direct conversion fails
                          paste_img = paste_img.convert('RGBA')
                          new_img = new_img.convert('RGBA')


                # Handle pasting transparency correctly
                if 'A' in paste_img.mode:
                    new_img.paste(paste_img, (0, 0), paste_img) # Use alpha channel as mask
                else:
                    new_img.paste(paste_img, (0, 0))

                current_image_for_processing = new_img
            elif handle_uneven == 'crop':
                # Apply optimal crop if no custom crop box is set
                # Crucially, apply crop to the *original* image reference stored in self.image
                # The crop_box is relative to self.image, not the potentially modified current_image_for_processing
                img_to_crop_from = self.image # Use the original image reference for cropping coordinates
                if not self.crop_box:
                    # Calculate based on original dimensions stored in self.image
                    crop_box = self.calculate_optimal_crop(rows, cols)
                else:
                    crop_box = self.crop_box # Use user-defined box relative to self.image

                current_image_for_processing = img_to_crop_from.crop(crop_box)
                # Don't reset self.crop_box here, let the GUI handle it if needed

        # Use the potentially processed image for tiling
        img_to_tile = current_image_for_processing
        img_width, img_height = img_to_tile.size
        # Recalculate tile dimensions based on the potentially modified image
        tile_width = img_width / cols
        tile_height = img_height / rows

        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        processed_tiles_data = [] # Store tuples of (processed_image, output_path, original_mode, detected_bg_color)

        for row in range(rows):
            for col in range(cols):
                left = int(col * tile_width)
                top = int(row * tile_height)
                # Use floor for right/bottom to prevent going over by fractional pixels
                right = int((col + 1) * tile_width)
                bottom = int((row + 1) * tile_height)

                # Clamp coordinates strictly within the image dimensions
                left = max(0, left)
                top = max(0, top)
                right = min(img_width, right)
                bottom = min(img_height, bottom)

                # Prevent zero-size crops if calculation is imperfect
                if left >= right or top >= bottom:
                    print(f"Warning: Skipping zero-size tile at row {row}, col {col}")
                    continue

                # Crop the tile from the (potentially processed) image
                try:
                    tile = img_to_tile.crop((left, top, right, bottom))
                except Exception as e:
                     print(f"Error cropping tile at row {row}, col {col}: {e}")
                     continue

                original_mode = tile.mode # Store mode before potential conversion in trim

                # --- Detect background color of this specific tile BEFORE trimming --- 
                detected_bg_color = None
                try:
                    # Simplified corner sampling for this tile
                    t_width, t_height = tile.size
                    if t_width >= 2 and t_height >= 2:
                        t_corners = [(0, 0), (t_width-1, 0), (0, t_height-1), (t_width-1, t_height-1)]
                        t_bg_colors = [tile.getpixel(p) for p in t_corners]
                        
                        from collections import Counter
                        def normalize_color_local(c):
                            if isinstance(c, int): return (c, c, c)
                            if len(c) == 4: return c[:3]
                            return c
                            
                        t_normalized_bg_colors = [normalize_color_local(c) for c in t_bg_colors]
                        t_color_counts = Counter(t_normalized_bg_colors)
                        detected_bg_color = t_color_counts.most_common(1)[0][0] # Store the RGB tuple
                    elif t_width > 0 and t_height > 0: # Fallback for tiny images: use top-left
                         detected_bg_color = normalize_color_local(tile.getpixel((0,0)))
                         
                except Exception as bg_e:
                    print(f"Warning: Could not detect background for tile r{row} c{col}: {bg_e}")
                    # Default fallback if detection fails (e.g., use white/black based on mode)
                    if original_mode == 'L': detected_bg_color = 255
                    else: detected_bg_color = (255, 255, 255)
                # --- End Background Detection ---

                output_filename = f"{base_name}_r{row}_c{col}.png"
                output_path = os.path.join(output_dir, output_filename)

                # Auto-trim borders if requested
                current_tile_to_process = tile
                if trim_borders:
                    try:
                        trimmed_tile = self.trim_image_borders(tile, threshold=trim_threshold)
                        if trimmed_tile.width > 0 and trimmed_tile.height > 0:
                            current_tile_to_process = trimmed_tile
                        else: # Handle cases where trimming results in an empty image
                             print(f"Warning: Tile at row {row}, col {col} trimmed to empty, using 1x1 placeholder.")
                             # Use a 1x1 transparent/white pixel based on original mode
                             if 'A' in original_mode or original_mode == 'P': # Check if alpha possible
                                  current_tile_to_process = Image.new('RGBA', (1, 1), (0,0,0,0))
                             elif original_mode == 'L':
                                  current_tile_to_process = Image.new('L', (1, 1), 255)
                             else:
                                  current_tile_to_process = Image.new('RGB', (1, 1), (255,255,255))
                    except Exception as e:
                         print(f"Error trimming tile at row {row}, col {col}: {e}")
                         # Fallback to using the untrimmed tile
                         current_tile_to_process = tile

                processed_tiles_data.append((current_tile_to_process, output_path, original_mode, detected_bg_color))

        # --- Post-processing: Padding to uniform size if trimming was enabled ---
        final_saved_paths = []
        if trim_borders and processed_tiles_data:
            max_width = 0
            max_height = 0
            for tile_img, _, _, _ in processed_tiles_data: # Adjusted tuple unpacking
                max_width = max(max_width, tile_img.width)
                max_height = max(max_height, tile_img.height)

            # Ensure max dimensions are at least 1x1
            max_width = max(1, max_width)
            max_height = max(1, max_height)

            for tile_img, output_path, mode_before_trim, bg_color_detected in processed_tiles_data: # Adjusted tuple unpacking
                # Determine padding mode and color based on original mode and detected bg
                if 'A' in mode_before_trim or mode_before_trim == 'P': # Prioritize transparency if possible
                    padding_color = (0, 0, 0, 0) # Transparent
                    final_mode = 'RGBA'
                elif mode_before_trim == 'L':
                    # Use detected grayscale bg, default white if detection failed
                    padding_color = bg_color_detected[0] if isinstance(bg_color_detected, tuple) else (bg_color_detected or 255)
                    final_mode = 'L'
                else: # Default to RGB or other non-alpha modes
                    # Use detected RGB bg, default white if detection failed
                    padding_color = bg_color_detected or (255, 255, 255)
                    final_mode = 'RGB' # Assume RGB if not L and no Alpha

                # Create padded image
                # Ensure the padding color matches the final_mode (e.g., int for L, tuple for RGB/RGBA)
                if final_mode == 'L' and isinstance(padding_color, tuple): padding_color = padding_color[0]
                elif final_mode == 'RGBA' and len(padding_color) == 3: padding_color += (0,) # Ensure alpha for transparent padding
                elif final_mode == 'RGB' and len(padding_color) == 4: padding_color = padding_color[:3] # Drop alpha if target is RGB
                elif final_mode == 'RGB' and isinstance(padding_color, int): padding_color = (padding_color,) * 3 # Convert L to RGB
                
                try:
                   padded_tile = Image.new(final_mode, (max_width, max_height), padding_color)
                except Exception as e_create:
                   print(f"Error creating padded tile {output_path} with mode {final_mode} and color {padding_color}: {e_create}")
                   # Fallback: create a simple RGBA transparent tile
                   final_mode = 'RGBA'
                   padding_color = (0,0,0,0)
                   padded_tile = Image.new(final_mode, (max_width, max_height), padding_color)

                # Calculate position to center the tile
                paste_x = (max_width - tile_img.width) // 2
                paste_y = (max_height - tile_img.height) // 2

                # Paste using alpha channel if possible
                try:
                    # Ensure target can handle alpha if source has it
                    if 'A' in tile_img.mode and padded_tile.mode != 'RGBA':
                        padded_tile = padded_tile.convert('RGBA')

                    # Ensure source mode matches target mode OR use alpha mask
                    if 'A' in tile_img.mode:
                        paste_tile = tile_img.convert('RGBA') # Ensure consistent RGBA for alpha paste
                        padded_tile.paste(paste_tile, (paste_x, paste_y), paste_tile.split()[-1]) # Use alpha channel as mask
                    elif tile_img.mode != padded_tile.mode:
                         paste_tile = tile_img.convert(padded_tile.mode)
                         padded_tile.paste(paste_tile, (paste_x, paste_y))
                    else:
                         padded_tile.paste(tile_img, (paste_x, paste_y)) # Modes match, simple paste

                    padded_tile.save(output_path)
                    final_saved_paths.append(output_path)
                except Exception as e:
                     print(f"Error processing or saving padded tile {output_path}: {e}")


        elif not trim_borders and processed_tiles_data: # Save directly if not trimming
            for tile_img, output_path, _, _ in processed_tiles_data: # Adjusted tuple unpacking
                 try:
                    tile_img.save(output_path)
                    final_saved_paths.append(output_path)
                 except Exception as e:
                     print(f"Error saving tile {output_path}: {e}")

        # Restore self.image to the state before this function call (important for GUI)
        self.image = original_image_ref

        return final_saved_paths
    
    def generate_preview(self, rows: int, cols: int, show_crop_region=False) -> Optional[Image.Image]:
        """Generate a preview image with grid lines showing the cuts"""
        if self.image is None:
            return None
        
        # Create a copy of the image
        preview = self.image.copy()
        
        # Always convert to RGBA for drawing
        preview_rgba = preview.convert('RGBA')
        
        # Create a transparent overlay for drawing
        draw_img = Image.new('RGBA', preview.size, (0, 0, 0, 0))
        
        # Get image dimensions
        img_width, img_height = preview.size
        
        # Import here to avoid circular import
        from PIL import ImageDraw
        
        # Create drawing object
        draw = ImageDraw.Draw(draw_img)
        
        # Draw crop region if available and requested
        if show_crop_region and self.crop_box:
            left, top, right, bottom = self.crop_box
            # Add semi-transparent overlay to the non-selected area
            # Top region
            if top > 0:
                draw.rectangle([(0, 0), (img_width, top)], fill=(0, 0, 0, 80))
            # Bottom region
            if bottom < img_height:
                draw.rectangle([(0, bottom), (img_width, img_height)], fill=(0, 0, 0, 80))
            # Left region
            if left > 0:
                draw.rectangle([(0, top), (left, bottom)], fill=(0, 0, 0, 80))
            # Right region
            if right < img_width:
                draw.rectangle([(right, top), (img_width, bottom)], fill=(0, 0, 0, 80))
            
            # Draw border around selected area
            draw.rectangle([left, top, right, bottom], outline=(0, 180, 0), width=3)
        
        # Calculate tile dimensions based on crop box or full image
        if show_crop_region and self.crop_box:
            left, top, right, bottom = self.crop_box
            grid_width = right - left
            grid_height = bottom - top
            tile_width = grid_width / cols
            tile_height = grid_height / rows
            
            # Draw horizontal lines within crop box
            for row in range(1, rows):
                y = int(top + row * tile_height)
                draw.line([(left, y), (right, y)], fill=(0, 128, 0, 128), width=2)
            
            # Draw vertical lines within crop box
            for col in range(1, cols):
                x = int(left + col * tile_width)
                draw.line([(x, top), (x, bottom)], fill=(0, 128, 0, 128), width=2)
        else:
            # Calculate tile dimensions for the full image
            tile_width = img_width / cols
            tile_height = img_height / rows
            
            # Draw horizontal lines
            for row in range(1, rows):
                y = int(row * tile_height)
                draw.line([(0, y), (img_width, y)], fill=(0, 128, 0, 128), width=2)
            
            # Draw vertical lines
            for col in range(1, cols):
                x = int(col * tile_width)
                draw.line([(x, 0), (x, img_height)], fill=(0, 128, 0, 128), width=2)
        
        # Composite the images - make sure they're both RGBA
        preview = Image.alpha_composite(preview_rgba, draw_img)
        self.preview_image = preview
        return preview

    def calculate_optimal_crop(self, rows: int, cols: int) -> Tuple[int, int, int, int]:
        """Calculate optimal crop dimensions to make the image evenly divisible"""
        if self.image is None:
            return (0, 0, 0, 0)
        
        img_width, img_height = self.image.size
        
        # Calculate optimal dimensions
        optimal_width = cols * (img_width // cols)
        optimal_height = rows * (img_height // rows)
        
        # Calculate crop box centered on the image
        left = (img_width - optimal_width) // 2
        top = (img_height - optimal_height) // 2
        right = left + optimal_width
        bottom = top + optimal_height
        
        return (left, top, right, bottom)

    def apply_crop(self, crop_box: Tuple[int, int, int, int]):
        """Apply the crop to the image"""
        if self.image is None:
            return
        
        self.image = self.image.crop(crop_box)
        self.crop_box = None  # Reset crop box after applying

    def trim_image_borders(self, img: Image.Image, threshold: int = 10) -> Image.Image:
        """More robustly trim borders based on background color similarity."""
        
        if img.mode == 'P': # Convert palette images to RGB(A)
            img = img.convert("RGBA" if 'transparency' in img.info else "RGB")
            
        # Determine background color (sample corners)
        width, height = img.size
        if width < 2 or height < 2:
            return img # Cannot sample corners

        corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
        bg_colors = [img.getpixel(p) for p in corners]
        
        # Find the most frequent corner color
        from collections import Counter
        
        # Normalize colors (ignore alpha for background detection)
        def normalize_color(c):
            if isinstance(c, int): return (c, c, c)
            if len(c) == 4: return c[:3]
            return c
            
        normalized_bg_colors = [normalize_color(c) for c in bg_colors]
        color_counts = Counter(normalized_bg_colors)
        bg = color_counts.most_common(1)[0][0]

        def color_diff_sq(c1, c2):
            c1_norm = normalize_color(c1)
            # c2 is already normalized bg
            return sum((a - b) ** 2 for a, b in zip(c1_norm, c2))

        # Find top border
        top = 0
        for y in range(height):
            is_bg_row = True
            for x in range(width):
                if color_diff_sq(img.getpixel((x, y)), bg) > threshold**2:
                    is_bg_row = False
                    break
            if not is_bg_row:
                top = y
                break
        else: # Image is entirely background
             # Use appropriate background color based on mode
             if 'A' in img.mode: bg_fill = bg + (0,) # Transparent
             elif img.mode == 'L': bg_fill = 255 # White
             else: bg_fill = bg # RGB
             return Image.new(img.mode, (1, 1), bg_fill)

        # Find bottom border
        bottom = height
        for y in range(height - 1, top - 1, -1):
            is_bg_row = True
            for x in range(width):
                if color_diff_sq(img.getpixel((x, y)), bg) > threshold**2:
                    is_bg_row = False
                    break
            if not is_bg_row:
                bottom = y + 1
                break

        # Find left border
        left = 0
        for x in range(width):
            is_bg_col = True
            for y in range(top, bottom):
                 if color_diff_sq(img.getpixel((x, y)), bg) > threshold**2:
                    is_bg_col = False
                    break
            if not is_bg_col:
                left = x
                break
        
        # Find right border
        right = width
        for x in range(width - 1, left - 1, -1):
            is_bg_col = True
            for y in range(top, bottom):
                if color_diff_sq(img.getpixel((x, y)), bg) > threshold**2:
                    is_bg_col = False
                    break
            if not is_bg_col:
                right = x + 1
                break
                
        if left >= right or top >= bottom:
             # Should not happen if the top loop didn't exit early
             # but as a fallback, return a 1x1 pixel
             if 'A' in img.mode: bg_fill = bg + (0,) # Transparent
             elif img.mode == 'L': bg_fill = 255 # White
             else: bg_fill = bg # RGB
             return Image.new(img.mode, (1, 1), bg_fill)

        return img.crop((left, top, right, bottom))

    def batch_process(self, image_paths: List[str], rows: int, cols: int, 
                      output_dir: str, handle_uneven: str = 'scale',
                      trim_borders: bool = False, trim_threshold: int = 75) -> dict:
        """Process multiple images in batch mode"""
        results = {}
        
        for img_path in image_paths:
            try:
                if self.load_image(img_path):
                    # Create a subdirectory for each image
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    img_output_dir = os.path.join(output_dir, base_name)
                    
                    # Cut the image
                    saved_tiles = self.cut_image(rows, cols, img_output_dir, 
                                                 handle_uneven, trim_borders, trim_threshold)
                    results[img_path] = {
                        'success': True,
                        'tiles': len(saved_tiles),
                        'output_dir': img_output_dir
                    }
                else:
                    results[img_path] = {
                        'success': False,
                        'error': 'Failed to load image'
                    }
            except Exception as e:
                results[img_path] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results


class ImageCutterGUI:
    """GUI interface for the Image Cutter application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Chop That Image")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Initialize image cutter
        self.image_cutter = ImageCutter()
        
        # Set forest green theme colors
        self.colors = {
            'primary': '#2A5C0B',     # Dark Forest Green
            'secondary': '#3E7C17',   # Medium Forest Green
            'accent': '#D6EFC7',      # Very Pale Green
            'text': '#FFFFFF',        # White
            'text_dark': '#1A3905',   # Very Dark Green
            'bg': '#E8F5E9',          # Pale Green (like a new leaf)
            'bg_dark': '#C8E6C9',     # Slightly darker pale green
            'brown': '#8B4513',       # Saddle Brown
            'brown_light': '#A0522D'  # Sienna
        }
        
        # Set background color for the main window - use explicit bg config
        self.root.configure(bg=self.colors['bg'])
        
        # Replace ttk styling with direct widget configuration
        # We'll use tk widgets instead of ttk where needed for better color control
        
        # Create main frame with explicit background color
        self.main_frame = tk.Frame(self.root, bg=self.colors['bg'], padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create variables for UI elements
        self.rows_var = tk.IntVar(value=3)
        self.cols_var = tk.IntVar(value=3)
        self.handle_uneven_var = tk.StringVar(value="crop")
        self.output_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.status_var = tk.StringVar(value="Ready")
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.trim_borders_var = tk.BooleanVar(value=True)  # Default to True
        self.trim_threshold_var = tk.IntVar(value=75) # Variable for trim threshold
        self.crop_mode_active = False
        self.crop_start_x = None
        self.crop_start_y = None
        
        # Create left frame for controls
        self.controls_frame = tk.Frame(self.main_frame, bg=self.colors['bg'], padx=10, pady=10)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create file selection area
        self.file_frame = tk.LabelFrame(self.controls_frame, text="Image Selection", 
                                      bg=self.colors['bg'], fg=self.colors['text_dark'],
                                      font=('Arial', 10, 'bold'), padx=10, pady=10)
        self.file_frame.pack(fill=tk.X, pady=5)
        
        self.browse_btn = tk.Button(self.file_frame, text="Browse...", 
                                  bg=self.colors['primary'], fg=self.colors['text'],
                                  activebackground=self.colors['secondary'],
                                  activeforeground=self.colors['accent'],
                                  font=('Arial', 10, 'bold'), relief=tk.RAISED,
                                  command=self.browse_image)
        self.browse_btn.pack(fill=tk.X, pady=5)
        
        self.batch_check = tk.Checkbutton(self.file_frame, text="Batch Mode", 
                                        variable=self.batch_mode_var,
                                        bg=self.colors['bg'], fg=self.colors['text_dark'],
                                        activebackground=self.colors['bg_dark'],
                                        activeforeground=self.colors['text_dark'],
                                        selectcolor=self.colors['bg'])
        self.batch_check.pack(anchor=tk.W, pady=5)
        
        self.file_path_label = tk.Label(self.file_frame, text="No image selected", 
                                     bg=self.colors['bg'], fg=self.colors['text_dark'],
                                     wraplength=200)
        self.file_path_label.pack(fill=tk.X, pady=5)
        
        # Create grid configuration area
        self.grid_frame = tk.LabelFrame(self.controls_frame, text="Grid Configuration", 
                                      bg=self.colors['bg'], fg=self.colors['text_dark'],
                                      font=('Arial', 10, 'bold'), padx=10, pady=10)
        self.grid_frame.pack(fill=tk.X, pady=5)
        
        # Rows control with value label
        rows_frame = tk.Frame(self.grid_frame, bg=self.colors['bg'])
        rows_frame.pack(fill=tk.X, pady=5)
        tk.Label(rows_frame, text="Rows:", bg=self.colors['bg'], 
               fg=self.colors['text_dark']).pack(side=tk.LEFT)
        self.rows_label = tk.Label(rows_frame, text=str(self.rows_var.get()), width=2,
                                bg=self.colors['bg'], fg=self.colors['text_dark'])
        self.rows_label.pack(side=tk.RIGHT)
        
        self.rows_scale = tk.Scale(self.grid_frame, from_=2, to=10, orient=tk.HORIZONTAL,
                                variable=self.rows_var, command=self.on_scale_changed,
                                bg=self.colors['bg'], fg=self.colors['text_dark'],
                                activebackground=self.colors['bg_dark'],
                                troughcolor=self.colors['bg_dark'], 
                                highlightbackground=self.colors['bg'])
        self.rows_scale.pack(fill=tk.X)
        
        # Columns control with value label
        cols_frame = tk.Frame(self.grid_frame, bg=self.colors['bg'])
        cols_frame.pack(fill=tk.X, pady=5)
        tk.Label(cols_frame, text="Columns:", bg=self.colors['bg'], 
               fg=self.colors['text_dark']).pack(side=tk.LEFT)
        self.cols_label = tk.Label(cols_frame, text=str(self.cols_var.get()), width=2,
                                bg=self.colors['bg'], fg=self.colors['text_dark'])
        self.cols_label.pack(side=tk.RIGHT)
        
        self.cols_scale = tk.Scale(self.grid_frame, from_=2, to=10, orient=tk.HORIZONTAL,
                                variable=self.cols_var, command=self.on_scale_changed,
                                bg=self.colors['bg'], fg=self.colors['text_dark'],
                                activebackground=self.colors['bg_dark'],
                                troughcolor=self.colors['bg_dark'],
                                highlightbackground=self.colors['bg'])
        self.cols_scale.pack(fill=tk.X)
        
        # Create options area
        self.options_frame = tk.LabelFrame(self.controls_frame, text="Options", 
                                        bg=self.colors['bg'], fg=self.colors['text_dark'],
                                        font=('Arial', 10, 'bold'), padx=10, pady=10)
        self.options_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(self.options_frame, text="Handle uneven divisions:", 
               bg=self.colors['bg'], fg=self.colors['text_dark']).pack(anchor=tk.W)
        
        # Create a custom-styled combobox with actual forest colors
        self.handle_uneven_combo = ttk.Combobox(self.options_frame, 
                                              textvariable=self.handle_uneven_var,
                                              values=["scale", "pad", "crop"],
                                              state="readonly")
        self.handle_uneven_combo.pack(fill=tk.X, pady=5)
        self.handle_uneven_combo.bind("<<ComboboxSelected>>", self.on_handling_method_changed)
        # Call once initially to set the correct state for crop controls - MOVED FROM HERE
        # self.on_handling_method_changed()
        
        # Add crop controls subframe (hidden initially)
        self.crop_controls_frame = tk.Frame(self.options_frame, bg=self.colors['bg'])
        
        # Add Select Region button
        self.select_crop_btn = tk.Button(self.crop_controls_frame, text="Select Crop Region", 
                                      bg=self.colors['primary'], fg=self.colors['text'],
                                      activebackground=self.colors['secondary'],
                                      activeforeground=self.colors['accent'],
                                      font=('Arial', 10, 'bold'), relief=tk.RAISED,
                                      command=self.start_crop_selection)
        self.select_crop_btn.pack(fill=tk.X, pady=5)
        
        # Add Reset Crop button
        self.reset_crop_btn = tk.Button(self.crop_controls_frame, text="Reset Crop", 
                                     bg=self.colors['brown'], fg=self.colors['text'],
                                     activebackground=self.colors['brown_light'],
                                     activeforeground=self.colors['accent'],
                                     font=('Arial', 10, 'bold'), relief=tk.RAISED,
                                     command=self.reset_crop)
        self.reset_crop_btn.pack(fill=tk.X, pady=5)
        
        # Call handle_uneven changed handler AFTER crop_controls_frame is defined
        self.on_handling_method_changed()
        
        # Add trim borders checkbox
        self.trim_borders_check = tk.Checkbutton(self.options_frame, text="Trim Tile Borders",
                                                variable=self.trim_borders_var,
                                                bg=self.colors['bg'], fg=self.colors['text_dark'],
                                                activebackground=self.colors['bg_dark'],
                                                activeforeground=self.colors['text_dark'],
                                                selectcolor=self.colors['bg'])
        self.trim_borders_check.pack(anchor=tk.W, pady=(10, 0)) # Adjust padding
        
        # Add threshold slider (initially hidden, shown when trim_borders is checked)
        self.threshold_frame = tk.Frame(self.options_frame, bg=self.colors['bg'])
        threshold_label = tk.Label(self.threshold_frame, text="Trim Threshold:",
                                   bg=self.colors['bg'], fg=self.colors['text_dark'])
        threshold_label.pack(side=tk.LEFT, padx=(15, 5)) # Indent under checkbox
        self.threshold_value_label = tk.Label(self.threshold_frame, 
                                              textvariable=self.trim_threshold_var, width=3,
                                              bg=self.colors['bg'], fg=self.colors['text_dark'])
        self.threshold_value_label.pack(side=tk.RIGHT)
        self.threshold_scale = tk.Scale(self.threshold_frame, from_=0, to=100, 
                                       orient=tk.HORIZONTAL,
                                       variable=self.trim_threshold_var,
                                       showvalue=0, # Hide default value display
                                       bg=self.colors['bg'], fg=self.colors['text_dark'],
                                       activebackground=self.colors['bg_dark'],
                                       troughcolor=self.colors['bg_dark'], 
                                       highlightbackground=self.colors['bg'])
        self.threshold_scale.pack(fill=tk.X, expand=True)
        
        # Link checkbox to show/hide threshold slider
        self.trim_borders_var.trace_add("write", self.toggle_threshold_slider)
        # Call once initially to set the correct hidden state
        self.toggle_threshold_slider()
        
        # Output directory controls
        tk.Label(self.options_frame, text="Output Directory:", 
               bg=self.colors['bg'], fg=self.colors['text_dark']).pack(anchor=tk.W)
        self.output_dir_frame = tk.Frame(self.options_frame, bg=self.colors['bg'])
        self.output_dir_frame.pack(fill=tk.X, pady=5)
        
        self.output_dir_entry = tk.Entry(self.output_dir_frame, textvariable=self.output_dir_var,
                                      bg=self.colors['accent'], fg=self.colors['text_dark'])
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.browse_dir_btn = tk.Button(self.output_dir_frame, text="...", width=3,
                                     bg=self.colors['brown'], fg=self.colors['text'],
                                     activebackground=self.colors['brown_light'],
                                     activeforeground=self.colors['accent'],
                                     command=self.browse_output_dir)
        self.browse_dir_btn.pack(side=tk.RIGHT)
        
        # Create action buttons
        self.action_frame = tk.Frame(self.controls_frame, bg=self.colors['bg'])
        self.action_frame.pack(fill=tk.X, pady=10)
        
        # Use a more prominent button for the main action
        self.cut_btn = tk.Button(self.action_frame, text="Slice Image",
                               bg=self.colors['primary'], fg=self.colors['text'],
                               activebackground=self.colors['secondary'],
                               activeforeground=self.colors['accent'],
                               font=('Arial', 12, 'bold'), relief=tk.RAISED,
                               height=2, command=self.cut_image)
        self.cut_btn.pack(fill=tk.X, pady=10)
        
        # Create right frame for image preview
        self.preview_frame = tk.LabelFrame(self.main_frame, text="Preview", 
                                        bg=self.colors['bg'], fg=self.colors['text_dark'],
                                        font=('Arial', 10, 'bold'), padx=10, pady=10)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Create canvas for image preview with forest themed background
        self.canvas = tk.Canvas(self.preview_frame, bg=self.colors['bg_dark'], 
                              highlightbackground=self.colors['primary'], highlightthickness=2)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create status bar with forest theme
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W,
                                bg=self.colors['primary'], fg=self.colors['text'])
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize preview image
        self.tk_preview = None
        self.gaxe_sprite_photo = None # Reference for the sprite PhotoImage

        # --- Add GAXE Sprite Easter Egg ---
        self._place_gaxe_sprite()
    
    def _place_gaxe_sprite(self):
        """Loads and places the GAXE sprite below the action frame."""
        sprite_path = "GAXE.png"
        sprite_size = 48 # Define sprite size
        padding = 10 # Define padding from edges/widgets

        try:
            # Load and resize sprite
            sprite_img = Image.open(sprite_path)
            sprite_img = sprite_img.resize((sprite_size, sprite_size), Image.LANCZOS)
            self.gaxe_sprite_photo_ref = ImageTk.PhotoImage(sprite_img)

            # Delay placement until after mainloop starts and sizes are stable
            def place_sprite_delayed():
                try:
                    self.root.update_idletasks() # Ensure sizes are updated
                    # Get position and height of the action frame
                    action_y = self.action_frame.winfo_y()
                    action_h = self.action_frame.winfo_height()
                    # Get width of controls frame to center sprite within it
                    controls_w = self.controls_frame.winfo_width()

                    # Calculate position below the action frame, centered horizontally within controls
                    target_x = (controls_w - sprite_size) // 2
                    target_y = action_y + action_h + padding 
                    
                    # Ensure coordinates are positive
                    target_x = max(padding, target_x)
                    target_y = max(padding, target_y)

                    # Check if calculated position is valid within main frame
                    main_w = self.main_frame.winfo_width()
                    main_h = self.main_frame.winfo_height()
                    if (target_x + sprite_size > main_w or target_y + sprite_size > main_h):
                         print("Warning: Calculated sprite position is outside main frame, using fallback.")
                         # Fallback: top-left corner within padding
                         target_x = padding
                         target_y = padding

                    sprite_label = tk.Label(self.main_frame, image=self.gaxe_sprite_photo_ref,
                                          bg=self.colors['bg'], bd=0)
                    sprite_label.image = self.gaxe_sprite_photo_ref
                    # Place using calculated coordinates relative to main_frame
                    sprite_label.place(x=target_x, y=target_y)

                except Exception as e_delayed:
                    # Catch errors that might happen if widgets aren't ready
                    print(f"Error placing sprite (delayed): {e_delayed}. Is window visible?")

            # Schedule the placement after a slightly longer delay
            self.root.after(200, place_sprite_delayed)

        except FileNotFoundError:
            print(f"Warning: Sprite image '{sprite_path}' not found. Cannot display easter egg.")
        except Exception as e:
            print(f"Error loading sprite: {e}")

    def browse_image(self):
        """Open file dialog to browse for an image"""
        if self.batch_mode_var.get():
            filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
            file_paths = filedialog.askopenfilenames(
                title="Select Images",
                filetypes=filetypes
            )
            if file_paths:
                self.file_path_label.config(text=f"{len(file_paths)} images selected")
                # Store the file paths
                self.image_paths = file_paths
                self.status_var.set(f"Ready to process {len(file_paths)} images")
        else:
            filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=filetypes
            )
            if file_path:
                self.file_path_label.config(text=os.path.basename(file_path))
                self.status_var.set("Loading image...")
                self.root.update()
                
                # Load the image
                if self.image_cutter.load_image(file_path):
                    self.status_var.set("Image loaded successfully")
                    self.update_preview()
                else:
                    messagebox.showerror("Error", "Failed to load the image")
                    self.status_var.set("Failed to load image")
    
    def browse_output_dir(self):
        """Open directory dialog to browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def on_handling_method_changed(self, event=None):
        """Show/hide crop controls based on handling method"""
        method = self.handle_uneven_var.get()
        if method == "crop":
            self.crop_controls_frame.pack(fill=tk.X, pady=5, after=self.handle_uneven_combo)
            # Calculate and set optimal crop initially
            if self.image_cutter.image is not None:
                self.image_cutter.crop_box = self.image_cutter.calculate_optimal_crop(
                    self.rows_var.get(), self.cols_var.get()
                )
                self.update_preview()
        else:
            self.crop_controls_frame.pack_forget()
            if self.image_cutter.image is not None:
                self.image_cutter.crop_box = None
                self.update_preview()
    
    def start_crop_selection(self):
        """Enter crop selection mode"""
        if self.image_cutter.image is None:
            return
        
        self.crop_mode_active = True
        self.status_var.set("Click and drag to select crop region")
        
        # Bind mouse events for crop selection
        self.canvas.bind("<ButtonPress-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)
    
    def on_crop_start(self, event):
        """Handle start of crop selection"""
        if not self.crop_mode_active:
            return
        
        # Convert canvas coordinates to image coordinates
        self.crop_start_x = event.x
        self.crop_start_y = event.y
    
    def on_crop_drag(self, event):
        """Handle mouse drag during crop selection"""
        if not self.crop_mode_active or self.crop_start_x is None:
            return
        
        # Calculate crop region in canvas coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get current image display size
        if hasattr(self, 'tk_preview') and self.tk_preview:
            # We need to convert from canvas to image coordinates
            img_width, img_height = self.image_cutter.image.size
            
            # Get size of displayed image
            display_width = self.tk_preview.width()
            display_height = self.tk_preview.height()
            
            # Calculate scaling factors
            scale_x = img_width / display_width
            scale_y = img_height / display_height
            
            # Calculate image offset in canvas
            offset_x = (canvas_width - display_width) // 2
            offset_y = (canvas_height - display_height) // 2
            
            # Convert canvas to image coordinates
            start_x = max(0, min(img_width, int((self.crop_start_x - offset_x) * scale_x)))
            start_y = max(0, min(img_height, int((self.crop_start_y - offset_y) * scale_y)))
            end_x = max(0, min(img_width, int((event.x - offset_x) * scale_x)))
            end_y = max(0, min(img_height, int((event.y - offset_y) * scale_y)))
            
            # Ensure start < end
            left = min(start_x, end_x)
            top = min(start_y, end_y)
            right = max(start_x, end_x)
            bottom = max(start_y, end_y)
            
            # Make the crop dimensions divisible by rows/cols
            rows = self.rows_var.get()
            cols = self.cols_var.get()
            
            # Adjust width and height to be divisible
            width = right - left
            height = bottom - top
            
            adjusted_width = (width // cols) * cols
            adjusted_height = (height // rows) * rows
            
            # If adjustment makes region too small, ensure at least one tile
            adjusted_width = max(cols, adjusted_width)
            adjusted_height = max(rows, adjusted_height)
            
            # Keep within image bounds
            if left + adjusted_width > img_width:
                adjusted_width = (img_width - left) // cols * cols
            if top + adjusted_height > img_height:
                adjusted_height = (img_height - top) // rows * rows
            
            # Update crop box
            self.image_cutter.crop_box = (left, top, left + adjusted_width, top + adjusted_height)
            
            # Update preview
            self.update_preview()
    
    def on_crop_end(self, event):
        """Handle end of crop selection"""
        if not self.crop_mode_active:
            return
        
        self.crop_mode_active = False
        self.status_var.set("Crop region selected")
        
        # Unbind mouse events
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
    
    def reset_crop(self):
        """Reset crop region to optimal default"""
        if self.image_cutter.image is None:
            return
        
        self.image_cutter.crop_box = self.image_cutter.calculate_optimal_crop(
            self.rows_var.get(), self.cols_var.get()
        )
        self.update_preview()
        self.status_var.set("Crop reset to optimal")
    
    def update_preview(self, *args):
        """Update the preview image with grid lines"""
        if self.image_cutter.image is not None and not self.batch_mode_var.get():
            rows = self.rows_var.get()
            cols = self.cols_var.get()
            
            show_crop = self.handle_uneven_var.get() == "crop"
            
            self.status_var.set(f"Generating preview for {rows}x{cols} grid...")
            self.root.update()
            
            preview = self.image_cutter.generate_preview(rows, cols, show_crop)
            
            if preview:
                # Resize the preview to fit the canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
                    # Calculate the aspect ratio
                    img_width, img_height = preview.size
                    img_ratio = img_width / img_height
                    canvas_ratio = canvas_width / canvas_height
                    
                    if img_ratio > canvas_ratio:
                        # Image is wider than canvas
                        display_width = canvas_width
                        display_height = int(canvas_width / img_ratio)
                    else:
                        # Image is taller than canvas
                        display_height = canvas_height
                        display_width = int(canvas_height * img_ratio)
                    
                    # Resize the preview image
                    display_preview = preview.resize((display_width, display_height), Image.LANCZOS)
                    
                    # Convert to PhotoImage
                    self.tk_preview = ImageTk.PhotoImage(display_preview)
                    
                    # Update canvas
                    self.canvas.delete("all")
                    self.canvas.create_image(
                        canvas_width // 2, canvas_height // 2,
                        image=self.tk_preview, anchor=tk.CENTER
                    )
                    
                    if show_crop:
                        self.status_var.set(f"Preview with crop region for {rows}x{cols} grid")
                    else:
                        self.status_var.set(f"Preview generated for {rows}x{cols} grid")
                else:
                    # Canvas not ready yet, schedule another update
                    self.root.after(100, self.update_preview)
    
    def on_scale_changed(self, *args):
        """Update the label values when scales change and update preview"""
        self.rows_label.config(text=str(self.rows_var.get()))
        self.cols_label.config(text=str(self.cols_var.get()))
        self.update_preview()
    
    def toggle_threshold_slider(self, *args):
        """Show or hide the threshold slider based on the checkbox state."""
        if self.trim_borders_var.get():
            self.threshold_frame.pack(fill=tk.X, after=self.trim_borders_check, pady=(0, 5))
        else:
            self.threshold_frame.pack_forget()
    
    def cut_image(self):
        """Cut the loaded image into tiles"""
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        output_dir = self.output_dir_var.get()
        handle_uneven = self.handle_uneven_var.get()
        trim_borders = self.trim_borders_var.get()
        trim_threshold = self.trim_threshold_var.get() # Get threshold value
        
        if self.batch_mode_var.get():
            if hasattr(self, 'image_paths') and self.image_paths:
                self.status_var.set(f"Processing {len(self.image_paths)} images...")
                self.root.update()
                
                try:
                    results = self.image_cutter.batch_process(
                        self.image_paths, rows, cols, output_dir, 
                        handle_uneven, trim_borders, trim_threshold # Pass threshold
                    )
                    
                    success_count = sum(1 for r in results.values() if r.get('success', False))
                    
                    if success_count > 0:
                        messagebox.showinfo(
                            "Batch Complete", 
                            f"Successfully processed {success_count} of {len(self.image_paths)} images.\n"
                            f"Output saved to: {self.output_dir_var.get()}"
                        )
                        self.status_var.set(f"Batch processing complete: {success_count}/{len(self.image_paths)} successful")
                    else:
                        messagebox.showerror("Error", "Failed to process any images in the batch")
                        self.status_var.set("Batch processing failed")
                
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred during batch processing: {str(e)}")
                    self.status_var.set("Batch processing error")
            else:
                messagebox.showwarning("Warning", "No images selected for batch processing")
                self.status_var.set("No images selected")
        
        else:  # Single image mode
            if self.image_cutter.image is not None:
                self.status_var.set(f"Cutting image into {rows}x{cols} grid...")
                self.root.update()
                
                try:
                    # Make sure output directory exists
                    os.makedirs(self.output_dir_var.get(), exist_ok=True)
                    
                    # Cut the image
                    saved_paths = self.image_cutter.cut_image(
                        rows, cols, output_dir, handle_uneven, 
                        trim_borders, trim_threshold # Pass threshold
                    )
                    
                    if saved_paths:
                        messagebox.showinfo(
                            "Success", 
                            f"Image cut into {len(saved_paths)} tiles.\n"
                            f"Output saved to: {self.output_dir_var.get()}"
                        )
                        self.status_var.set(f"Image cut into {len(saved_paths)} tiles")
                    else:
                        messagebox.showerror("Error", "Failed to cut the image")
                        self.status_var.set("Image cutting failed")
                
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred: {str(e)}")
                    self.status_var.set("Error cutting image")
            else:
                messagebox.showwarning("Warning", "No image loaded")
                self.status_var.set("No image loaded")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cut an image into a grid of tiles")
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows in the grid")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--handle-uneven", type=str, default="crop",
                        choices=["scale", "pad", "crop"],
                        help="How to handle images that don't divide evenly")
    parser.add_argument("--no-trim-borders", action="store_false", dest="trim_borders", 
                        help="Disable automatically trimming empty space around each tile (default is enabled)")
    parser.set_defaults(trim_borders=True)
    parser.add_argument("--trim-threshold", type=int, default=75,
                        help="Color difference threshold for trimming (0-255)")
    parser.add_argument("--batch", action="store_true", 
                        help="Process multiple images in batch mode")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI interface")
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Launch GUI if requested or if no arguments provided
    if args.gui or len(sys.argv) == 1:
        root = ThemedTk(theme="arc")
        root.title("Chop That Image")

        # --- Load and set window icon ---
        try:
            icon_path = "GAXE.png"
            icon_img = Image.open(icon_path)
            # No resize needed for icon usually, but keep reference
            root.gaxe_icon_photo = ImageTk.PhotoImage(icon_img) 
            root.iconphoto(False, root.gaxe_icon_photo)
            # Also try setting wm_iconphoto for better taskbar compatibility
            root.wm_iconphoto(False, root.gaxe_icon_photo) 
        except FileNotFoundError:
            print(f"Warning: Icon image '{icon_path}' not found.")
        except Exception as e:
            print(f"Error setting window icon: {e}")
        # --- End Icon Setting ---

        app = ImageCutterGUI(root)
        
        # Update preview when window is resized
        def on_resize(event):
            app.update_preview()
        
        app.canvas.bind("<Configure>", on_resize)
        
        # Size and position window to occupy right half of screen, full height
        root.update_idletasks() # Ensure widgets are initially placed
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Calculate dimensions and position
        target_width = screen_width // 2
        target_height = int((screen_height - 70) * 0.8) 
        x_pos = screen_width // 2
        y_pos = 0 # Start at top
        
        # Apply geometry
        root.geometry(f'{target_width}x{target_height}+{x_pos}+{y_pos}')
        # Optional: Set minimum size based on calculated target or initial req size
        root.minsize(max(800, root.winfo_reqwidth()), max(600, root.winfo_reqheight()))
        
        root.mainloop()
        return
    
    # CLI mode
    cutter = ImageCutter()
    
    # Pass the arguments directly from args
    trim_borders_cli = args.trim_borders
    trim_threshold_cli = args.trim_threshold
    handle_uneven_cli = args.handle_uneven
    
    if args.batch:
        # Batch mode requires image to be a directory or a list of files
        image_path = args.image
        if os.path.isdir(image_path):
            # Process all images in the directory
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                image_files.extend(Path(image_path).glob(f'*{ext}'))
            
            if not image_files:
                print(f"No image files found in {image_path}")
                return
            
            print(f"Processing {len(image_files)} images...")
            results = cutter.batch_process(
                [str(f) for f in image_files],
                args.rows, args.cols, args.output, handle_uneven_cli,
                trim_borders_cli, trim_threshold_cli # Use vars
            )
            
            # Print summary
            success_count = sum(1 for r in results.values() if r.get('success', False))
            print(f"Batch processing complete: {success_count}/{len(image_files)} successful")
            
        else:
            print("For batch mode, please provide a directory of images")
            return
    else:
        # Single image mode
        if args.image:
            if cutter.load_image(args.image):
                print(f"Cutting image into {args.rows}x{args.cols} grid...")
                
                # Cut the image
                saved_paths = cutter.cut_image(
                    args.rows, args.cols, args.output, handle_uneven_cli,
                    trim_borders_cli, trim_threshold_cli # Use vars
                )
                
                print(f"Image cut into {len(saved_paths)} tiles")
                print(f"Output saved to: {args.output}")
            else:
                print("Failed to load the image")
        else:
            print("Please provide an input image path with --image")


if __name__ == "__main__":
    main() 