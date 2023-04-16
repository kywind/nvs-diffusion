import os

# os.makedirs('vis/re10k_vid/', exist_ok=True)
pattern = f'data/nerfstudio/poster/images/*.png'
os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{pattern}' -c:v libx264 -pix_fmt yuv420p 'vis/poster.mp4'")
