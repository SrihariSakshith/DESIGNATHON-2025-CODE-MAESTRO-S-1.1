import importlib

required_packages = [
    'gradio', 'torch', 'facenet_pytorch', 'numpy', 'PIL', 'cv2', 'pytorch_grad_cam',
    'warnings', 'os', 'glob', 'mediapipe', 'subprocess', 'streamlit', 'io',
    'transformers', 'librosa', 'moviepy.editor', 'pandas'
]

def check_packages(packages):
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Package {package} is not installed.")

check_packages(required_packages)
