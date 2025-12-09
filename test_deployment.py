"""
Test script to verify Streamlit app can load models before deployment
Run this locally: streamlit run streamlit_app.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("🔍 Checking deployment readiness...")
print("-" * 50)

# Check 1: Model files exist
model_dir = "ser/models"
required_models = [
    "enhanced_ser_model.pkl",
    "ser_cnn_bilstm_att_best.keras",
    "classes.pkl",
    "feature_scaler.pkl"
]

print("\n✅ Checking model files:")
for model_file in required_models:
    full_path = os.path.join(model_dir, model_file)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path) / (1024 * 1024)  # MB
        print(f"  ✓ {model_file} ({size:.2f} MB)")
    else:
        print(f"  ✗ {model_file} - NOT FOUND (optional)")

# Check 2: Required Python modules
print("\n✅ Checking Python modules:")
required_modules = [
    "streamlit",
    "numpy",
    "librosa",
    "soundfile",
    "tensorflow",
    "scikit-learn"
]

for module in required_modules:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError:
        print(f"  ✗ {module} - NOT INSTALLED")
        print(f"    Install with: pip install {module}")

# Check 3: SER package structure
print("\n✅ Checking SER package:")
ser_modules = [
    "ser",
    "ser.models",
    "ser.models.enhanced_emotion_model",
    "ser.config",
    "ser.utils"
]

for module in ser_modules:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError as e:
        print(f"  ✗ {module} - ERROR: {e}")

# Check 4: Configuration files
print("\n✅ Checking configuration files:")
config_files = [
    "streamlit_app.py",
    "requirements_streamlit.txt",
    "packages.txt",
    ".streamlit/config.toml"
]

for config_file in config_files:
    if os.path.exists(config_file):
        print(f"  ✓ {config_file}")
    else:
        print(f"  ✗ {config_file} - MISSING")

print("\n" + "=" * 50)
print("🎉 Deployment readiness check complete!")
print("=" * 50)

print("\n📝 Next steps:")
print("1. Run locally: streamlit run streamlit_app.py")
print("2. Test with sample audio file")
print("3. Push to GitHub: git add . && git commit -m 'Deploy' && git push")
print("4. Deploy on Streamlit Cloud: https://share.streamlit.io")
print("\n✨ Your live link will be ready in minutes!")
