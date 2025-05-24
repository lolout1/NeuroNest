#!/bin/bash

echo "🚀 Deploying Fixed Ultra-Comprehensive Analysis System..."

# Install required packages
echo "📦 Installing required packages..."
python -c "
import sys
packages = ['scipy', 'scikit-learn']
missing = []
for pkg in packages:
    try:
        if pkg == 'scikit-learn':
            import sklearn
        else:
            __import__(pkg)
    except ImportError:
        missing.append(pkg)
        
if missing:
    print(f'Installing missing packages: {missing}')
    import subprocess
    for pkg in missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
else:
    print('All required packages installed')
"

echo "✅ System deployed with enhancements:"
echo "   🔍 Ultra-sensitive contrast detection"
echo "   ⚫ Floors-only blackspot detection (black areas only)"
echo "   🎨 Multiple visualization modes"
echo "   📊 Robust error handling"
echo "   🔄 Combined analysis view"
echo ""
echo "🎯 Visualization Options:"
echo "   • High Contrast: Enhanced overlays"
echo "   • Side by Side: Original vs Analysis"
echo "   • Blackspots Only: Pure blackspot view"
echo "   • Segmentation Only: Clean segmentation"
echo "   • Annotated: Detailed annotations"
echo "   • Combined Analysis: Blackspots + Contrast"
echo ""
echo "🚀 Ready to test!"
chmod +x deploy.sh
