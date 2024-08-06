#!/usr/bin/env python3

import os
import subprocess
import sys

def update_version():
    with open('setup.py', 'r') as f:
        content = f.read()
    
    import re
    version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if version_match:
        version = version_match.group(1)
        parts = version.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        new_version = '.'.join(parts)
        new_content = re.sub(r'(version\s*=\s*["\'])([^"\']+)(["\'])', r'\g<1>' + new_version + r'\g<3>', content)
        
        with open('setup.py', 'w') as f:
            f.write(new_content)
        
        return new_version
    else:
        raise ValueError("Version not found in setup.py")
def main():
    # Versiyon numarasını güncelle
    new_version = update_version()
    print(f"Version updated to {new_version}")

    # Eski dağıtım dosyalarını temizle
    os.system('rm -rf dist build *.egg-info')

    # Yeni dağıtım dosyalarını oluştur
    subprocess.check_call([sys.executable, 'setup.py', 'sdist', 'bdist_wheel'])

    # Twine ile PyPI'ya yükle
    subprocess.check_call(['twine', 'upload', 'dist/*'])

    print(f"Successfully uploaded version {new_version} to PyPI")

if __name__ == "__main__":
    main()