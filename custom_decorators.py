from functools import wraps

def pip(libraries):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            import os
            import subprocess
            import sys
            for library, version in libraries.items():
                print("Pip Install:", library, version)
                if version != "":
                    subprocess.run([sys.executable, "-m", "pip", "install", library + "==" + version])
                else:
                    subprocess.run([sys.executable, "-m", "pip", "install", library])
            return function(*args, **kwargs)
        return wrapper
    return decorator