from setuptools import find_packages, setup
import subprocess, os

this_dir = os.path.dirname(os.path.abspath(__file__))
varuna_dir = os.path.join(this_dir, "varuna")
cmd = ["g++", "-std=c++11", "generate_schedule.cc", "-o", "genschedule"]
subprocess.run(cmd, cwd=varuna_dir, check=True)
tools_dir = os.path.join(this_dir, "tools", "simulator")
cmd = ["g++","-std=c++11", "simulate-varuna-main.cc", "generate_schedule.cc", "simulate-varuna.cc", "-o", "simulate-varuna"]
subprocess.run(cmd, cwd=tools_dir, check=True)
setup(
    name="varuna",
    version="0.0.1",
    author="MSR India",
    author_email="muthian@microsoft.com",
    description="Pipeline parallel training for PyTorch",
    keywords='deep learning microsoft research pipelining',
    packages=['varuna']
)
# cmd = ["cp","-r", "/workspace/varuna/tools", "/home/ubuntu/.local/lib/python3.8/site-packages/varuna-0.0.1-py3.6.egg"]
# subprocess.run(cmd, cwd=tools_dir, check=True)
# cmd = ["cp", "-r", "/workspace/varuna/varuna/", "/home/ubuntu/.local/lib/python3.8/site-packages/varuna-0.0.1-py3.6.egg/"]
# subprocess.run(cmd, cwd=tools_dir, check=True)
# cmd = ["cp", "-r", "/workspace/varuna/varuna/", "/home/ubuntu/.local/lib/python3.8/site-packages/"]
# subprocess.run(cmd, cwd=tools_dir, check=True)
# cmd = ["cp","-r", "/workspace/varuna/tools", "/home/ubuntu/.local/lib/python3.8/site-packages/"]
# subprocess.run(cmd, cwd=tools_dir, check=True)
# cmd = ["cp", "/workspace/varuna/varuna/genschedule", "/home/ubuntu/.local/lib/python3.8/site-packages/varuna/"]
# subprocess.run(cmd, cwd=tools_dir, check=True)