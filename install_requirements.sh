# install dependencies
echo "Installing dependencies..."
python3 -m pip install --no-cache-dir --upgrade -r requirements.txt
python3 -m pip install numpy==1.26.2
python3 -m pip install urllib3==1.26.6

# install cuda-12.6
cd ../LLaVA_KD_whls/
sh cuda_12.9.1_575.57.08_linux.run

cd ../LLaVA_KD_whls/bitsandbytes-0.46.0
python3.12 setup.py install 

pip install ptflops
pip install deepspeed==0.14.4

# maybe not necessary
# cd ../LLaVA_KD_whls/
# rpm -i cusparselt-local-repo-rhel9-0.7.1-1.0-1.x86_64.rpm
# dnf clean all
# dnf -y install libcusparselt0 libcusparselt-devel
