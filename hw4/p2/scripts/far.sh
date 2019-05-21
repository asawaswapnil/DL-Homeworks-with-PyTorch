sudo apt install cmake
version=3.13
build=3
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/

./bootstrap
make -j4
sudo make install

cmake --version

sudo pip3 install python-Levenshtein
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make


cd pytorch_binding
python setup.py install

cd ../../..

mkdir hw3

git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .

sudo apt remove --purge --auto-remove cmake


conda install pytorch torchvision cudatoolkit=9.0 -c pytorch







# mv .py var
# mv abhi2.py var
# mv val.tar.xz var
# mv abhidata.tar.xz var
# cd var
# tar xf val.tar.xz
# tar xf abhidata.tar.xz
# cd ..
# mv v2mob3.tar.xz var/mnet/
# cd var/mnet/
# tar xf v2mob3.tar.xz
# screen 

# python3 abhi2.py

