git clone git@github.com:Pseudomanifold/Aleph.git
cd Aleph/
mkdir build
cd build/

PYBIND_PATH=$(python3 -c "import pybind11; print(pybind11.get_include())")

cmake ../ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYBIND11_INCLUDE_DIR="$PYBIND_PATH"
make
make test

cd bindings/python/aleph/

# Define the version
VERSION="1.0.0"

# Use sed to find the version line and replace the empty quotes with the variable
# This looks for the pattern: version=''
sed -i '' "s/version='',/version='$VERSION',/" setup.py

python setup.py install