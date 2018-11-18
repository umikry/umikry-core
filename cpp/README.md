# umikry-core (C++)

This is the C++ umikry-core implementation to support the integation to other languages and applications.

## Setup

```zsh
pip(3) install conan
cd build
conan install ..
cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .
./bin/umikry -s "/Users/fabianbormann/test.jpg" -d "/Users/fabianbormann/test_out.jpg"
```