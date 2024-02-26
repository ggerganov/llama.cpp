# simple automation of cmake
# usage is ./cmakescript.sh Debug || Release

rm -r build
cmake -B build
cd build

if [ $# -eq 1 ] && [[ "$1" == "Debug" || "$1" == "Release" ]]; then
  cmake --build . --config "$1"
else
  echo "Usage: $0 (Debug|Release)"
  exit 1
fi

cd build
