# This makefile is optimized to be run from WSL and to interact with the 
# Windows host as there are limitations when building GPU programs. This
# makefile contains the commands for interacting with the visual studio
# build via command line for faster iterations, as the intention is to 
# support other editors (optimised for vim). There are also commands that
# support the builds for linux-native compilations and these are the commands
# starting with mk_.

VERSION := $(shell cat ./VERSION)

VCPKG_WIN_PATH ?= "C:\\Users\\axsau\\Programming\\lib\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake"
VCPKG_UNIX_PATH ?= "/c/Users/axsau/Programming/lib/vcpkg/scripts/buildsystems/vcpkg.cmake"

# These are the tests that don't work with swiftshader but can be run directly with vulkan
FILTER_TESTS ?= "-TestAsyncOperations.TestManagerParallelExecution:TestSequence.SequenceTimestamps:TestPushConstants.TestConstantsDouble"

ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
	CMAKE_BIN ?= "C:\Program Files\CMake\bin\cmake.exe"
	SCMP_BIN="C:\\VulkanSDK\\1.2.141.2\\Bin32\\glslangValidator.exe"
	MSBUILD_BIN ?= "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe"
else
	CLANG_FORMAT_BIN ?= "/home/alejandro/Programming/lib/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang-format"
	CMAKE_BIN ?= "/c/Program Files/CMake/bin/cmake.exe"
	MSBUILD_BIN ?= "/c/Program Files (x86)/Microsoft Visual Studio/2019/Community/MSBuild/Current/Bin/MSBuild.exe"
	# Choosing the binary based on whether it's on WSL or linux-native
	KERNEL := $(shell uname -r)
	IS_WSL := $(shell (if [[ "$(KERNEL)" =~ Microsoft$  ]]; then echo '0'; fi))
	ifeq ($(IS_WSL),0)
		SCMP_BIN ?= "/c/VulkanSDK/1.2.141.2/Bin32/glslangValidator.exe"
	else
		SCMP_BIN ?= "/usr/bin/glslangValidator"
	endif
endif


####### Main Target Rules #######

push_docs_to_ghpages:
	GIT_DEPLOY_DIR="build/docs/sphinx/" \
		GIT_DEPLOY_BRANCH="gh-pages" \
		GIT_DEPLOY_REPO="origin" \
			./scripts/push_folder_to_branch.sh

####### CMAKE quickstart commands #######

clean_cmake:
	rm -rf build/

####### Visual studio build shortcut commands #######

MK_BUILD_TYPE ?= "Release"
MK_INSTALL_PATH ?= "build/src/CMakeFiles/Export/" # Set to "" if prefer default
MK_CMAKE_EXTRA_FLAGS ?= ""
MK_KOMPUTE_EXTRA_CXX_FLAGS ?= ""

mk_cmake:
	cmake \
		-Bbuild \
		-DCMAKE_CXX_FLAGS=$(MK_KOMPUTE_EXTRA_CXX_FLAGS) \
		-DCMAKE_BUILD_TYPE=$(MK_BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(MK_INSTALL_PATH) \
		-DKOMPUTE_OPT_INSTALL=ON \
		-DKOMPUTE_OPT_BUILD_TESTS=ON \
		-DKOMPUTE_OPT_BUILD_DOCS=ON \
		-DKOMPUTE_OPT_BUILD_SHADERS=ON \
		-DKOMPUTE_OPT_CODE_COVERAGE=ON \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DKOMPUTE_OPT_LOG_LEVEL=Debug \
		$(MK_CMAKE_EXTRA_FLAGS) \
		-G "Unix Makefiles"

mk_build_all:
	cmake --build build/. --parallel

mk_build_docs:
	cmake --build build/. --target gendocsall --parallel

mk_build_kompute:
	cmake --build build/. --target kompute --parallel

mk_build_tests:
	cmake --build build/. --target kompute_tests --parallel

mk_run_docs: mk_build_docs
	(cd build/docs/sphinx && python2.7 -m SimpleHTTPServer)

# An alternative would be: ctest -vv --test-dir build/.
# But this is not possible since we need to filter specific tests, not complete executables, which is not possible with ctest.
# https://gitlab.kitware.com/cmake/cmake/-/issues/13168 
mk_run_tests: mk_build_tests
	./build/bin/kompute_tests --gtest_filter=$(FILTER_TESTS)

mk_build_swiftshader_library:
	git clone https://github.com/google/swiftshader || echo "Assuming already cloned"
	# GCC 8 or above is required otherwise error on "filesystem" lib will appear
	CC="/usr/bin/gcc-8" CXX="/usr/bin/g++-8" cmake swiftshader/. -Bswiftshader/build/
	cmake --build swiftshader/build/. --parallel

mk_run_tests_cpu: export VK_ICD_FILENAMES=$(PWD)/swiftshader/build/vk_swiftshader_icd.json
mk_run_tests_cpu: mk_build_swiftshader_library mk_build_tests mk_run_tests_cpu_only


####### Visual studio build shortcut commands #######

VS_BUILD_TYPE ?= "Debug"
# Run with multiprocessin / parallel build by default
VS_CMAKE_EXTRA_FLAGS ?= ""
VS_KOMPUTE_EXTRA_CXX_FLAGS ?= ""
VS_INSTALL_PATH ?= "build/src/CMakeFiles/Export/" # Set to "" if prefer default

vs_cmake:
	$(CMAKE_BIN) \
		-Bbuild \
		$(VS_CMAKE_EXTRA_FLAGS) \
		-DCMAKE_TOOLCHAIN_FILE=$(VCPKG_WIN_PATH) \
		-DCMAKE_CXX_FLAGS=$(VS_KOMPUTE_EXTRA_CXX_FLAGS) \
		-DCMAKE_INSTALL_PREFIX=$(VS_INSTALL_PATH) \
		-DKOMPUTE_OPT_INSTALL=ON \
		-DKOMPUTE_OPT_BUILD_TESTS=ON \
		-DKOMPUTE_OPT_BUILD_SHADERS=ON \
		-DKOMPUTE_OPT_CODE_COVERAGE=OFF \
		-DKOMPUTE_OPT_BUILD_DOCS=OFF \
		-G "Visual Studio 16 2019" \
		-DCMAKE_BUILD_TYPE=$(VS_BUILD_TYPE)

vs_build_all:
	cmake --build build/. --parallel

vs_build_docs:
	cmake --build build/. --target gendocsall --parallel

vs_install_kompute:
	cmake --build build/. --target install --parallel

vs_build_kompute:
	cmake --build build/. --target kompute --parallel

vs_build_tests:
	cmake --build build/. --target kompute_tests --parallel

vs_run_docs: vs_build_docs
	(cd build/docs/sphinx && python2.7 -m SimpleHTTPServer)

vs_run_tests: vs_build_tests
	./build/test/$(VS_BUILD_TYPE)/bin/kompute_tests.exe --gtest_filter=$(FILTER_TESTS)


#### PYTHONG ####

test_python:
	python3 -m pytest -s --log-cli-level=DEBUG -v python/test/

####### Run CI Commands #######

# This command uses act to replicate github action
# https://github.com/nektos/act
run_ci:
	act

####### General project commands #######

generate_python_docstrings:
	python -m pybind11_mkdoc \
		-o python/src/docstrings.hpp \
		kompute/Kompute.hpp \
		-Iexternal/fmt/include/ \
		-Iexternal/spdlog/include/ \
		-Iexternal/glslang/ \
		-I/usr/include/c++/7.5.0/

install_python_reqs:
	python3 -m pip install -r scripts/requirements.txt

install_lcov:
	sudo apt install lcov -y

build_shaders:
	python3 scripts/convert_shaders.py \
		--shader-path shaders/glsl \
		--shader-binary $(SCMP_BIN) \
		--header-path src/include/kompute/shaders/ \
		-v
	python3 scripts/convert_shaders.py \
		--shader-path test/shaders/glsl \
		--shader-binary $(SCMP_BIN) \
		--header-path test/compiled_shaders_include/kompute_test/shaders/ \
		-v

build_single_header:
	quom \
		--include_directory \
		"src/include/" \
		"single_include/AggregateHeaders.cpp" \
		"single_include/kompute/Kompute.hpp"

win_build_xxd:
	cd external/bin/ && gcc.exe -o xxd.exe xxd.c -DCYGWIN

format:
	for val in "examples single_include src test" ; do \
    	find $$val -depth -iname *.h -or -iname *.c -or -iname *.hpp -or -iname *.cpp | grep -v "shaders" | xargs $(CLANG_FORMAT_BIN) -style=file -i; \
	done

static_scan:
	cppcheck --project=build/compile_commands.json -iexternal/

build_changelog:
	docker run --rm -it -v "$(PWD)":/usr/local/src/your-app -e CHANGELOG_GITHUB_TOKEN=${CHANGELOG_GITHUB_TOKEN} ferrarimarco/github-changelog-generator:1.15.2 -u KomputeProject -p kompute
	chmod 664 CHANGELOG.md # (Read+Write, Read+Write, Read)
	sed -i -e 's/\(HEAD\|Unreleased\)/v${VERSION}/g' CHANGELOG.md # Replacing unreleased version with latest tag
