# SRPM for building from source and packaging an RPM for RPM-based distros.
# https://fedoraproject.org/wiki/How_to_create_an_RPM_package
# Built and maintained by John Boero - boeroboy@gmail.com
# In honor of Seth Vidal https://www.redhat.com/it/blog/thank-you-seth-vidal

# Notes for llama.cpp:
# 1. Tags are currently based on hash - which will not sort asciibetically.
#    We need to declare standard versioning if people want to sort latest releases.
# 2. Builds for CUDA/OpenCL support are separate, with different depenedencies.
# 3. NVidia's developer repo must be enabled with nvcc, cublas, clblas, etc installed.
#    Example: https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
# 4. OpenCL/CLBLAST support simply requires the ICD loader and basic opencl libraries.
#    It is up to the user to install the correct vendor-specific support.

Name:           llama.cpp
Version:        master
Release:        1%{?dist}
Summary:        CPU Inference of LLaMA model in pure C/C++ (no CUDA/OpenCL)
License:        MIT
Source0:        https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.tar.gz
BuildRequires:  coreutils make gcc-c++ git
URL:            https://github.com/ggerganov/llama.cpp

%define debug_package %{nil}
%define source_date_epoch_from_changelog 0

%description
CPU inference for Meta's Lllama2 models using default options.

%prep
%autosetup

%build
make -j

%install
mkdir -p %{buildroot}%{_bindir}/
cp -p main %{buildroot}%{_bindir}/llamacpp
cp -p server %{buildroot}%{_bindir}/llamacppserver
cp -p simple %{buildroot}%{_bindir}/llamacppsimple

%clean
rm -rf %{buildroot}
rm -rf %{_builddir}/*

%files
%{_bindir}/llamacpp
%{_bindir}/llamacppserver
%{_bindir}/llamacppsimple

%pre

%post

%preun
%postun

%changelog
