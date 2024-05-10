# SRPM for building from source and packaging an RPM for RPM-based distros.
# https://docs.fedoraproject.org/en-US/quick-docs/creating-rpm-packages
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

Name:           llama.cpp-clblast
Version:        %( date "+%%Y%%m%%d" )
Release:        1%{?dist}
Summary:        OpenCL Inference of LLaMA model in C/C++
License:        MIT
Source0:        https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.tar.gz
BuildRequires:  coreutils make gcc-c++ git mesa-libOpenCL-devel clblast-devel
Requires:       clblast
URL:            https://github.com/ggerganov/llama.cpp

%define debug_package %{nil}
%define source_date_epoch_from_changelog 0

%description
CPU inference for Meta's Lllama2 models using default options.

%prep
%setup -n llama.cpp-master

%build
make -j LLAMA_CLBLAST=1

%install
mkdir -p %{buildroot}%{_bindir}/
cp -p main %{buildroot}%{_bindir}/llamaclblast
cp -p server %{buildroot}%{_bindir}/llamaclblastserver
cp -p simple %{buildroot}%{_bindir}/llamaclblastsimple

mkdir -p %{buildroot}/usr/lib/systemd/system
%{__cat} <<EOF  > %{buildroot}/usr/lib/systemd/system/llamaclblast.service
[Unit]
Description=Llama.cpp server, CPU only (no GPU support in this build).
After=syslog.target network.target local-fs.target remote-fs.target nss-lookup.target

[Service]
Type=simple
EnvironmentFile=/etc/sysconfig/llama
ExecStart=/usr/bin/llamaclblastserver $LLAMA_ARGS
ExecReload=/bin/kill -s HUP $MAINPID
Restart=never

[Install]
WantedBy=default.target
EOF

mkdir -p %{buildroot}/etc/sysconfig
%{__cat} <<EOF  > %{buildroot}/etc/sysconfig/llama
LLAMA_ARGS="-m /opt/llama2/ggml-model-f32.bin"
EOF

%clean
rm -rf %{buildroot}
rm -rf %{_builddir}/*

%files
%{_bindir}/llamaclblast
%{_bindir}/llamaclblastserver
%{_bindir}/llamaclblastsimple
/usr/lib/systemd/system/llamaclblast.service
%config /etc/sysconfig/llama


%pre

%post

%preun
%postun

%changelog
