# Kompute Pi4 Mesa Build Example

The Raspberry Pi 4 is an impressive little computer when you consider that the Broadcom GPU is able to run 2x 4K displays. This example intends to show how to get Kompute up and running on a Raspberry Pi 4. This has huge potential for edge processing using the power of the Pi 4 GPU.

Special thanks to [Alejandro Piñeiro](https://blogs.igalia.com/apinheiro/) and others for their work on Broadcom drivers for [Mesa](https://gitlab.freedesktop.org/mesa/mesa) which make this example possible.

## Raspberry Pi Operating System

For this experiment we used [RaspiOS Lite 2021-01-12](https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2021-01-12/2021-01-11-raspios-buster-armhf-lite.zip), though it is likely best to start with the latest available operation system from [Raspberry Pi operating system images](https://www.raspberrypi.org/software/operating-systems/). In other experiments the full Raspberry Pi operating system (with desktop environment) was found to work. However, when attempting to use Ubuntu on the Raspberry Pi we were not able to run the Python Kompute examples.

## Running the Pi headless

By far the easiest way to get up and running with a Raspberry Pi is to configure it for headless operation. This removes the requirement to have a monitor, keyboard or mouse. To run headless the Pi needs access to the internet and for SSH enabled. The following guides from the Raspberry Pi foundation should help.

- [Setting up a Raspberry Pi headless](https://www.raspberrypi.org/documentation/configuration/wireless/headless.md)
- [SSH (Secure Shell)](https://www.raspberrypi.org/documentation/remote-access/ssh/)

## Ensure all packages are using the latest version

```
sudo apt-get update
sudo apt-get upgrade
```

## Install dependencies for building mesa and running Kompute

```
sudo apt-get install \
    git build-essential cmake \
    python3-dev python3-mako python3-venv \
    flex bison meson ninja-build \
    libxcb-shm0-dev libxcb1-dev libxcb-*-dev \
    libx11-dev libx11-xcb-dev x11proto-dri2-dev x11proto-dri3-dev \
    libdrm-dev libxshmfence-dev libxrandr-dev libxfixes-dev \
    vulkan-tools libvulkan-dev
```

## Clone mesa repository

```
git clone --depth 1 https://gitlab.freedesktop.org/mesa/mesa.git
```

## Build mesa

Use meson and ninja to build mesa using the Broadcom Vulkan SDK drivers. For information on the Gallium drivers please see [V3D — The Mesa 3D Graphics Library latest documentation](https://docs.mesa3d.org/drivers/v3d.html).

```
meson --libdir lib \
    --prefix /mesa-install \
    -D platforms=x11 \
    -D vulkan-drivers=broadcom \
    -D gallium-drivers=v3d \
    -D dri-drivers=[] \
    -D buildtype=debug \
    build

ninja -C build
sudo ninja -C build install
```

## Configure preferred Vulkan SDK driver

Export the path for the Broadcom drivers, this command will need to be run for every new terminal session.

```
export VK_ICD_FILENAMES=/mesa-install/share/vulkan/icd.d/broadcom_icd.armv7l.json
```

## Allow access to render

In order to access the render from remote login there are two options. Both options work.

**Option 1: provide read write access to everyone.**

```
sudo chmod ugo+rw /dev/dri/renderD128
```

**Option 2: Change group from render to video.**

```
sudo chown root:video /dev/dri/renderD128
```

## Confirm correct Vulkan SDK operation

To confirm that mesa was configured and built correctly run the following command.

```
vulkaninfo
```

## Clone Kompute

Clone Kompute for access to the latest Python tests.

```
git clone https://github.com/KomputeProject/kompute.git
```

## Install dependencies to run the tests 

Navigate to the available tests and install required dependencies.

```
cd kompute/python/test
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements-dev.txt
pip install git+git://github.com/KomputeProject/kompute.git
```

## Run the available tests

Use the following command to run the python tests for Kompute.

```
pytest
```

If the tests pass then congratulations! You are now able to make full use of the Pi 4 Broadcom GPU for running parallel computing. If however, there are any issues with the tests they can be run in debug mode to see the logs.

```
pytest --log-cli-level debug
```

Please share any issues with the maintainers and they will be more than happy to help.

## Closing remarks

To avoid the need to export `VK_ICD_FILENAMES` every time you login, it is possible to symlink the json file into the default directory. The Vulkan SDK loader looks in the `/etc/vulkan/icd.d/` directory for `.json` files.

```
sudo ln -s /mesa-install/share/vulkan/icd.d/broadcom_icd.armv7l.json /etc/vulkan/icd.d/broadcom_icd.armv7l.json
```

As a word of warning, configuring the icd filenames in this way will stop certain tests being skipped. At the time of writing this will mean that some tests fail when running on the Pi.
