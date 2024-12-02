
CI, Docker Images, Docs & Tests
======================

This section contains an overview of the steps run on CI, as well as the tools used to simplify the testing (such as running on CPU).

We use Github Actions to run the tests, which simplifies the workflows significantly for contributors. 

The tests run on CPU, and can be triggered using the ACT command line interface (https://github.com/nektos/act) - once you install the command line (And start the Docker daemon) you just have to type:

.. code-block::

    $ act

    [Python Tests/python-tests] 🚀  Start image=axsauze/kompute-builder:0.2
    [C++ Tests/cpp-tests      ] 🚀  Start image=axsauze/kompute-builder:0.2
    [C++ Tests/cpp-tests      ]   🐳  docker run image=axsauze/kompute-builder:0.2 entrypoint=["/usr/bin/tail" "-f" "/dev/null"] cmd=[]
    [Python Tests/python-tests]   🐳  docker run image=axsauze/kompute-builder:0.2 entrypoint=["/usr/bin/tail" "-f" "/dev/null"] cmd=[]
    ...


CI Commands Triggered
~~~~~~~~~~~~~

The simplest way to see how this works is by looking at the github actions commands that are run.

These can be found through the following files:

* `CPP Tests <https://github.com/KomputeProject/kompute/blob/master/.github/workflows/cpp_tests.yml>`_
* `Python Tests <https://github.com/KomputeProject/kompute/blob/master/.github/workflows/python_tests.yml>`_

When submitting a PR or merging a PR into master, both of these will run - you can see the logs through the github interface.



Running on the CPU
~~~~~~~~~~~~~

We use `Swiftshader <https://github.com/google/swiftshader>`_ to enable us to run the Kompute framework directly on the CPU for the CI tests.

Even though Swiftshader is optimized to function as a high-performance CPU backend for the Vulkan SDK, there are several limitations, the most notable are limitations in extensions.

This is one of the main reason why only a subset of the tests are run in the CI.

Dockerfiles
~~~~~~~~~~~~~

The dockerfiles created provide functionality to simplify the interaction with the system. 

.. list-table::
   :header-rows: 1

   * - Image
     - Description
   * - axsauze/kompute-builder:0.2
     - Main CI builder image with all required dependencies to build and run C++ & Python tests.
   * - axsauze/swiftshader:0.1
     - Image building Swiftshader libraries only to reduce time via multi-staged builds
   * - axsauze/vulkan-sdk:0.1
     - Image contained a linux build of the full Vulkan SDK to reduce time via multi-staged builds


Running / Building Documentation
~~~~~~~~~~~~~

In order to build the documentation you will need the following dependencies:

* Install CI dependencies under `scripts/requirements.txt`

Once this installed:

* You can build the documentation using the `gendocsall` cmake target
* You can serve the documentation locally using the `mk_run_docs` command in the Makefile

Performing Release
~~~~~~~~~~~~

In order to perform the release the following steps need to be carried out:

* Build changelog
    * Create branch called `v<VERSION>-release`
    * Generate latest changelog `make build_changelog`
    * Update latest tag in new CHANGELOG.md to be the vesion to release 
* Python Release
    * Build dependency:
        * Intsall dependency: `pip install .`
        * Ensure all tests pass in GPU and CPU: `python -m pytest`
        * Build distribution `python setup.py sdist bdist_wheel`
    * Test repo:
        * Push to test repo `python -m twine upload --repository testpypi dist/*`
        * Install python dependency: `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps kp`
        * Ensure all tests pass in GPU and CPU: `python -m pytest`
    * Prod repo:
        * Push to test repo `python -m twine upload dist/*`
        * Install package from prod pypi `pip install kp`
        * Ensure all tests pass in GPU and CPU: `python -m pytest`


