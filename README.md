 Here is the complete set of instructions:

1. Clone the Nexus repository

```bash
git clone https://github.com/nexus
```

2. Enter the Nix environment

```bash
nix-shell
```

This will activate the flox environment defined in the nexus/flox.nix file. This environment has all the necessary dependencies installed to build and run your project.

3. Run cosmo

Now you can use cosmo launch your project:

```bash
cosmo
```

**If this is your first run of cosmo**, it will automatically start the tutorial.

**If not your first run**, you can start the tutorial with:

```bash
cosmo tutorial hello
```

4. Explaining the components

- flox is a tool for managing declarative Nix-based environments. The nexus/flox.nix file defines an environment with all the dependencies for your project.
- cosmo is a tool for building and deploying WebAssembly actors. You use it to build and launch your actor from within the flox environment.
- Nix is a purely functional package manager that is used by flox to define environments.

5. Installation (if not already completed)

Follow the instructions to install flox and configure your system to use it. This will install the necessary tools to get started with the Nexus project.

- Install Nix (if not already installed)
- Install flox

You now have all the necessary components installed and configured to build and run the Nexus project! Let me know if you have any other questions.
