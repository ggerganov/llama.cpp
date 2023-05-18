Title: Cosmonic fails to launch server on macOS due to incorrect OpenSSL version

**Description:**

Cosmonic is unable to properly launch a server on macOS systems because the default OpenSSL version provided by the operating system is LibreSSL, which is incompatible with Cosmonic's requirements. The Cosmonic documentation suggests installing OpenSSL 1.1 using Homebrew (`brew install openssl@1.1`), which resolves the issue for installations managed by Homebrew. However, when attempting to use Nix to install the same package, the issue persists.

It is assumed that this problem occurs because Homebrew installs OpenSSL 1.1 as a "keg-only" package, avoiding symlinking it to avoid conflicts with macOS's LibreSSL dependencies. As a result, Cosmonic may be hard-coded to look for the OpenSSL installation in the Homebrew-specific directory, causing it to miss the Nix installation.

**Steps to reproduce:**

1. Use macOS with the default LibreSSL version installed.
2. Try to launch a Cosmonic server.
3. Observe the server failing to start due to the incorrect OpenSSL version.
4. Install OpenSSL 1.1 using Nix.
5. Attempt to launch the Cosmonic server again.
6. Observe the server still failing to start, even with the correct OpenSSL version installed via Nix.

**Expected behavior:**

Cosmonic should be able to detect and utilize the correct OpenSSL version installed via Nix, similar to how it works with Homebrew installations.

**System information:**

- macOS version: (please provide your macOS version)
- Cosmonic version: (please provide your Cosmonic version)
- Nix version: (please provide your Nix version)

**Possible solution:**

Update Cosmonic to search for OpenSSL installations in common Nix paths, or provide a configuration option to specify the location of the OpenSSL installation.
