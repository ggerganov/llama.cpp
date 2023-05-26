# Nexus

Welcome to the Nexus! This README will guide you through setting up your development environment and getting started with the project.

## Prerequisites

Before you begin, make sure you have the following installed on your system:

- [Nix](https://nixos.org)
- [Git](https://git-scm.com/)
- [Rust](https://www.rust-lang.org/)
- wasm32-unknown-unknown target
- [OpenSSL 1.1](https://www.openssl.org/)
- [Cosmoonic](https://cosmonic.com)

## Getting Started

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/plurigrid/nexus.git
cd nexus
```

2. Start nix shell and run the following command to install all required dependencies:

```bash
nix-shell
make all
```

This will automatically check for and install Rust, wasm32-unknown-unknown target, OpenSSL 1.1, and Cosmo CLI if they are not already installed on your system.

3. Create a new actor using Cosmo CLI:

```bash
cosmo new actor <your_project_name>
```

Replace `<your_project_name>` with your desired project name.

4. Navigate to your newly created project directory:

```bash
cd <your_project_name>
```

5. Edit `src/lib.rs` file in your favorite text editor.

The default file content looks like this:

```rust
use wasmbus_rpc::actor::prelude::*;
use wasmcloud_interface_httpserver::{HttpRequest, HttpResponse, HttpServer, HttpServerReceiver};

#[derive(Debug, Default, Actor, HealthResponder)]
#[services(Actor, HttpServer)]
struct <your_project_name>Actor {}

/// Implementation of the HttpServer capability contract
#[async_trait]
impl HttpServer for <your_project_name>Actor {
    async fn handle_request(&self, _ctx: &Context, _req: &HttpRequest) -> RpcResult<HttpResponse> {
        Ok(HttpResponse::ok("message"))
    }
}
```

You can modify the file to accommodate more text like this:

```rust
use wasmbus_rpc::actor::prelude::*;
use wasmcloud_interface_httpserver::{HttpRequest, HttpResponse, HttpServer, HttpServerReceiver};

#[derive(Debug, Default, Actor, HealthResponder)]
#[services(Actor, HttpServer)]
struct <your_project_name>Actor {}

/// Implementation of the HTTP server capability
#[async_trait]
impl HttpServer for <your_project_name>Actor {
    async fn handle_request(&self, _ctx: &Context, _req: &HttpRequest) -> RpcResult<HttpResponse> {
        let message: &str = r#"message"#;

        Ok(HttpResponse::ok(message))
    }
}

```
## Launching the Project

1. Login to Cosmonic:

```bash
cosmo login
```

2. Build and sign your actor:

```bash
cosmo build
```

3. Start your wasmCloud host:

```bash
cosmo up
```

4. Launch the actor using Cosmo CLI:

```bash
cosmo launch
```

5. Navigate to [Cosmonic App](https://app.cosmonic.com) and sign in with your account.

6. In the Logic view, you should see the new actor you just launched.

7. To make your actor accessible from the web, launch a new provider for an HTTP server with the following OCI URL: `cosmonic.azurecr.io/httpserver_wormhole:0.5.3`. Give the link a name, and note that the HTTP server must be launched on a Cosmonic Manager resource.

8. Once the HTTP server is launched, link it to your actor.

9. Launch a wormhole and connect it to your actor link (the HTTP server and the actor).

10. Your actor should now be accessible at the domain of the wormhole followed by `.cosmonic.app`. For example: `https://white-morning-5041.cosmonic.app`.

Now you can access your project from any web browser using the provided URL!

You're all set! You can start building your project and explore the Nexus repository. Happy coding!
