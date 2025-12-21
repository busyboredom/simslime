{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        runtimeLibs = with pkgs; [
          udev
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          wayland
          libxkbcommon
          vulkan-loader
          mesa
        ];

        buildInputs = with pkgs; [
          pkg-config
          rustup
          rust-analyzer
          lldb # Debugger
          typos-lsp
          wgsl-analyzer
          openssl # For cargo-outdated
        ];

        clippyCmd = pkgs.writeShellScriptBin "clippy" ''
          cargo +nightly clippy --all-targets --all-features "$@"
        '';

        testCmd = pkgs.writeShellScriptBin "run-tests" ''
          cargo +nightly test --all-targets --all-features "$@"
        '';

      in
      {
        devShells.default =
          pkgs.mkShell.override
            {
              stdenv = pkgs.stdenvAdapters.useMoldLinker pkgs.clangStdenv;
            }
            {
              packages =
                buildInputs
                ++ runtimeLibs
                ++ [
                  clippyCmd
                  testCmd
                ];

              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

              shellHook = ''
                export PATH=$PATH:${pkgs.vscode-extensions.vadimcn.vscode-lldb}/share/vscode/extensions/vadimcn.vscode-lldb/adapter

                # Disable rerun analytics
                ~/.cargo/bin/rerun analytics disable
              '';
            };
      }
    );
}
