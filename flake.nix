{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
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
        deps = with pkgs; [
          gcc
          rustup
          pkg-config
          rust-analyzer
          typos-lsp
          wgsl-analyzer
          openssl # For cargo-outdated

          # Needed by bevy
          alsa-lib
          udev
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          wayland
          libxkbcommon
          vulkan-loader
          mesa # Is this needed?
        ];
      in
      {
        devShells.default =
          pkgs.mkShell.override
            {
              stdenv = pkgs.stdenvAdapters.useMoldLinker pkgs.clangStdenv;
            }
            {
              packages = deps;

              shellHook = ''
                alias clippy="cargo +nightly clippy --all-targets --all-features"
                alias test="cargo +nightly test --all-targets --all-features"

                export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath deps}
              '';
            };
      }
    );
}
