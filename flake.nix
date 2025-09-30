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
      in
      {
        devShells.default =
          pkgs.mkShell.override
            {
              stdenv = pkgs.stdenvAdapters.useMoldLinker pkgs.clangStdenv;
            }
            {
              packages = with pkgs; [
                gcc
                rustup
                pkg-config
                rust-analyzer
                typos-lsp

                # Needed by bevy
                alsa-lib
                udev
                xorg.libX11
                xorg.libXcursor
                xorg.libXi
              ];

              shellHook = ''
                alias clippy="cargo +nightly clippy --all-targets --all-features"
                alias test="cargo +nightly test --all-targets --all-features"

                export LD_LIBRARY_PATH=${
                  pkgs.lib.makeLibraryPath [
                    pkgs.libxkbcommon
                    pkgs.vulkan-loader
                  ]
                }
              '';
            };
      }
    );
}
