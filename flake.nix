{
  description = "Bevy and Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system;
          inherit overlays;
        };
        rustToolchain = pkgs.rust-bin.nightly."2026-04-11".default.override {
          extensions = [ "rust-src" "rustc-dev" "llvm-tools-preview" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkg-config
            rustToolchain
          ];

          buildInputs = with pkgs; [
            udev
            alsa-lib
            vulkan-loader
            libx11
            libxcursor
            libxi
            libxrandr
            libxkbcommon
            wayland
          ];

          # Ensure dynamic libraries are found at runtime
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            udev
            alsa-lib
            vulkan-loader
            libx11
            libxcursor
            libxi
            libxrandr
            libxkbcommon
            wayland
          ]);
        };
      }
    );
}

