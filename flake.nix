{
  description = "Bevy and Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkg-config
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
