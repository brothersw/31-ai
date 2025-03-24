{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    # idk if I need all of these yet
    python311Packages.torch-bin
    python311Packages.numpy
    python311Packages.matplotlib
    python311Packages.scipy
  ];
}