{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.mypy # static type checker
    python311Packages.matplotlib
  ];
}
