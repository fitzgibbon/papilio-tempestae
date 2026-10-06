#!/usr/bin/env bash
# Builds the browser version of planet_lod into dist/. Run inside `nix develop`.
set -euo pipefail

cd "$(dirname "$0")/.."
out=dist

cargo build --profile web --target wasm32-unknown-unknown --example planet_lod

rm -rf "$out"
mkdir -p "$out/assets/shaders"
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  target/wasm32-unknown-unknown/web/examples/planet_lod.wasm
wasm-opt -Os --enable-bulk-memory --enable-nontrapping-float-to-int \
  "$out/planet_lod_bg.wasm" -o "$out/planet_lod_bg.wasm"

cp web/index.html "$out/"
cp assets/shaders/terrain.wgsl assets/shaders/render_shaders.wgsl assets/shaders/water.wgsl \
  "$out/assets/shaders/"
touch "$out/.nojekyll"
