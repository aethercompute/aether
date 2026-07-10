{
  psycheLib,
  python312,
  stdenvNoCC,
  config,
  lib,
}:
let
  inherit (psycheLib)
    cargoArtifacts
    craneLib
    rustWorkspaceArgs
    ;

  # build the actual rust extension that the python aether code loads
  rustExtension = craneLib.buildPackage (
    rustWorkspaceArgs
    // {
      inherit cargoArtifacts;
      pname = "aether-python-extension";
      cargoExtraArgs =
        " --package aether-python-extension --features python-extension"
        + lib.optionalString (config.cudaSupport) ",parallelism";
      doCheck = false;
    }
  );

  # expected lib file ext for the python extension
  ext = if stdenvNoCC.isDarwin then "dylib" else "so";

in
# a combination of the python files and rust ext for the aether python code
stdenvNoCC.mkDerivation {
  __structuredAttrs = true;

  name = "aether";
  version = "0.2.0";

  src = ./python/aether;

  installPhase = ''
    runHook preInstall

    # create python package dir
    mkdir -p $out/${python312.sitePackages}/aether

    # copy all python files
    cp -r * $out/${python312.sitePackages}/aether/

    # copy the extension binary file
    cp ${rustExtension}/lib/lib${builtins.replaceStrings [ "-" ] [ "_" ] rustExtension.pname}.${ext} \
       $out/${python312.sitePackages}/aether/_aether_ext.so

    runHook postInstall
  '';

  doCheck = false;
}
