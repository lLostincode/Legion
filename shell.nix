{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python
    python311
    python311Packages.pip
    
    # Build tools
    gcc
    gnumake
    
    # Development tools
    git
    
    # Additional development dependencies
    openssl
    pkg-config
  ];

  shellHook = ''
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements if requirements.txt exists
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi
    
    echo "Python development environment ready!"
  '';
}
