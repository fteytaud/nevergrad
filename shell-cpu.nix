with import <nixpkgs> {};

(let

  rev = "dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b";

  channel = fetchTarball "https://github.com/NixOS/nixpkgs/archive/${rev}.tar.gz";

  config = {
    allowUnfree = true;
    cudaSupport = false;
  };

  pkgs = import channel { inherit config; };

  python = let
    packageOverrides = self: super: {

      bayesian-optimization = self.buildPythonPackage rec {
        pname = "bayesian-optimization";
        version = "1.0.1";
        src = self.fetchPypi {
          inherit pname version;
          sha256 = "0j0cwicq6y4bbhpgwppay4cnv347zdmw2lk9k4gl7zn3pl6kkfmp";
        };
        propagatedBuildInputs = [
          self.scikitlearn
          self.scipy
        ];
      };

      cma = self.buildPythonPackage rec {
        pname = "cma";
        version = "2.7.0";
        src = self.fetchPypi {
          inherit pname version;
          sha256 = "0nfdq7qznfqcnqwbk04812xiddq8azzbvdj2pa2lvgzxbr00sqzl";
        };
        propagatedBuildInputs = [
          self.numpy
        ];
      };
      
      nevergrad = self.buildPythonPackage rec {
        pname = "nevergrad";
        version = "git";
        src = ./.;
        buildInputs = [ self.pytest ];
        propagatedBuildInputs = [
          self.bayesian-optimization
          self.cma
          self.matplotlib
          self.pandas
          self.typing-extensions
          self.pytorchWithoutCuda
          self.requests
          self.gym
          self.pytest
        ];
        # doCheck = false;
      };

    };
  in pkgs.python3.override {inherit packageOverrides; self = python;};

in python.withPackages(ps: [ps.nevergrad])).env
