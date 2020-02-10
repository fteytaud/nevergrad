# nix-shell -- run "python -m nevergrad.benchmark illcond --seed=12 --repetitions=1 --plot"

with import <nixpkgs> {};

(let

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

      nevergrad = self.buildPythonPackage {
        pname = "nevergrad";
        version = "git";
        src = ./.;
        buildInputs = [self.pytest];
        propagatedBuildInputs = [
          self.bayesian-optimization
          self.cma
          self.gym 
          self.matplotlib
          self.pandas
          self.pytorch
          self.typing-extensions
        ];
      };

    };

  in pkgs.python3.override { inherit packageOverrides; self = python; };

in python.withPackages(ps: [ ps.nevergrad ])).env

