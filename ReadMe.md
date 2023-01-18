To build and run the docker:
```
docker build . -t thorin
docker run -it --rm -t thorin
```

You start in the ad directory inside the docker.
Execute `./compile [file.impala]` to get started.
The result is placed into `./build/[file]`.


Links:
* [NeuralCoder3/thorin2 Branch: feature/autodiff-for-null](https://github.com/NeuralCoder3/thorin2/tree/feature/autodiff-for-null)
* [NeuralCoder3/impala Branch: feature/autodiff-for](https://github.com/NeuralCoder3/impala/tree/feature/autodiff-for)
* [christopherhjung/impala-adbench](https://github.com/christopherhjung/impala-adbench)
