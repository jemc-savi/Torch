This repository was intended to be a wrapper for `libtorch` (the C++ library underpinning `pytorch`) for Savi.

During the process, I figured out that it would be better to wrap TensorFlow instead, so I created [these TensorFlow bindings](https://github.com/jemc-savi/Tensor) - you should use those instead (this Torch repo never got far off the ground).

Why did I choose TensorFlow over PyTorch?
- the TensorFlow codebase is much more mature and clean
- TensorFlow has mature features related to building declarative graphs that can execute outside of an imperative context, and which are transferable in a well-designed serialized format (via protobuf)
- the `libpytorch-dev` headers have a lot of issues with how they are structured that make them difficult to deal with in a portable way
