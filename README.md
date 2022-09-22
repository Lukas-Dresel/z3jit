# z3jit

A JIT engine to compile z3 constraints to make model-validation as fast as possible

This uses the LLVM framework to JIT a set of z3 constraints to native code to make validating a given model 
(ensuring that it satisfies the given constraints) blazingly fast. 

This is a highly WIP project, with a ton of unimplemented functionality, mainly the things I've encountered needing in my research project, so no
guarantees that this will work for your use case. However, if you have any functionality you'd like to see added, feel free to create an issue
and I'm more than happy to take a look. Alternatively PRs are also always welcome to add functionality :)

- Lukas

# Requirements

Requires `llvm` to be installed on your machine. Update `Cargo.toml` with the corresponding version of the `inkwell` rust bindings for your version of LLVM.
