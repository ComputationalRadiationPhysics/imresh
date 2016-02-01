The style mentioned here don't have to be followed, but it would be nice and it would be nice not to change something using these style back to something different.

1. Type qualifiers inside declarations

    Declarations inside headers should be indentical to their definitions. Type qualifiers leaving the function signature unchanged is more or less a C-only feature.

        void func( float * const __restricted__ );

    instead of

        void func( float * );

    In the case of CUDA kernels i.e. declarations with `__global__` or `__device__` specifiers and if they are templated, then the CUDA kernel launch will abort the whole program even though the program will compile. See [this](http://stackoverflow.com/questions/35106360/why-does-this-cuda-program-crash-when-omitting-the-const-qualifier) Stackoverflow question.

2. Type specifiers

    The complete forms should be preferred to short versions.

        unsigned int

    instead of

        unsigned

    This is not necessary for `signed int`, `int` suffices in this case.

