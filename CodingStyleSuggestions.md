The style mentioned here don't have to be followed, but it would be nice and it would be nice not to change something using these style back to something different.

1. Type qualifiers inside declarations

    Declarations inside headers should be concise, meaning qualifiers which leave the function signature unchanged should be ommit.

        void func( float * );

    instead of

        void func( float * const __restricted__ );

2. Type specifiers

    The complete forms should be preferred to short verisons.

        unsigned int

    instead of

        unsigned

    This is not necessary for `signed int`, `int` suffices in this case.

    One notable exceptions are function declarations with the `__global__` or `__device__` specifiers. In those cases even the const specifiers must be equal in declaration and definition, especially if they are templated or else the CUDA kernel launch will abort the whole program. See [this](http://stackoverflow.com/questions/35106360/why-does-this-cuda-program-crash-when-omitting-the-const-qualifier) Stackoverflow question.
