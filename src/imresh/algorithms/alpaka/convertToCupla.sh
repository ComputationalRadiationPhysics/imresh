#!/bin/bash

file=$1
if [ -z "${file%%*.hpp}" ]; then isHpp=1; else isHpp=0; fi
if [ -z "${file%%*.cpp}" ]; then isCpp=1; else isCpp=0; fi
if [ -z "${file%%*.tpp}" ]; then isTpp=1; else isTpp=0; fi
if test -z "$file" || ( [ $isHpp -eq 0 ] && [ $isCpp -eq 0 ] && [ $isTpp -eq 0 ] ) || test ! -f "$file"; then
    echo "Please specify an existing .hpp or .cpp file to convert!"
    exit
fi


trash -f $file.old
cp $file $file.old
if [ -f $file.old ]; then
    echo "Created backup in $file.old"
else
    echo "Couldn't create backup in $file.old. Does the file exist alread with not enough permissions?"
    exit
fi

# Note: nothing here will work for spaghetty code, because of the missing 'g'
#       operator for sed

# replace cuda.h with cuda_to_cupla.h (leaving all spaces untouched)
sed -r -i 's|^([^/]*)#([ \t]*)include([ \t]*)<([ \t]*)cuda.*\.h([ \t]*)>|\1#\2include\3<\4cuda_to_cupla.hpp\5>|' $file

# cufft.h -> cufft_to_cupla.hpp (if not commented out)
#sed -r -i 's|^([^[/]]*)(#[ \t]*include[ \t]*<[ \t]*cufft\.h[ \t]*>)|//\1\2|' $file
sed -r -i 's|^([^/]*)#([ \t]*)include([ \t]*)<([ \t]*)cufft\.h([ \t]*)>|\1#\2include\3<\4cufft_to_cupla.hpp\5>|' $file

# __device__ __host__ -> ALPAKA_FN_ACC_CUDA_ONLY
sed -r -i 's|__device__[ \t]+__host__|ALPAKA_FN_ACC_CUDA_ONLY|' $file
sed -r -i 's|__host__[ \t]+__device__|ALPAKA_FN_ACC_CUDA_ONLY|' $file
sed -r -i 's|__host__[ \t]*||' $file

# __device__ -> ALPAKA_FN_ACC_CUDA_ONLY
sed -r -i 's|__device__|ALPAKA_FN_ACC_CUDA_ONLY|' $file

# Replace CUDA kernel calls kernelName<<<...>>> -> CUPLA_KERNEL( kernelName )(...)
# Note that this won't fully work for
#   - templated kernels: you will need to add all template arguments manually:
#     CUPLA_KERNEL(kernel) -> CUPLA_KERNEL( kernelName< float,cufftComplex > )
#   - <<< or >>> being on a separate line than 'kernelName'
#   - <<< or >>> begin used e.g. for templates. You should add spaces > > >
#     when using templates!
sed -r -i 's|([A-Za-z]+[A-Za-z0-9_]*)[ \t]*<<<|CUPLA_KERNEL( \1 )(|; s|>>>|)|' $file
if grep -q '<<<' $file; then
    echo "Found some kernel calls which were replaced with CUPLA_KERNEL( kernelName )( ... ). Please note, that in the case of templated kernels you will have to add the kernel parameters manually in order to give a data type to CUPLA_KERNEL, i.e. CUPLA_KERNEL( kernelName<T,P> )( ... )"
fi


if [ $isHpp -eq 1 ]; then
    #    template<class T_PREC, class T_FUNC>
    #    __global__ void kernelVectorReduce
    #    (
    #        T_PREC const * const __restrict__ rdpData,
    #        unsigned int const rnData,
    #        T_PREC * const __restrict__ rdpResult,
    #        T_FUNC f,
    #        T_PREC const rInitValue
    #    );
    #
    #               vvvvvvvvvvvvvvvv
    #
    #    template< class T_COMPLEX, class T_PREC >
    #    struct cudaKernelApplyHioDomainConstraints
    #    {
    #        template< typename T_ACC >
    #        ALPAKA_FN_ACC
    #        void operator()
    #        (
    #            T_ACC const & acc,
    #            T_COMPLEX       * const rdpgPrevious,
    #            T_COMPLEX const * const rdpgPrime,
    #            T_PREC    const * const rdpIsMasked,
    #            unsigned int const rnElements,
    #            T_PREC const rHioBeta
    #        ) const;
    #    };

    sed -r -i '#
# execute the block {} for all lines in the addrees range between __global__
# and ); Please note that this only works, if ( and ); containing the parameter
# list are on a single line. If there are other parantheses alone on a line,
# then the sed rule will confuse those! To avoid that write e.g.
# __attribute__(...) on a single line.
/__global__/,/^[ \t]*\);[ \t]*$/{
    s/__global__ void/struct/
# indent all lines exclusive between ( and );
    /^[ \t]*\(/,/^[ \t]*\);/{
# do not indent line with (, because it will be replaced by { in next command
        /[()]+/!s/^/    /
    }
# replace line with ( by the operator() method header. \ continues line, but
# the line break will be printed, which is what we want.
# Without escaping & would print the matched address, meaning (
    s|([ \t]*)\(|\1{\
\1    template< typename T_ACC >\
\1    ALPAKA_FN_ACC\
\1    void operator()\
\1    (\
\1        T_ACC const \& acc,|
# replace ); with
#     );
# }
    s/(^[ \t]*)\);/\1    ) const;\n\1};/
}' $file
fi

if [ $isTpp -eq 1 ]; then
    #       template< class T_PREC, class T_COMPLEX >
    #  -    __global__ void cudaKernelCopyFromRealPart
    #  +    template< class T_ACC >
    #  +    ALPAKA_FN_ACC void cudaKernelCopyFromRealPart<T_PREC, T_COMPLEX>
    #  +    ::template operator()
    #       (
    #  +        T_ACC const & acc,
    #           T_PREC    * const rTargetComplexArray,
    #           T_COMPLEX * const rSourceRealArray,
    #           unsigned int const rnElements
    #  -    )
    #  +    ) const
    #       {

    sed -r -i '#
# execute the block {} for all lines in the addrees range between __global__
# and {
# Note that if the struct is templated, you will need to add those template
# parameters manually, e.g.
# ALPAKA_FN_ACC void kernelName
# to
# ALPAKA_FN_ACC void kernelName<T_PREC, T_COMPLEX>
# Furthermore this rules wronlgy inserts template< class T_ACC > in
# explicit instantiations.
/__global__/,/^[ \t]*\)[ \t]*$/{
    s/([ \t]*)__global__/\1template< class T_ACC >\
\1ALPAKA_FN_ACC/
# replace line with ( by the operator() method header
    s|([ \t]*)\(|\1::template operator()\
\1(\
\1    T_ACC const \& acc,|
# replace ) with
#     ) const
    s/(^[ \t]*)\)/\1) const/
}' $file
fi

if [ $isCpp -eq 1 ]; then
    sed -r -i '#
/__global__/,/^[ \t]*\);[ \t]*$/{
    s/([ \t]*)__global__/\1ALPAKA_FN_ACC/
# replace line with ( by the operator() method header
    s|([ \t]*)\(|\1::template operator()\
\1(\
\1    T_ACC const \& acc,|
# replace ) with
#     ) const
    s/(^[ \t]*)\)/\1) const/
}' $file
fi
