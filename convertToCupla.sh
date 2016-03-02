#!/bin/bash

file=$1
if [ -z "${file%%*.hpp}" ]; then isHpp=1; else isHpp=0; fi
if [ -z "${file%%*.cpp}" ]; then isCpp=1; else isCpp=0; fi
if [ -z "${file%%*.tpp}" ]; then isTpp=1; else isTpp=0; fi
if test -z "$file" || ( [ $isHpp -eq 0 ] && [ $isCpp -eq 0 ] && [ $isTpp -eq 0 ] ) || test ! -f "$file"; then
    echo "Please specify an existing .hpp or .cpp file to convert!"
    exit
fi


if test -f $file.old; then
    trash $file.old
fi
cp $file $file.old
if [ -f $file.old ]; then
    echo "Created backup in $file.old"
else
    echo "Couldn't create backup in $file.old. Does the file exist alread with not enough permissions?"
    exit
fi
echo ""

# Note: nothing here will work for spaghetty code, because of the missing 'g'
#       operator for sed

# replace cuda.h with cuda_to_cupla.h (leaving all spaces untouched)
sed -r -i 's|^([^/]*)#([ \t]*)include([ \t]*)<([ \t]*)cuda.*\.h([ \t]*)>|\1#\2include\3<\4cuda_to_cupla.hpp\5>|' $file

# cufft.h -> cufft_to_cupla.hpp (if not commented out)
#sed -r -i 's|^([^[/]]*)(#[ \t]*include[ \t]*<[ \t]*cufft\.h[ \t]*>)|//\1\2|' $file
sed -r -i 's|^([^/]*)#([ \t]*)include([ \t]*)<([ \t]*)cufft\.h([ \t]*)>|\1#\2include\3<\4cufft_to_cupla.hpp\5>|' $file

# __device__ __host__ -> ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
sed -r -i 's|__device__[ \t]+__host__|ALPAKA_FN_NO_INLINE_ACC|' $file
sed -r -i 's|__host__[ \t]+__device__|ALPAKA_FN_NO_INLINE_ACC|' $file
sed -r -i 's|__host__[ \t]*|ALPAKA_FN_NO_INLINE_HOST|' $file

# __device__ -> ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
if grep -q '__device__' $file; then
    echo "  - Found some device only function. If you are using geometry dependent functions like blockIdx, gridDim, atomicXXX, then please add:

      template< class T_ACC >
      __device__ function
      (
          T_ACC const & acc,
          <oldArguments>
      )

    You can search for those functions by searching for 'ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY' (instead of __device__)
    And also add 'acc' as the first parameter to all those device function calls. Note that 'acc' is only available form inside kernels by default.
    Currently this is not done automatically, because a simple sed script can't remember what function calls are device functions and also if those device functions really need 'acc'. It may not add much overhead, but it may seem useless from a user perspective. Furthermore if the function is already templated, then instead of prepending a 'template <class T_ACC>' we would have to insert 'class T_ACC,' at the right position, which is hard to detect with sed.
    "
fi
sed -r -i 's|__device__|ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY|' $file

# Replace CUDA kernel calls kernelName<<<...>>> -> CUPLA_KERNEL( kernelName )(...)
# Note that this won't fully work for
#   - templated kernels: you will need to add all template arguments manually:
#     CUPLA_KERNEL(kernel) -> CUPLA_KERNEL( kernelName< float,cufftComplex > )
#   - <<< or >>> being on a separate line than 'kernelName'
#   - <<< or >>> begin used e.g. for templates. You should add spaces > > >
#     when using templates!
# '#' is allowed inside kernel names, because I had some macros which had this
# kind of usage:
#   #define macro(NAME) \
#   kernelVectorReduce##NAME<<< ... >>>(...)
foundKernels=0
if grep -q '<<<' $file; then
    echo "  - Found some kernel calls which were replaced with CUPLA_KERNEL( kernelName )( ... ). Please note, that in the case of templated kernels you will have to add the kernel parameters manually in order to give a data type to CUPLA_KERNEL, i.e. CUPLA_KERNEL( kernelName<T,P> )( ... )"
    echo ""
    foundKernels=1
fi
sed -r -i 's|([A-Za-z]+[A-Za-z0-9_#]*)[ \t]*<<<|CUPLA_KERNEL( \1 )(|; s|>>>|)|' $file
if test "$foundKernels" -eq 1; then
    grep --color -n 'CUPLA_KERNEL' $file
fi

if grep -q '__global__' $file; then
    echo "  - Some __global__ kernel functions were found, please note that if those were templated you will see something like:

      template< class T_PREC, class T_FUNC >
      template< class T_ACC >
      ALPAKA_FN_NO_INLINE_ACC void kernelName
      ::template operator()
      (
          T_ACC const & acc,
          <oldArguments>
      ) const
      {
        ...
      }

    If the kernel function is also declared in the header, then you simply need to add the template parameters to the struct name in line 3

      ALPAKA_FN_NO_INLINE_ACC void kernelName< T_PREC, T_FUNC >

    If it was not declared, then you need to change it like follows:

      template< class T_PREC, class T_FUNC >
      struct kernelName
      {
          template< class T_ACC >
          ALPAKA_FN_NO_INLINE_ACC
          void operator()
          (
              T_ACC const & acc,
              <oldArguments>
          ) const
          {
            ...
          }
      };

    Currently this is hard to detect and execute automatically.
"
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
    #        ALPAKA_FN_NO_INLINE_ACC
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
\1    ALPAKA_FN_NO_INLINE_ACC\
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
    #  +    ALPAKA_FN_NO_INLINE_ACC void cudaKernelCopyFromRealPart<T_PREC, T_COMPLEX>
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
# ALPAKA_FN_NO_INLINE_ACC void kernelName
# to
# ALPAKA_FN_NO_INLINE_ACC void kernelName<T_PREC, T_COMPLEX>
# Furthermore this rules wronlgy inserts template< class T_ACC > in
# explicit instantiations.
/__global__/,/^[ \t]*\)[ \t]*$/{
    s/([ \t]*)__global__/\1template< class T_ACC >\
\1ALPAKA_FN_NO_INLINE_ACC/
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
    s/([ \t]*)__global__/\1ALPAKA_FN_NO_INLINE_ACC/
# replace line with ( by the operator() method header
    s|([ \t]*)\(|\1::template operator()\
\1(\
\1    T_ACC const \& acc,|
# replace ) with
#     ) const
    s/(^[ \t]*)\)/\1) const/
}' $file
    if grep -q 'T_ACC' $file; then
        echo "  - Found explicit kernel instantiations which have to make use of a 'T_ACC' template parameter. In order for this to work you will have to enclose all instantiations making use of this with:

    #include "libs/alpaka_T_ACC.hpp"
    /* explicit template instantiations */
    #undef T_ACC
"
    fi
fi

#Test Pattern:
#    __shared__ T_PREC   smReduced ;
#    __shared__ long int smReduced;
#    __shared__ T_PREC   smReduced ; // wugen; wegionwe
#    __shared__ T_PREC   smReduced[23];
#    __shared__ unsigned int smReduced[a wef w*e ];//
#    extern __shared__ __align__( sizeof(T_PREC) ) unsigned char dynamicSharedMemory[];
#    extern __shared__ __align__( sizeof(T_PREC) ) unsigned char * dynamicSharedMemory;
# simple data type version
typeAndName='([\t _[:alnum:]*]*)[\t ]+([_[:alnum:]]+)'
sed -r -i 's|__shared__[\t ]+'"$typeAndName"'[\t ]*;([ \t]*(//.*)?)$|sharedMem( \2, \1 );\3|' $file
# array version
sed -r -i 's|__shared__[\t ]+'"$typeAndName"'[\t ]*\[(.*)\][\t ]*;([ \t]*(//.*)?)$|sharedMem( \2, cupla::Array< \1, \3 > );\4|' $file
# extern shared memory declaration versions
sed -r -i 's|extern[\t ]+__shared__[\t ]+(__align__[\t]*\(.*\))?'"$typeAndName"'[\t ]*;([ \t]*(//.*)?)$|sharedMemExtern( \3, \2 );\4|' $file
sed -r -i 's|extern[\t ]+__shared__[\t ]+(__align__[\t]*\(.*\))?'"$typeAndName"'[\t ]*\[(.*)\][\t ]*;([ \t]*(//.*)?)$|sharedMemExtern( \3, \2 * );\4|' $file
#Result:
#    shared( smReduced, T_PREC   ) ;
#    shared( smReduced, long int );
#    shared( smReduced, T_PREC   ) ; // wugen; wegionwe
#    shared( smReduced, cupla::Array< T_PREC  , 23 > );
#    shared( smReduced, cupla::Array< unsigned int, a wef w*e  > );//

searchString='warpSize|__shfl_down'
if grep -q -E "$searchString" $file; then
    echo "  - Found a problematic keyword which can't be mapped to alpaka, please review the following lines provided by grep:"
    grep --color -n -E "$searchString" $file
    echo ""
fi

if grep -q 'atomic[A-Z][a-z]+' $file; then
    echo "  - You are making use of alpaka atomic function, please note that calls to those are much more strict than normal atomic function calls. I.e. because of templatisation they don't allow implicit type conversions. All arguments need to be converted to the same type manually."
    echo ""
fi
