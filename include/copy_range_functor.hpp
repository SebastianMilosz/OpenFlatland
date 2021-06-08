#ifndef COPY_RANGE_FUNCTOR_HPP
#define COPY_RANGE_FUNCTOR_HPP

#include <thrust/device_vector.h>

template<typename T>
struct copy_range_functor
{
    public:
        copy_range_functor(thrust::host_vector<T>& targetVector, const uint32_t targetSize) :
            m_targetVector(targetVector),
            m_targetSize(targetSize),
            m_currentTargetPos(0U)
        {
        }

        __device__ __host__ void operator()(T value)
        {
            if (m_currentTargetPos < m_targetSize)
            {
                m_targetVector[m_currentTargetPos++] = value;
            }
        }

    private:
        thrust::host_vector<T>& m_targetVector;
        const uint32_t m_targetSize;
        uint32_t m_currentTargetPos;
};

#endif // COPY_RANGE_FUNCTOR_HPP
