#pragma once
#include<string>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include<vector_functions.hpp>
#include<tuple>
#include<utility>
#include<array>
#include<algorithm>
#define M_PI       3.14159265358979323846   // pi
#define SAFE_DELETE(x) if (x) {delete x; x = nullptr;}
#define SAFE_DELETE_ARRAY(x) if (x) {delete[] x; x = nullptr;}


template<class T1,class T2, typename Function>
__global__ void device_customFunctor(
    const T1* __restrict__ in,
    T2* __restrict__ out,
   const Function f,
    const int stride)
{
    const int gid = (threadIdx.x + blockDim.x * blockIdx.x) * stride + (stride - 1);
    const int bid = gid / stride;

    out[bid] = f(in[gid]);
}
template<class T, typename Function>
__global__ void device_1arrayFunctor(
	const  T* __restrict__ in,
	T* __restrict__ out,
	const Function f)
{
	const int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	out[gid] = f(in);
}

template<class T, typename Function>
__global__ void device_2arrayFunctor(
	const  T* __restrict__ in1,
	const  T* __restrict__ in2,
	T* __restrict__ out,
	const Function f)
{
	const int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	out[gid] = f(in1, in2);
}
template<class T,class T1, class T2, class T3, typename Function>
//out,in1,in2,in3
__global__ void device_3arrayFunctor(
	const T1* __restrict__ in1,
	const T2* __restrict__ in2,
	const T3* __restrict__ in3,
	T* __restrict__ out,
	const Function f)
{
	const int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	out[gid] = f(in1, in2, in3);
}

template<class T>
__global__ void device_XZplane(
	const T* __restrict__ origin, T* __restrict__ out,
	const int current, const int x, const int xy)//, const int z)
{
	out[threadIdx.x + blockIdx.x * blockDim.x] = origin[current + threadIdx.x * x + blockIdx.x * xy];
}

template<class T>
__global__ void device_YZplane(
	const T* __restrict__ origin, T* __restrict__ out,
	const int current, const int x, const int xy)//, const int z)
{
	out[threadIdx.x + blockIdx.x * blockDim.x] = origin[threadIdx.x + x * current + blockIdx.x * xy];
}
template<class T>
__global__ void device_XZplane_rev(
	const T* __restrict__ origin, T* __restrict__ out,
	const int current, const int x, const int xy)//, const int z)
{
	out[current + threadIdx.x * x + blockIdx.x * xy] = origin[threadIdx.x + blockIdx.x * blockDim.x];
}

template<class T>
__global__ void device_YZplane_rev(
	const T* __restrict__ origin, T* __restrict__ out,
	const int current, const int x, const int xy)//, const int z)
{
	out[threadIdx.x + x * current + blockIdx.x * xy] = origin[threadIdx.x + blockIdx.x * blockDim.x];
}

#pragma region transpose
template<class T>
__global__ void device_transposeXY(const T* input, T* output, const int shape_x, const int shape_y, const int shape_z)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		output[z * shape_x * shape_y + x * shape_y + y] = input[z * shape_x * shape_y + y * shape_x + x];
	}
}

__global__ void device_transposeXZ(const char* input, char* output, const int shape_x, const int shape_y, const int shape_z)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		output[x * shape_z * shape_y + y * shape_z + z] = input[z * shape_x * shape_y + y * shape_x + x];
	}

}

__global__ void device_transposeYZ(const char* input, char* output, const int shape_x, const int shape_y, const int shape_z)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		output[y * shape_x * shape_z + z * shape_x + x] = input[z * shape_x * shape_y + y * shape_x + x];
	}
}

#pragma endregion

template<class T>
__global__ void device_padded(const T* in, float* out, const int windowSize, const int depth)
{
	const int x = threadIdx.x;
	const int y = blockIdx.x;
	const int z = blockIdx.y;
	int z1 = z - windowSize;
	if (z1 < 0) z1 = 0;
	else if (z1 >= depth) z1 = depth - 1;

	const int gid = x + y * blockDim.x + z * blockDim.x * gridDim.x;
	const int gid0 = x + y * blockDim.x + z1 * blockDim.x * gridDim.x;

	if(windowSize ==16)
		out[gid] = static_cast<unsigned char>(in[gid0])/255.0f;
	else
		out[gid] = static_cast<float>(in[gid0]) / 255.0f;
}
template<unsigned int blockSize, class T, typename Function>
__global__ void device_reduction(const T* __restrict__ in, T* __restrict__ out, const Function f)
{
	extern __shared__ __align__(sizeof(T)) unsigned char sdata[];
	T* sdata_ = reinterpret_cast<T*>(sdata);
	const unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	const unsigned int gridSize = blockSize * 2 * gridDim.x;

	sdata_[tid] = in[tid + blockIdx.x * blockDim.x];
	while (i < blockDim.x * gridDim.x)
	{
		sdata_[tid] = f(sdata_[tid], f(in[i], in[i + blockSize]));
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata_[tid] = f(sdata_[tid], sdata_[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata_[tid] = f(sdata_[tid], sdata_[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata_[tid] = f(sdata_[tid], sdata_[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata_[tid] = f(sdata_[tid], sdata_[tid + 64]); } __syncthreads(); }
	if (tid < 32)
	{
		//warpReduce(sdata_, g_idata, tid);
		if (blockSize >= 64) sdata_[tid] = f(sdata_[tid], sdata_[tid + 32]);
		if (blockSize >= 32) sdata_[tid] = f(sdata_[tid], sdata_[tid + 16]);
		if (blockSize >= 16) sdata_[tid] = f(sdata_[tid], sdata_[tid + 8]);
		if (blockSize >= 8)  sdata_[tid] = f(sdata_[tid], sdata_[tid + 4]);
		if (blockSize >= 4)  sdata_[tid] = f(sdata_[tid], sdata_[tid + 2]);
		if (blockSize >= 2)  sdata_[tid] = f(sdata_[tid], sdata_[tid + 1]);
	}
	if (tid == 0) out[blockIdx.x] = sdata_[0];
}

//union datIdx
//{
//	char data, index;
//	short a;
//};
#define _max(a,b) (a)>(b)?a:b
#define _min(a,b) (a)<(b)?a:b
__global__ void device_verticThreshold(const char* __restrict__ in, int2* out)
{
	extern __shared__ int sd[];

	//tmp var
	int id = threadIdx.x;
	
	const int warpSize = 32;
	const int warpId = id / warpSize;
	//const int laneId = id % warpSize;
	const int dimx = 1 << 9;
	int height = (1 << 9) - 1;
	
	const int min = dimx;
	const int max = -1;

	sd[id] = min;
	sd[id + dimx] = max;
	//__syncthreads();
	//for (int h = 0; h < height; h++)
	while(height--)
	{
		if (in[id + dimx * height])
		{
			sd[id] = id;
			sd[id + dimx] = id;
			break;
		}
	}
	__syncthreads();
	// sdata[id + dimx] = sdata[id];
	// __syncthreads();
	
	for (int mask = warpSize / 2; mask > 0; mask >>= 1)
	{
		{
			const int val = __shfl_xor_sync(0xffffffff, sd[id], mask);
			sd[id] = _min(sd[id], val);
		}
		{
			const int val = __shfl_xor_sync(0xffffffff, sd[id + dimx], mask);
			sd[id + dimx] = _max(sd[id + dimx], val);
		}
		//const int val2 = __shfl_xor_sync(0xffffffff, sd[id + dimx], mask);

		//sd[id+dimx] = _max(sd[id + dimx],__shfl_xor_sync(0xffffffff, sd[id + dimx], mask));
	}
	__syncthreads();
	
	if (warpId == 0 && id == 0)
	{
		for (int wid = warpSize; wid < dimx; wid += warpSize)
		{
			sd[0] = _min(sd[0], sd[wid]);
			sd[dimx] = _max(sd[dimx], sd[wid + dimx]);
		}
		__syncthreads();
		out[0].x = sd[0];
		out[0].y = sd[dimx];
	}

	//T* sdata_ = reinterpret_cast<T*>(sdata);

}
template<class T, class R, typename Function>
void reduceFunctor(const T* data, R& result, const Function f, const int& num, const cudaStream_t& stream = 0)
{
	constexpr int threads = 1 << 9;
	const int blocks = num / threads;
	const size_t sharedMemSize = fmaxf(threads, blocks) * sizeof(T);

	T* reduce;
	
	gpuErrChk(cudaMalloc((void**)&reduce, sizeof(T) * num));
	gpuErrChk(cudaMemsetAsync(reduce, 0, sizeof(T) * num, stream));

	device_reduction<threads, T> << <blocks, threads, sharedMemSize, stream >> > (data, reduce, f);
	device_reduction<1, T> << <1, blocks, sharedMemSize, stream >> > (reduce, reduce, f);

	gpuErrChk(cudaMemcpyAsync(&result, &reduce[0], sizeof(T), cudaMemcpyDeviceToHost, stream));

	cudaStreamSynchronize(stream);

	gpuErrChk(cudaFree(reduce));
}

template<class T>
void isModified(const T* prev, const T* newer,  const int& num, bool& cond, const cudaStream_t& stream = 0)
{
	constexpr int width = 1 << 9;
	const int height = num / width;
	//const size_t sharedMemSize = fmaxf(width, height) * sizeof(T);

	//T* result;
	T* buffer;
	//T* reduce;

	gpuErrChk(cudaMalloc((void**)&buffer, sizeof(T) * num));
	//gpuErrChk(cudaMalloc((void**)&reduce, sizeof(T) * num));
	//gpuErrChk(cudaMemsetAsync(reduce, 0, sizeof(T) * num, stream));
	//gpuErrChk(cudaMallocHost((void**)&result, sizeof(T)));
	
	auto isModif = [] __device__(const T* in1, const T* in2)
	{
		const int gid = threadIdx.x + blockIdx.x * blockDim.x;
		return (in1[gid] ^ in2[gid]);
	};
	auto _or = [=] __device__(const T & a, const T & b) { return a | b; };
	device_2arrayFunctor<T> << <height, width, 0, stream >> > (prev, newer, buffer, isModif);

	reduceFunctor<T, bool>(buffer, cond, _or, num, stream);


	//device_reduction<width, T> << <height, width, sharedMemSize, stream >> > (buffer, reduce, _or);
	//device_reduction<1, T> << <1, height, sharedMemSize, stream >> > (reduce, reduce, _or);

	//gpuErrChk(cudaMemcpyAsync(&cond, &reduce[0], sizeof(T), cudaMemcpyDeviceToHost, stream));


	//rese = *result;
	//cudaStreamSynchronize(stream);

	gpuErrChk(cudaFree(buffer));
	//gpuErrChk(cudaFree(reduce));

	//gpuErrChk(cudaFreeHost(result));

	//return res;
}



const int THREADS_PER_BLOCK = 512;
const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 6
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

template<class T>
__global__ void device_prescan_large(T* output, T* input, int n, T* sums) {

	extern __shared__ __align__(sizeof(T)) unsigned char sdata[];
	T* temp = reinterpret_cast<T*>(sdata);

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			//temp[bi] = f(temp[bi] , temp[ai] );
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T _t = temp[ai];
			temp[ai] = temp[bi];
			//temp[bi] = f(temp[bi], _t);// += _t;
			temp[bi] += _t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

//template<class T, typename Function>
template<class T>
__global__ void device_prescan_arbitrary(T* output, T* input, int n, int powerOfTwo)
{
	extern __shared__ __align__(sizeof(T)) unsigned char sdata[];
	T* temp = reinterpret_cast<T*>(sdata);
	//extern __shared__ T temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			//temp[bi] = f(temp[bi], temp[ai]);// += temp[ai];
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T _t = temp[ai];
			temp[ai] = temp[bi];
			//temp[bi] = f(temp[bi], _t);// += _t;
			temp[bi] += _t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

template<class T>
__global__ void device_add(T* output, int length, T* n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

template<class T>
__global__ void device_add(T* output, int length, T* n1, T* n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

template<class T>
void scanLargeDeviceArray(T* d_out, T* d_in, int length, const cudaStream_t& stream = 0);
template<class T>
void scanSmallDeviceArray(T* d_out, T* d_in, int length, const cudaStream_t& stream = 0);

inline int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}
template<class T>
void scanLargeEvenDeviceArray(T* d_out, T* d_in, int length, const cudaStream_t& stream = 0)
{
	const int blocks = std::fmaxf(1, length / ELEMENTS_PER_BLOCK);
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(T);

	T* d_sums, * d_incr;
	cudaMalloc((void**)&d_sums, blocks * sizeof(T));
	cudaMalloc((void**)&d_incr, blocks * sizeof(T));
	
	device_prescan_large<T> << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize, stream >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	//cudaDeviceSynchronize();
	cudaStreamSynchronize(stream);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, stream);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, stream);
	}

	device_add<T> << <blocks, ELEMENTS_PER_BLOCK, 0, stream >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}
template<class T>
void scanSmallDeviceArray(T* d_out, T* d_in, int length, const cudaStream_t& stream) {
	int powerOfTwo = nextPowerOfTwo(length);
	device_prescan_arbitrary<T> << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(T), stream >> > (d_out, d_in, length, powerOfTwo);
	//cudaDeviceSynchronize();
	cudaStreamSynchronize(stream);
}

template<class T>
void scanLargeDeviceArray(T* d_out, T* d_in, int length, const cudaStream_t& stream)
{
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0)
	{
		scanLargeEvenDeviceArray(d_out, d_in, length,stream);
	}
	else
	{
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, stream);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		T* startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, stream);

		device_add<T> << <1, remainder,0,stream >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

template <class T, bool isBackward>
__global__ void compactData(T* d_out,
	size_t* d_numValidElements,
	const unsigned int* d_indices, // Exclusive Sum-Scan Result
	const unsigned int* d_isValid,
	const T* d_in,
	unsigned int       numElements)
{
	if (threadIdx.x == 0)
	{
		if (isBackward)
			d_numValidElements[0] = d_isValid[0] + d_indices[0];
		else
			d_numValidElements[0] = d_isValid[numElements - 1] + d_indices[numElements - 1];
	}

	// The index of the first element (in a set of eight) that this
	// thread is going to set the flag for. We left shift
	// blockDim.x by 3 since (multiply by 8) since each block of 
	// threads processes eight times the number of threads in that
	// block
	unsigned int iGlobal = blockIdx.x * (blockDim.x << 3) + threadIdx.x;

	// Repeat the following 8 (SCAN_ELTS_PER_THREAD) times
	// 1. Check if data in input array d_in is null
	// 2. If yes do nothing
	// 3. If not write data to output data array d_out in
	//    the position specified by d_isValid
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
	iGlobal += blockDim.x;
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
	iGlobal += blockDim.x;
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
	iGlobal += blockDim.x;
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
	iGlobal += blockDim.x;
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
	iGlobal += blockDim.x;
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
	iGlobal += blockDim.x;
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
	iGlobal += blockDim.x;
	if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
		d_out[d_indices[iGlobal]] = d_in[iGlobal];
	}
}

__inline__ __device__ __host__ bool isLeft(
	const float2& s,
	const float2& e,
	const float2& t)
{
	return ((e.x - s.x) * (t.y - s.y) - (t.x - s.x) * (e.y - s.y)) > 0;
}


template<class T, class U>
inline __host__ __device__ float2 operator+(const T& a, const U& b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
template<class T, class U>
inline __host__ __device__ float2 operator-(const T& a, const U& b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ bool operator!=(const float2& a, const float2& b)
{
	return !((a.x == b.x) && (a.y == b.y));
}


template<class T>
inline __host__ __device__ float2 operator*(const T& a, const float& b)
{
	return make_float2(a.x * b, a.y * b);
}
template<class T>
inline __host__ __device__ float2 operator/(const T& a, const float& b)
{
	return make_float2(a.x / b, a.y / b);
}
template<class T>
inline __host__ __device__ float2 operator*(const float& b, const T& a)
{
	return make_float2(a.x * b, a.y * b);
}


template<class T, class U>
inline __host__ __device__ float dot(const T& a, const U& b)
{
	return a.x * b.x + a.y * b.y;
}
template<class T, class U>
inline __host__ __device__ float cross(const T&a, const U&b)
{
	return a.x * b.y - a.y * b.x;
}

__inline__ __device__ float getDist(
	const ushort2& s,
	const ushort2& e,
	const ushort2& t)
{
	//auto dot = [=](const ushort2& a, const ushort2& b)->float {return a.x * b.x + a.y * b.y; };
	
	auto length_inv = [&](const auto& v)->float {return rsqrtf(dot(v, v)); };
	auto dir = [&](const ushort2& a, const ushort2& b)->float2 {const auto d = b - a; return d * length_inv(d); };

	const auto v = t - s;
	const float2 d = dir(s, e);
	const auto T = dot(v, d);
	const auto p = s + T * d;
	const auto tmp = t - p;
	return sqrtf(dot(tmp, tmp));

}

__device__ void findhull(float2** const ary, const int size, const float2& p, const float2& q, float2* _1, int& _Size)
{
	auto len = [](const float2& p)->float {return sqrtf(dot(p, p)); };
	auto conv2uint = [](const float2& pos)->unsigned int
	{
		return pos.x + pos.y * (1 << 9);
	};

	auto getDist = [&](const float2& t) ->float
	{
		const float2 d = (q - p);
		const float T = cross(d, (t - p));
		return abs(T) / len(d);
	};
	
	int _s1size = 0;
	int _s2size = 0;

	int ind = -1;
	float xx = -FLT_MAX;
	for (int i = 0; i < size; i++)
	{
		float _xx = getDist(*ary[i]);
		if (_xx > xx)
		{
			xx = _xx;
			ind = i;
		}
	}

	float2** buffer1 = (float2**)malloc(sizeof(float2*) * size);
	float2** buffer2 = (float2**)malloc(sizeof(float2*) * size);

	float2 _r = *ary[ind];

	for (int i = 0; i < size; i++)
	{
		if (i == ind)
			continue;
		if (isLeft(p, _r, *ary[i]))
			buffer1[_s1size++] = ary[i];
		else if (isLeft(_r, q, *ary[i]))
			buffer2[_s2size++] = ary[i];
	}

	float2** _s1 = (float2**)malloc(sizeof(float2*) * _s1size);
	float2** _s2 = (float2**)malloc(sizeof(float2*) * _s2size);
	memcpy(_s1, buffer1, sizeof(float2*) * _s1size);
	memcpy(_s2, buffer2, sizeof(float2*) * _s2size);

	free(buffer1);
	free(buffer2);
	
	if (_s1size == 0)
	{	
		//const auto result = conv2uint(_r);
		if(_r.x != 0 && _r.y != 0)
		if (_1[_Size] != _r)
			_1[_Size++] = _r;
	}
	else
	{
		findhull(_s1, _s1size, p, _r, _1, _Size);
	}
	free(_s1);
	
	if (_s2size == 0)
	{
		//const auto result = conv2uint(q);
		if (_r.x != 0 && _r.y != 0)
		if (_1[_Size] != q)
			_1[_Size++] = q;
	}
	else
	{
		findhull(_s2, _s2size, _r, q, _1, _Size);
	}
	free(_s2);
};


__global__ void quickHull2(
	const unsigned int* __restrict__ in,
	float2* out,
	int* outnum, 
	const int2* __restrict__ initx, const size_t* __restrict__ num, const uint3 dim)
{
	const int d = threadIdx.x + blockIdx.x * blockDim.x;
	if (d < dim.z)
	{

		const int Size = num[d];

		if (Size > 0)
		{
			const unsigned int* _0 = in + dim.x * dim.y * d;
			float2* _1 = out + dim.x * dim.y * d;

			float2 _s{ float(initx[d].y),float(dim.y) };
			float2 _e{ float(initx[d].x),-1.0f };

			float2* S0 = (float2*)malloc(sizeof(float2) * Size);

			int s1size = 0;
			int s2size = 0;

			auto conv2float2 = [&](const unsigned int& pos)->float2
			{
				return make_float2(pos % dim.x, pos / dim.x);
			};

			int index = 0;

			for (int i = 0; i < Size; i++)
			{
				S0[i] = conv2float2(_0[i]);
				if (S0[i].x == _s.x)
					_s.y = _min(_s.y, S0[i].y);
				else if (S0[i].x == _e.x)
					_e.y = _max(_e.y, S0[i].y);
			}
			float2** buffer1 = (float2**)malloc(sizeof(float2*) * Size);
			float2** buffer2 = (float2**)malloc(sizeof(float2*) * Size);

			for (int i = 0; i < Size; i++)
			{
				if (isLeft(_s, _e, S0[i]))
					buffer1[s1size++] = &S0[i];
				else
					buffer2[s2size++] = &S0[i];
			}

			float2** S1 = (float2**)malloc(sizeof(float2*) * s1size);
			float2** S2 = (float2**)malloc(sizeof(float2*) * s2size);
			memcpy(S1, buffer1, sizeof(float2*) * s1size);
			memcpy(S2, buffer2, sizeof(float2*) * s2size);

			free(buffer1);
			free(buffer2);

			if (s1size != 0)
			{
				findhull(S1, s1size, _s, _e, _1, outnum[d]);
			}
			free(S1);

			if (s2size != 0)
			{
				findhull(S2, s2size, _e, _s, _1, outnum[d]);

			}
			free(S2);
			free(S0);
		}
	}
}

__global__ void drawContour(char* __restrict__ inout, const float2* __restrict__ hullPoint, const int* hullNumber, const uint3 dim, const float thickness = 1, const bool cw = true)
{
	const int hid = threadIdx.x;
	const int hnum = hullNumber[blockIdx.x];
	if (hnum > 0 && hid < hnum)
	{
		if (thickness > 0)
		{
	
			const float theta = atan2f(1.0f, 0.0f) * (cw ? 1.0f : -1.0f);
			
			char* out = inout + dim.x * dim.y * blockIdx.x;
			const float2* in = hullPoint + dim.x * dim.y * blockIdx.x;

			auto conv2uint = [&](const float2& pos)->unsigned int
			{
				return lroundf(pos.x) + lroundf(pos.y) * dim.x;
			};

			auto length = [](const float2& p, const float2& q) -> float
			{
				const float2 tmp = q - p;
				return sqrtf(dot(tmp, tmp));
			};

			auto transp = [&](const float2& origin)->float2
			{
				return make_float2(cosf(theta) * origin.x - sinf(theta) * origin.y,
					sinf(theta) * origin.x + cosf(theta) * origin.y);
			};
			
			const int hid2 = (hid + 1) % hnum;
			float2 s = in[cw ? hid : hid2];
			float2 e = in[cw ? hid2 : hid];

			if (s.x != 0 && s.y != 0 && e.x != 0 && e.y != 0)
			{
				float len = length(s, e);
				const float2 dirf = (e - s) / len;
				const float2 dirb = dirf * -1;
				auto dirl = transp(dirf);
				auto dirr = dirl * -1;

				const auto deg = thickness - 1;

				const int iter = len * 10;
				const auto step = (e - s) / iter;

				for (int j = 0; j < iter; j++)
				{
					auto pivot = s + step * j;

					const auto now = conv2uint(pivot);
					if (out[now] != 1) out[now] = 1;
					float deg2 = deg;
					while (deg2 > 0)
					{
						if (deg > 0)
						{
							// fl,f,fr,l,r,bl,b,br
#pragma unroll (8)
							for (int k = 0; k < 8; k++)
							{
								float2 pivot2 = pivot;
								switch (k)
								{
								case 0: pivot2 = pivot2 + (dirf + dirl) * deg; break;
								case 1: pivot2 = pivot2 + (dirf)*deg; break;
								case 2: pivot2 = pivot2 + (dirf + dirr) * deg; break;
								case 3: pivot2 = pivot2 + (dirl)*deg; break;
								case 4: pivot2 = pivot2 + (dirl)*deg; break;
								case 5: pivot2 = pivot2 + (dirb + dirl) * deg; break;
								case 6: pivot2 = pivot2 + (dirb)*deg; break;
								case 7: pivot2 = pivot2 + (dirb + dirr) * deg; break;
								}
								const auto now2 = conv2uint(pivot2);
								if (out[now2] != 1) out[now2] = 1;
							}
						}
						deg2--;
					}
				}
			}
			//}
		}
	}
}


#pragma region GetLargestComponent
__device__ unsigned FindLabel(const int* s_buf, unsigned n)
{
	unsigned label = s_buf[n];

	while (label - 1 != n)
	{
		n = label - 1;
		label = s_buf[n];
	}

	return n;
}

__inline__ __device__ void UnionLabel(int* s_buf, unsigned a, unsigned b)
{
	bool done = false;

	do {
		a = FindLabel(s_buf, a);
		b = FindLabel(s_buf, b);
		if (a < b)
		{
			int old = atomicMin(s_buf + b, a + 1);
			done = (old == b + 1);
			b = old - 1;
		}
		else if (b < a)
		{
			int old = atomicMin(s_buf + a, b + 1);
			done = (old == a + 1);
			a = old - 1;
		}
		else
		{
			done = true;
		}

	} while (!done);
}

__global__ void device_InitLabeling(const char* image, int* labels, int shape_x, int shape_y, int shape_z)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

	//unsigned image_idx = x * shape_y * shape_z + y * shape_z + z;
	//unsigned labels_idx = x * shape_y * shape_z + y * shape_z + z;
	unsigned image_idx = z * shape_x * shape_y + y * shape_x + x;
	unsigned labels_idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		if (image[image_idx])
		{
			labels[labels_idx] = labels_idx + 1;
		}
		else
		{
			labels[labels_idx] = 0;
		}
	}
}

__global__ void device_MergeLabel(int* labels, int shape_x, int shape_y, int shape_z)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

	//unsigned labels_idx = x * shape_y * shape_z + y * shape_z + z;
	unsigned labels_idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		if (labels[labels_idx])
		{
			if (z > 0)
			{
				unsigned current_plane = labels_idx - (shape_x * shape_y);
				{
					unsigned current_row = current_plane;
					if (labels[current_row])
						UnionLabel(labels, labels_idx, current_row);
				}
			}
			{
				unsigned current_plane = labels_idx;
				if (y > 0)
				{
					unsigned current_row = current_plane - shape_x;
					if (labels[current_row])
						UnionLabel(labels, labels_idx, current_row);
				}
				{
					unsigned current_row = current_plane;
					if (x > 0 && labels[current_row - 1])
						UnionLabel(labels, labels_idx, current_row - 1);
					if (x + 1 < shape_x && labels[current_row + 1])
						UnionLabel(labels, labels_idx, current_row + 1);
				}
				if (y + 1 < shape_y)
				{
					unsigned current_row = current_plane + shape_x;
					if (labels[current_row])
						UnionLabel(labels, labels_idx, current_row);
				}
			}
			if (z + 1 < shape_z)
			{
				unsigned current_plane = labels_idx + (shape_x * shape_y);
				{
					unsigned current_row = current_plane;
					if (labels[current_row])
						UnionLabel(labels, labels_idx, current_row);
				}
			}
		}
	}
}

__global__ void device_PathCompressionLabel(int* labels, int shape_x, int shape_y, int shape_z)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

	//unsigned labels_idx = x * shape_y * shape_z + y * shape_z + z;
	unsigned labels_idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		unsigned int val = labels[labels_idx];
		if (val)
			labels[labels_idx] = FindLabel(labels, labels_idx) + 1;
	}
}

__global__ void device_ReplaceLabel(int* labels, const int* dict, int labelsSize, int dictSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < labelsSize)
	{
		if (labels[idx])
		{
			for (int i = 0; i < dictSize; i++)
			{
				if (labels[idx] == dict[i])
				{
					labels[idx] = i + 1;
					break;
				}
			}
		}
	}
}

__global__ void device_SumLabeledArray(const int* labels, const int* range, int* output, int shape_x, int shape_y, int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	//int labels_idx = x * shape_y * shape_z + y * shape_z + z;
	unsigned labels_idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		if (labels[labels_idx] >= range[0]
			&& labels[labels_idx] <= range[1])
		{
			atomicAdd(&output[labels[labels_idx] - range[0]], 1);
		}
	}
}
/// 


__global__ void device_GetLargestComponentArray(const int* labels, char* result, int target_label, int shape_x, int shape_y, int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	//int labels_idx = x * shape_y * shape_z + y * shape_z + z;
	unsigned labels_idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		if (labels[labels_idx] == target_label)
		{
			result[labels_idx] = 1;
		}
		else
		{
			result[labels_idx] = 0;
		}
	}
}
#pragma endregion

#pragma region Morphology
__device__ bool check_erosion(const char* image, const char* structure, const int shape_x, const int shape_y, const int shape_z, const int structure_radius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int scale = structure_radius * 2 + 1;

	for (int i = -structure_radius; i <= structure_radius; i++)
	{
		if (x + i < 0 || x + i >= shape_x)
			return false;
		for (int j = -structure_radius; j <= structure_radius; j++)
		{
			if (y + j < 0 || y + j >= shape_y)
				return false;
			for (int k = -structure_radius; k <= structure_radius; k++)
			{
				if (z + k < 0 || z + k >= shape_z)
					return false;
				if (structure[(structure_radius + k) * scale * scale + (structure_radius + j) * scale + (structure_radius + i)])
				{
					if (!image[(z + k) * shape_x * shape_y + (y + j) * shape_x + (x + i)])
						return false;
				}
			}
		}
	}
	return true;
}

__global__ void device_binary_erosion(const char* image, char* output, const char* structure, const int shape_x, const int shape_y, const int shape_z, const int structure_radius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		if (image[idx])
		{
			if (true == check_erosion(image, structure, shape_x, shape_y, shape_z, structure_radius))
				output[idx] = 1;
		}
	}
}

__global__ void device_binary_dilation(const char* image, char* output, const char* structure, const int shape_x, const int shape_y, const int shape_z, const int structure_radius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = z * shape_x * shape_y + y * shape_x + x;

	int scale = structure_radius * 2 + 1;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		if (image[idx])
		{
			for (int i = -structure_radius; i <= structure_radius; i++)
			{
				if (x + i < 0 || x + i >= shape_x)
					continue;
				for (int j = -structure_radius; j <= structure_radius; j++)
				{
					if (y + j < 0 || y + j >= shape_y)
						continue;
					for (int k = -structure_radius; k <= structure_radius; k++)
					{
						if (z + k < 0 || z + k >= shape_z)
							continue;
						if (structure[(structure_radius + k) * scale * scale + (structure_radius + j) * scale + (structure_radius + i)])
						{
							output[(z + k) * shape_x * shape_y + (y + j) * shape_x + (x + i)] = 1;
						}
					}
				}
			}
		}
	}
}

__global__ void device_binary_dilation2D(const char* image, char* output, const char* structure, const int shape_x, const int shape_y, const int shape_z, const int structure_radius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = z * shape_x * shape_y + y * shape_x + x;

	int scale = structure_radius * 2 + 1;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		if (image[idx])
		{
			for (int i = -structure_radius; i <= structure_radius; i++)
			{
				if (x + i < 0 || x + i >= shape_x)
					continue;
				for (int j = -structure_radius; j <= structure_radius; j++)
				{
					if (y + j < 0 || y + j >= shape_y)
						continue;
					if (structure[(structure_radius + j) * scale + (structure_radius + i)] == 1)
					{
						output[z * shape_x * shape_y + (y + j) * shape_x + (x + i)] = 1;
					}
				}
			}
		}
	}
}

#pragma endregion

#pragma region GaussianBlur
template<class T>
__global__ void device_gaussianBlur(const T* image, T* output, const float* kernel, int shape_x, int shape_y, int shape_z, int shape_kernel_x, int shape_kernel_y)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = z * shape_x * shape_y + y * shape_x + x;

	int radius_x = shape_kernel_x / 2;
	int radius_y = shape_kernel_y / 2;

	float sum = 0;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		for (int i = -radius_x; i <= radius_x; i++)
		{
			int x1 = x + i;
			if (x1 < 0)
			{
				x1 = -x1;
			}
			else if (x1 >= shape_x)
			{
				x1 = shape_x - (x1 - (shape_x - 1)) - 1;
			}
			for (int j = -radius_y; j <= radius_y; j++)
			{
				int y1 = y + j;
				if (y + j < 0)
				{
					y1 = -y1;
				}
				else if (y + j >= shape_y)
				{
					y1 = shape_y - (y1 - (shape_y - 1)) - 1;
				}
				sum += float(image[z * shape_x * shape_y + y1 * shape_x + x1]) * kernel[(j + radius_y) * shape_kernel_x + (i + radius_x)];
			}
		}
		if (sizeof(T) == 1)
		{
			unsigned char ucTemp = __float2uint_rn(sum);
			memcpy(&output[idx], &ucTemp, sizeof(char));
		}
		else
			output[idx] = sum;
	}
}
#pragma endregion

#pragma region Skeletonize
__device__ const int lut[256] = { 0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
	   0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
	   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
	   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	   0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
	   0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
	   0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	   0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	   2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
	   1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	   0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
	   0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0 };

__device__ int CheckNeighbors(const char* input, int shape_x, int shape_y, int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	return lut[input[z * shape_x * shape_y + (y - 1) * shape_x + (x - 1)] + 2 * input[z * shape_x * shape_y + (y)*shape_x + (x - 1)]
		+ 4 * input[z * shape_x * shape_y + (y + 1) * shape_x + (x - 1)] + 8 * input[z * shape_x * shape_y + (y + 1) * shape_x + (x)]
		+ 16 * input[z * shape_x * shape_y + (y + 1) * shape_x + (x + 1)] + 32 * input[z * shape_x * shape_y + (y)*shape_x + (x + 1)]
		+ 64 * input[z * shape_x * shape_y + (y - 1) * shape_x + (x + 1)] + 128 * input[z * shape_x * shape_y + (y - 1) * shape_x + (x)]];
}

__device__ int Test1(const char* input, int shape_x, int shape_y, int shape_z)
{
	int neighbors = CheckNeighbors(input, shape_x, shape_y, shape_z);

	return (neighbors == 1 || neighbors == 3);
}

__device__ int Test2(const char* input, int shape_x, int shape_y, int shape_z)
{
	int neighbors = CheckNeighbors(input, shape_x, shape_y, shape_z);

	return (neighbors == 2 || neighbors == 3);
}

__global__ void device_BlackToWhite(char* image, const char* markers, int shape_x, int shape_y, int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = z * shape_x * shape_y + y * shape_x + x;

	if (z > 0 && y > 0 && x > 0 && x < shape_x - 1 && y < shape_y - 1 && z < shape_z - 1)
	{
		if (markers[idx] == 1)
		{
			image[idx] = 0;
		}
	}
}

__global__ void device_Skeletonize2DStep1(const char* image, char* markers, int* count, int shape_x, int shape_y, int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = z * shape_x * shape_y + y * shape_x + x;

	if (z > 0 && y > 0 && x > 0 && x < shape_x - 1 && y < shape_y - 1 && z < shape_z - 1)
	{
		if (image[idx] == 1 && Test1(image, shape_x, shape_y, shape_z))
		{
			markers[idx] = 1;
			atomicAdd(count, 1);
		}
	}
}

__global__ void device_Skeletonize2DStep2(const char* image, char* markers, int* count, int shape_x, int shape_y, int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = z * shape_x * shape_y + y * shape_x + x;

	if (z > 0 && y > 0 && x > 0 && x < shape_x - 1 && y < shape_y - 1 && z < shape_z - 1)
	{
		if (image[idx] == 1 && Test2(image, shape_x, shape_y, shape_z))
		{
			markers[idx] = 1;
			atomicAdd(count, 1);
		}
	}
}
#pragma endregion

__global__ void device_sumLabel(const float* image, const unsigned char* label, float* sum, unsigned int* cnt, const int imageSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < imageSize)
	{
		atomicAdd(&sum[label[gid]], image[gid]);
		atomicAdd(&cnt[label[gid]], 1);
	}
}

__global__ void device_exp2dev(const float* image, const unsigned char* label, const float* mean, float* output, unsigned int imageSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < imageSize)
	{
		float temp = image[gid] - mean[label[gid]];
		output[gid] = temp * temp;
	}
}

__global__ void device_gaussianDist(const float* image, float* output, float* pred, const float mu, const float sigma, const float prior, const unsigned int imageSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < imageSize)
	{
		output[gid] = 1. / sqrt(2. * M_PI * (sigma * sigma)) * (expf(-0.5 * ((image[gid] - mu) * (image[gid] - mu)) / (sigma * sigma)));

		atomicAdd(&pred[gid], output[gid] * prior);
	}
}

__global__ void device_clip(const float* image, float* output, const float imageMin, const float imageMax, const unsigned int imageSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < imageSize)
	{
		if (image[gid] < imageMin)
			output[gid] = imageMin;
		else if (image[gid] > imageMax)
			output[gid] = imageMax;
		else
			output[gid] = image[gid];
	}
}

__global__ void device_post(const float* lkh, const float* pred, const float* prior, float* output, const unsigned int arrSize, const unsigned int compSize)
{
	const int arrIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int compIdx = blockIdx.y * blockDim.y + threadIdx.y;

	const int gid = compIdx * arrSize + arrIdx;

	if (arrIdx < arrSize && compIdx < compSize)
	{
		output[gid] = lkh[gid] * prior[compIdx] / pred[arrIdx];
	}
}

__global__ void device_muSum(const float* post, const float* vec, float* sumPost, float* sumPostVec, const unsigned int arrSize, const unsigned int compSize)
{
	const int arrIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int compIdx = blockIdx.y * blockDim.y + threadIdx.y;

	const int gid = compIdx * arrSize + arrIdx;

	if (arrIdx < arrSize && compIdx < compSize)
	{
		atomicAdd(&sumPost[compIdx], post[gid]);
		atomicAdd(&sumPostVec[compIdx], post[gid] * vec[arrIdx]);
	}
}

__global__ void device_sigmaSum(const float* post, const float* vec, const float* prevMu, float* sumPost, float* sumPostVec, const unsigned int arrSize, const unsigned int compSize)
{
	const int arrIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int compIdx = blockIdx.y * blockDim.y + threadIdx.y;

	const int gid = compIdx * arrSize + arrIdx;

	if (arrIdx < arrSize && compIdx < compSize)
	{
		float temp = vec[arrIdx] - prevMu[compIdx];

		atomicAdd(&sumPost[compIdx], post[gid]);
		atomicAdd(&sumPostVec[compIdx], post[gid] * temp * temp);
	}
}

__global__ void device_mseSum(const float* pred, const float* prev, float* output, const unsigned int arrSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < arrSize)
	{
		float temp = prev[gid] - pred[gid];
		atomicAdd(output, temp * temp);
	}
}

template<class T>
__global__ void device_filter(const float* image, T* output, const int minI, const int maxI, const unsigned int imageSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < imageSize)
	{
		float temp = image[gid] > maxI ? maxI : (image[gid] < minI ? minI : image[gid]);
		output[gid] = (temp - (float)minI) * 255. / (float)(maxI - minI);
	}
}



#pragma region logicaloperation

__global__ void device_logical_and(const char* inputA, const char* inputB, char* output, const int inputSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		if (inputA[gid] && inputB[gid])
			output[gid] = 1;
		else
			output[gid] = 0;
	}
}

__global__ void device_logical_or(const char* inputA, const char* inputB, char* output, const int inputSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		if (inputA[gid] || inputB[gid])
			output[gid] = 1;
		else
			output[gid] = 0;
	}
}

#pragma endregion


#pragma region canny
__global__ void device_sobel(const char* input, char* magnitude, char* angle, const int shape_x, const int shape_y, const int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int idx = z * shape_x * shape_y + y * shape_x + x;

	if (x > 0 && x < shape_x - 1 && y > 0 && y < shape_y - 1 && z > 0 && z < shape_z - 1)
	{
		int vKer = 0, hKer = 0;

		vKer = (1 * input[z * shape_x * shape_y + (y - 1) * shape_x + (x - 1)]) + (2 * input[z * shape_x * shape_y + (y - 1) * shape_x + x]) + (1 * input[z * shape_x * shape_y + (y - 1) * shape_x + (x + 1)]) +
			(-1 * input[z * shape_x * shape_y + (y + 1) * shape_x + (x - 1)]) + (-2 * input[z * shape_x * shape_y + (y + 1) * shape_x + x]) + (-1 * input[z * shape_x * shape_y + (y + 1) * shape_x + (x + 1)]);

		hKer = (1 * input[z * shape_x * shape_y + (y - 1) * shape_x + (x - 1)]) + (-1 * input[z * shape_x * shape_y + (y - 1) * shape_x + (x + 1)]) +
			(2 * input[z * shape_x * shape_y + y * shape_x + (x - 1)]) + (-2 * input[z * shape_x * shape_y + y * shape_x + (x + 1)]) +
			(1 * input[z * shape_x * shape_y + (y + 1) * shape_x + (x - 1)]) + (-1 * input[z * shape_x * shape_y + (y + 1) * shape_x + (x + 1)]);

		unsigned char chTemp = sqrtf(vKer * vKer + hKer * hKer);
		memcpy(&magnitude[idx], &chTemp, sizeof(unsigned char));
		chTemp = (unsigned char)((atan2f(vKer, hKer) + 9 / 8 * M_PI) * 4 / M_PI) & 0x3;
		memcpy(&angle[idx], &chTemp, sizeof(unsigned char));
	}
}

__global__ void device_edge_thin(const char* magnitude, const char* angle, char* output, const int shape_x, const int shape_y, const int shape_z)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int idx = z * shape_x * shape_y + y * shape_x + x;

	if (x > 0 && x < shape_x - 1 && y > 0 && y < shape_y - 1 && z > 0 && z < shape_z - 1)
	{
		int x1 = 0, x2 = 0, y1 = 0, y2 = 0;
		switch (angle[idx])
		{
		case 0:
			y1 = y2 = y;
			x1 = x - 1;
			x2 = x + 1;
			break;
		case 3:
			y1 = y - 1;
			x1 = x + 1;
			y2 = y + 1;
			x2 = x - 1;
			break;
		case 2:
			x1 = x2 = x;
			y1 = y - 1;
			y2 = y + 1;
			break;
		case 1:
			y1 = y - 1;
			x1 = x - 1;
			y2 = y + 1;
			x2 = x + 1;
		}

		unsigned char chTemp1, chTemp2, chTemp;
		memcpy(&chTemp, &magnitude[idx], sizeof(unsigned char));
		memcpy(&chTemp1, &magnitude[z * shape_x * shape_y + y1 * shape_x + x1], sizeof(unsigned char));
		memcpy(&chTemp2, &magnitude[z * shape_x * shape_y + y2 * shape_x + x2], sizeof(unsigned char));

		if (chTemp1 > chTemp || chTemp2 > chTemp)
			output[idx] = 0;
		else
			output[idx] = magnitude[idx];
	}
}

#define MSK_LOW 0x0
#define MSK_THR	0x60
#define MSK_NEW 0x90
#define MSK_DEF	0xff

__global__ void device_edge_thin(const char* input, unsigned char* output, const int shape_x, const int shape_y, const int shape_z, const char t1, const char t2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		unsigned char chTemp;
		memcpy(&chTemp, &input[idx], sizeof(unsigned char));

		int grad = chTemp;
		if (grad < t1)
			output[idx] = MSK_LOW;
		else if (grad < t2)
			output[idx] = MSK_THR;
		else
			output[idx] = MSK_NEW;
	}
}

#define CAS(buf, cond, x2, y2, z, width, height) if ((cond) && buf[z * width * height + (y2) * (width) + (x2)] == MSK_THR) { buf[z * width * height + (y2) * (width) + (x2)] = MSK_NEW; }

__global__ void device_hysteresis(unsigned char* input, const int shape_x, const int shape_y, const int shape_z, bool final)
{
	__shared__ char changes;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int idx = z * shape_x * shape_y + y * shape_x + x;

	do {
		__syncthreads();
		changes = 0;
		__syncthreads();

		if ((x < shape_x && y < shape_y && z < shape_z) && input[idx] == MSK_NEW)
		{
			input[idx] = MSK_DEF;
			changes = 1;

			CAS(input, x > 0 && y > 0, x - 1, y - 1, z, shape_x, shape_y);
			CAS(input, y > 0, x, y - 1, z, shape_x, shape_y);
			CAS(input, x < shape_x - 1 && y > 0, x + 1, y - 1, z, shape_x, shape_y);
			CAS(input, x < shape_x - 1, x + 1, y, z, shape_x, shape_y);
			CAS(input, x < shape_x - 1 && y < shape_y - 1, x + 1, y + 1, z, shape_x, shape_y);
			CAS(input, y < shape_y - 1, x, y + 1, z, shape_x, shape_y);
			CAS(input, x > 0 && y < shape_y - 1, x - 1, y + 1, z, shape_x, shape_y);
			CAS(input, x > 0, x - 1, y, z, shape_x, shape_y);
		}

		__syncthreads();
	} while (changes);

	if (final && (x < shape_x && y < shape_y && z < shape_z) && input[idx] != MSK_DEF)
		input[idx] = 0;

}
#pragma endregion


#pragma region KMeans
__global__ void device_KMeansStep1(const float* image, const float* clusters, unsigned char* outputLabel, int imageSize, int clustersSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < imageSize)
	{
		int closeClusterIdx = 0;
		float closeClusterDist = FLT_MAX;
		for (int i = 0; i < clustersSize; i++)
		{
			if (abs(clusters[i] - image[idx]) < closeClusterDist)
			{
				closeClusterIdx = i;
				closeClusterDist = abs(clusters[i] - image[idx]);
			}
		}
		outputLabel[idx] = closeClusterIdx;
	}
}

__global__ void device_KMeansStep2(const float* image, const unsigned char* label, float* outputSum, unsigned int* outputCnt, int imageSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < imageSize)
	{
		atomicAdd(&outputSum[label[idx]], image[idx]);
		atomicAdd(&outputCnt[label[idx]], 1);
	}
}

__global__ void device_Euclidean_Squared_distances(const float* input, float* output, const float center, const int inputSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < inputSize)
	{
		float temp = input[idx] - center;
		output[idx] = temp * temp;
	}
}

__global__ void device_minimum(float* inputA, float* inputB, const int inputSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < inputSize)
	{
		inputA[idx] = inputA[idx] < inputB[idx] ? inputA[idx] : inputB[idx];
	}
}
#pragma endregion

__global__ void device_findborder_tmp(const char* input, const char* border, char* output, const int inputSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		if (border[gid] == 1)
			output[gid] = 0;
		else
			output[gid] = input[gid];
	}
}

__global__ void device_la_line_tmp(const char* input, const char* mark, char* output, const int inputSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		if (mark[gid] > 0)
			output[gid] = 0;
		else
			output[gid] = input[gid];
	}
}

__global__ void device_la_line_merge(const char* inputA, const char* inputB, const char* inputC, const char* wall, char* output, const int shape_x, const int shape_y, const int shape_z)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int idx = z * shape_x * shape_y + y * shape_x + x;

	if (x < shape_x && y < shape_y && z < shape_z)
	{
		output[idx] = inputA[idx] || inputB[idx] || inputC[idx];
		if (wall[idx] != 0)
			output[idx] = 0;
		if (z == 0)
			output[idx] = 0;
	}
}

__global__ void device_sumMarked(const char* voxel, const char* vmark, int* sum, int* cnt, const int voxelSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < voxelSize)
	{
		if (vmark[gid] > 0)
		{
			unsigned char temp;
			memcpy(&temp, &voxel[gid], sizeof(char));
			atomicAdd(sum, (int)temp);
			atomicAdd(cnt, 1);
		}
	}
}

__global__ void device_sumMarked(const int* voxel, const char* vmark, unsigned long long* sum, unsigned int* cnt, const int voxelSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < voxelSize)
	{
		if (vmark[gid] > 0)
		{
			atomicAdd(sum, voxel[gid]);
			atomicAdd(cnt, 1);
		}
	}
}

__global__ void device_seperateStep1(const char* input, const char* vmark, int* output, const int mean, const int inputSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		if (vmark[gid] > 0)
		{
			unsigned char chT;
			memcpy(&chT, &input[gid], sizeof(char));
			int temp = (int)chT - mean;
			output[gid] = temp * temp;
		}
	}
}

__global__ void device_seperateStep2(char* arr, const char* voxel, float threshold, int arrSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < arrSize)
	{
		unsigned char temp;
		memcpy(&temp, &voxel[gid], sizeof(char));
		if (temp < threshold)
		{
			arr[gid] = 0;
		}
	}
}

__global__ void device_exp_left_cardiac_tmp(const char* input, const char* mark, char* output, const int inputSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		if (mark[gid] != 0)
			output[gid] = 0;
		else
			output[gid] = input[gid];
	}
}


void skeletonize(char* d_arr, int shape_x, int shape_y, int shape_z)
{
	char* d_markers;
	int* d_count;

	size_t size = shape_x * shape_y * shape_z * sizeof(char);

	gpuErrChk(cudaMalloc((void**)&d_markers, size));
	gpuErrChk(cudaMalloc((void**)&d_count, sizeof(int)));

	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	int processed = 0, count = 0;

	do {
		count = 0;
		processed = 0;
		gpuErrChk(cudaMemset(d_markers, 0, size));
		gpuErrChk(cudaMemset(d_count, 0, sizeof(int)));

		device_Skeletonize2DStep1 << <gridDim, blockDim >> > (d_arr, d_markers, d_count, shape_x, shape_y, shape_z);

		gpuErrChk(cudaDeviceSynchronize());

		gpuErrChk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

		std::cout << "first : " << count << std::endl;
		processed = (count > 0);

		if (processed)
		{
			device_BlackToWhite << <gridDim, blockDim >> > (d_arr, d_markers, shape_x, shape_y, shape_z);
			gpuErrChk(cudaDeviceSynchronize());
		}

		gpuErrChk(cudaMemset(d_markers, 0, size));
		gpuErrChk(cudaMemset(d_count, 0, sizeof(int)));

		device_Skeletonize2DStep2 << <gridDim, blockDim >> > (d_arr, d_markers, d_count, shape_x, shape_y, shape_z);

		gpuErrChk(cudaDeviceSynchronize());

		gpuErrChk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
		std::cout << "second : " << count << std::endl;
		if (processed == 0)
			processed = (count > 0);

		if (processed)
		{
			device_BlackToWhite << <gridDim, blockDim >> > (d_arr, d_markers, shape_x, shape_y, shape_z);
			gpuErrChk(cudaDeviceSynchronize());
		}
	} while (processed == 1);

	gpuErrChk(cudaFree(d_markers));
	gpuErrChk(cudaFree(d_count));
}

void binaryErosion(const char* d_input, const char* d_structure, char* d_output, const int shape_x, const int shape_y, const int shape_z, const int structure_radius, const int iterations)
{
	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	char* d_arr;
	gpuErrChk(cudaMalloc((void**)&d_arr, shape_x * shape_y * shape_z));

	gpuErrChk(cudaMemcpy(d_arr, d_input, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < iterations; i++)
	{
		gpuErrChk(cudaMemset(d_output, 0, shape_x * shape_y * shape_z));

		device_binary_erosion << <gridDim, blockDim >> > (d_arr, d_output, d_structure, shape_x, shape_y, shape_z, structure_radius);

		gpuErrChk(cudaMemcpy(d_arr, d_output, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));
	}

	gpuErrChk(cudaDeviceSynchronize());

	gpuErrChk(cudaFree(d_arr));
}

void binaryDilation(const char* d_input, const char* d_structure, char* d_output, const int shape_x, const int shape_y, const int shape_z, const int structure_radius, const int iterations)
{
	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	char* d_arr;
	gpuErrChk(cudaMalloc((void**)&d_arr, shape_x * shape_y * shape_z));

	gpuErrChk(cudaMemcpy(d_arr, d_input, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < iterations; i++)
	{
		gpuErrChk(cudaMemset(d_output, 0, shape_x * shape_y * shape_z));

		device_binary_dilation << <gridDim, blockDim >> > (d_arr, d_output, d_structure, shape_x, shape_y, shape_z, structure_radius);

		gpuErrChk(cudaMemcpy(d_arr, d_output, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));
	}

	gpuErrChk(cudaDeviceSynchronize());

	gpuErrChk(cudaFree(d_arr));
}

void binaryDilation2D(const char* d_input, const char* d_structure, char* d_output, const int shape_x, const int shape_y, const int shape_z, const int structure_radius, const int iterations)
{
	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	char* d_arr;
	gpuErrChk(cudaMalloc((void**)&d_arr, shape_x * shape_y * shape_z));

	gpuErrChk(cudaMemcpy(d_arr, d_input, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < iterations; i++)
	{
		gpuErrChk(cudaMemset(d_output, 0, shape_x * shape_y * shape_z));

		device_binary_dilation2D << <gridDim, blockDim >> > (d_arr, d_output, d_structure, shape_x, shape_y, shape_z, structure_radius);

		gpuErrChk(cudaDeviceSynchronize());

		gpuErrChk(cudaMemcpy(d_arr, d_output, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));
	}

	gpuErrChk(cudaFree(d_arr));
}




template<class T>
void extractIndex(const T* origin, unsigned int* oVal, size_t* oIdx, const Dims& dim, const cudaStream_t* const cs)
{
	constexpr int threads = 1 << 10;
	constexpr int snum = 1 << 4;
	const int pitch = dim.d[0] * dim.d[1];
	const int num = pitch * dim.d[2];
	const int blocks = num / threads;

	unsigned int* in = nullptr;
	unsigned int* out = nullptr;
	unsigned int* flag = nullptr;
	auto bin = []__device__(const auto & x) { return x > 0; };

	auto numbering = [pitch]__device__(const auto & x) {
		return (threadIdx.x + blockDim.x * blockIdx.x) % pitch;
	};

	gpuErrChk(cudaMalloc((void**)&in, sizeof(unsigned int) * num));
	gpuErrChk(cudaMalloc((void**)&out, sizeof(unsigned int) * num));
	gpuErrChk(cudaMalloc((void**)&flag, sizeof(unsigned int) * num));
	gpuErrChk(cudaMemsetAsync(out, 0, sizeof(unsigned int) * num, cs[0]));

	device_customFunctor<T, unsigned int> << <blocks, threads, 0, cs[1] >> > (origin, flag, bin, 1);
	device_customFunctor<unsigned int, unsigned int> << <blocks, threads, 0, cs[2] >> > (oVal, in, numbering, 1);

	for (int d = 0, stride = 0, si = 0;
		d < dim.d[2];
		d++, stride += pitch)
	{

		scanLargeDeviceArray(out + stride, flag + stride, pitch, cs[si]);

		compactData<unsigned int, false> << < dim.d[1], dim.d[0], 0, cs[si] >> > (oVal + stride, oIdx + d, out + stride, flag + stride, in + stride, pitch);

	}


	gpuErrChk(cudaFree(in));
	gpuErrChk(cudaFree(out));
	gpuErrChk(cudaFree(flag));
}

template<class T>
void convexFill(T* origin, /*unsigned int** const oVal, size_t** const oIdx,*/ const Dims& dim, const cudaStream_t* const cs)
{
	constexpr int snum = 1 << 4;
	const int pitch = dim.d[0] * dim.d[1];
	const int num = pitch * dim.d[2];
	//const int blocks = num / threads;

	//unsigned int* in = nullptr;
	//unsigned int* out = nullptr;
	//unsigned int* flag = nullptr;
	unsigned int* d_val = nullptr;
	size_t* d_idx = nullptr;
	int2* d_nx;

	//std::vector<cudaStream_t> cs(snum);

	const int pad = 10;

	auto padding = [pad]__device__(const T * in) -> T
	{
		const int xid = threadIdx.x;
		const int yid = blockIdx.x;
		const int gid = xid + yid * blockDim.x;

		if ((xid < pad) || (yid < pad) || (xid >= (blockDim.x - pad)) || (yid >= (gridDim.x - pad)))
			return 0;
		else
			return in[gid];
	};

	gpuErrChk(cudaMalloc((void**)&d_val, sizeof(unsigned int) * num));
	gpuErrChk(cudaMalloc((void**)&d_idx, sizeof(size_t) * dim.d[2]));
	gpuErrChk(cudaMalloc((void**)&d_nx, sizeof(int2) * dim.d[2]));


	gpuErrChk(cudaDeviceSynchronize());

	extractIndex<T>(origin, d_val, d_idx, dim, cs);

	for (int d = 0, stride = 0, si = 0;
		d < dim.d[2];
		d++, stride += pitch)
	{

		//scanLargeDeviceArray(out + stride, flag + stride, pitch,cs[si]);

		//compactData<unsigned int, false> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_val + stride, d_idx + d, out + stride, flag + stride, in + stride, pitch);

		device_verticThreshold << <dim.d[1], dim.d[0], sizeof(int2)* dim.d[0], cs[si] >> > (origin + stride, d_nx + d);

		si = (++si == snum) ? 0 : si;
	}
	gpuErrChk(cudaDeviceSynchronize());

	T* d_buffer;
	float2* d_hull;
	int* d_hullnum;

	//float2* h_hull;
	//int* h_hullnum;

	uint3 dim_ = make_uint3(dim.d[0], dim.d[1], dim.d[2]);
	gpuErrChk(cudaMalloc((void**)&d_buffer, sizeof(T) * num));
	gpuErrChk(cudaMalloc((void**)&d_hull, sizeof(float2) * num));
	gpuErrChk(cudaMalloc((void**)&d_hullnum, sizeof(int) * dim.d[2]));
	//gpuErrChk(cudaMallocHost((void**)&h_hull, sizeof(float2)* num));
	//gpuErrChk(cudaMallocHost((void**)&h_hullnum, sizeof(int)* dim.d[2]));
	gpuErrChk(cudaMemset(d_buffer, 0, sizeof(T) * num));

	size_t sz0;
	cudaDeviceGetLimit(&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, sz0 * 64);
	gLogInfo << "device stack size change to bytes of" << sz0 * 64 << '\n';

	quickHull2 << <dim.d[2] / 32 + 1, 32 >> > (d_val, d_hull, d_hullnum, d_nx, d_idx, dim_);
	gpuErrChk(cudaDeviceSynchronize());

	cudaDeviceSetLimit(cudaLimitStackSize, sz0);
	gLogInfo << "device stack size change to bytes of" << sz0 << '\n';

	//cudaMemcpy(h_hull, d_hull, sizeof(float2)* num, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_hullnum, d_hullnum, sizeof(int)* dim.d[2], cudaMemcpyDeviceToHost);
	drawContour << < dim.d[2], 128 >> > ((char*)d_buffer, d_hull, d_hullnum, dim_, 2);

	//char* h_test;
	//float2* h_hull;
	//int* h_hullnum;
	//gpuErrChk(cudaMallocHost((void**)&h_test, sizeof(char)* num));
	//gpuErrChk(cudaMallocHost((void**)&h_hull, sizeof(float2)* num));
	//gpuErrChk(cudaMallocHost((void**)&h_hullnum, sizeof(int)* dim.d[2]));

	//cudaMemcpy(h_test, origin, sizeof(char)* num, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_hull, d_hull, sizeof(float2)* num, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_hullnum, d_hullnum, sizeof(int)* dim.d[2], cudaMemcpyDeviceToHost);

	//for(int d = 0;d<dim.d[2];d++)
	//	for(int i =0;i<128;i++)
	//		drawContour (h_test, h_hull, h_hullnum,dim_, d, i, 2);

	gpuErrChk(cudaDeviceSynchronize());
	for (int d = 0, stride = 0, si = 0;
		d < dim.d[2];
		d++, si = (++si == snum) ? 0 : si, stride += pitch)
	{
		device_1arrayFunctor<T> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_buffer + stride, origin + stride, padding);
	}
	gpuErrChk(cudaDeviceSynchronize());
	/*for (int d = 0, stride = 0, si = 0;
		d < dim.d[2];
		d++, stride += pitch)
	{

		scanLargeDeviceArray(out + stride, flag + stride, pitch, cs[si]);

		compactData<unsigned int, false> << < dim.d[1], dim.d[0], 0, cs[si] >> > (*d_val + stride, *d_idx + d, out + stride, flag + stride, in + stride, pitch);

		device_verticThreshold << <dim.d[1], dim.d[0], sizeof(int2)* dim.d[0], cs[si] >> > (origin + stride, d_nx + d);

		si = (++si == snum) ? 0 : si;
	}*/

	//auto contour = []__device__

	//for (auto& _cs : cs)
	//{
	//	gpuErrChk(cudaStreamDestroy(_cs));
	//}

	gpuErrChk(cudaFree(d_nx));
	gpuErrChk(cudaFree(d_buffer));
	gpuErrChk(cudaFree(d_hull));
	gpuErrChk(cudaFree(d_hullnum));
	gpuErrChk(cudaFree(d_val));
	gpuErrChk(cudaFree(d_idx));

	
}



__global__ void device_ChkThreshold(const float* input, unsigned char* output, float lowerThd, float upperThd, int inputSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < inputSize)
	{
		if (input[idx] > lowerThd && input[idx] < upperThd)
		{
			output[idx] = 1;
		}
		else
		{
			output[idx] = 0;
		}
	}
}

__global__ void device_histogram(const float* input, const float* bin, float* output, const unsigned int inputSize, const unsigned int binSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		for (int i = 1; i < binSize; i++)
		{
			if (input[gid] >= bin[i - 1] && input[gid] < bin[i])
				output[gid] = i;
		}
	}
}

__global__ void device_piecewise(const float* input, float* output, const int threshold, const float slop0, const float slop1, const unsigned int inputSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < inputSize)
	{
		if (input[gid] < threshold)
			output[gid] = slop0 * (float)input[gid] * (float)input[gid] / threshold;
		else
			output[gid] = (slop1 * ((float)input[gid] - threshold)) + (slop0 * threshold);
	}
}


template<class T>
void gaussianBlur(const T* d_input, T* d_output, const float* d_kernel, const int shape_x, const int shape_y, const int shape_z, const int shape_kernel_x, const int shape_kernel_y)
{
	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	device_gaussianBlur<T> << <gridDim, blockDim >> > (d_input, d_output, d_kernel, shape_x, shape_y, shape_z, 5, 5);
}

float getMarkedMean(const char* d_input, const char* d_mark, const int inputSize)
{
	int* d_sum;
	int* d_cnt;

	const int tnum = 1024;

	dim3 threads(tnum);
	dim3 blocks(inputSize / tnum);

	gpuErrChk(cudaMalloc((void**)&d_sum, sizeof(int)));
	gpuErrChk(cudaMalloc((void**)&d_cnt, sizeof(int)));

	gpuErrChk(cudaMemset(d_sum, 0, sizeof(int)));
	gpuErrChk(cudaMemset(d_cnt, 0, sizeof(int)));

	device_sumMarked << <blocks, threads >> > (d_input, d_mark, d_sum, d_cnt, inputSize);

	int sum;
	int cnt;

	gpuErrChk(cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrChk(cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));

	std::cout << "sum : " << sum << std::endl;
	std::cout << "cnt : " << cnt << std::endl;

	std::cout << "mean : " << sum / cnt << std::endl;

	gpuErrChk(cudaFree(d_sum));
	gpuErrChk(cudaFree(d_cnt));

	return (float)sum / (float)cnt;
}

float getMarkedStdDevitation(const char* d_input, const char* d_mark, const int inputSize, const float _mean = FLT_MAX)
{
	const int tnum = 1024;

	dim3 threads(tnum);
	dim3 blocks(inputSize / tnum);

	float mean = _mean == FLT_MAX ? getMarkedMean(d_input, d_mark, inputSize) : _mean;

	std::cout << "mean : " << mean << std::endl;

	int* d_temp;
	gpuErrChk(cudaMalloc((void**)&d_temp, inputSize * sizeof(int)));

	device_seperateStep1 << <blocks, threads >> > (d_input, d_mark, d_temp, mean, inputSize);

	unsigned long long* d_sum;
	unsigned int* d_cnt;

	gpuErrChk(cudaMalloc((void**)&d_sum, sizeof(unsigned long long)));
	gpuErrChk(cudaMalloc((void**)&d_cnt, sizeof(unsigned int)));

	gpuErrChk(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
	gpuErrChk(cudaMemset(d_cnt, 0, sizeof(unsigned int)));

	device_sumMarked << <blocks, threads >> > (d_temp, d_mark, d_sum, d_cnt, inputSize);

	unsigned int long long sum;
	unsigned int cnt;

	gpuErrChk(cudaMemcpy(&sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	gpuErrChk(cudaMemcpy(&cnt, d_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	unsigned long long mean2 = sum / cnt;

	std::cout << "sum2 : " << sum << std::endl;
	std::cout << "cnt2 : " << cnt << std::endl;

	std::cout << "mean2 : " << mean2 << std::endl;
	std::cout << "std : " << sqrt(mean2) << std::endl;

	gpuErrChk(cudaFree(d_temp));
	gpuErrChk(cudaFree(d_sum));
	gpuErrChk(cudaFree(d_cnt));

	return sqrtf(mean2);
}


void KMean(const float* d_input, unsigned char* d_output, float lower, float upper, int clusterSize, int inputSize)
{
	const int thNum = 1024;

	dim3 threads(thNum);
	dim3 blocks(inputSize / thNum + 1);

	float* clusters = new float[clusterSize];

	memset(clusters, 0, clusterSize * sizeof(float));

	// 	for (int i = 0; i < clusterSize; i++)
	// 	{
	// 		clusters[i] = lower + ((upper - lower) / (float)(clusterSize + 2)) * (i + 1);
	// 	}

		// kmeans++
	float* centers = new float[clusterSize];
	unsigned int* indices = new unsigned int[clusterSize];
	{
		int localTrials = 2 + (int)log(clusterSize);

		float* h_input = new float[inputSize];
		gpuErrChk(cudaMemcpy(h_input, d_input, inputSize * sizeof(float), cudaMemcpyDeviceToHost));

		srand(931016);

		float fTemp = ((float)rand() / (float)RAND_MAX) * clusterSize;

		unsigned int centerId = (unsigned int)fTemp;

		centers[0] = h_input[centerId];
		indices[0] = centerId;

		float* d_closestDistSq;
		gpuErrChk(cudaMalloc((void**)&d_closestDistSq, inputSize * sizeof(float)));

		device_Euclidean_Squared_distances << <blocks, threads >> > (d_input, d_closestDistSq, centers[0], inputSize);

		gpuErrChk(cudaDeviceSynchronize());

		float* h_closestDistSq = new float[inputSize];
		gpuErrChk(cudaMemcpy(h_closestDistSq, d_closestDistSq, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
		float currentPot = 0.;
		for (int i = 0; i < inputSize; i++)
			currentPot += h_closestDistSq[i];

		float* randVals = new float[localTrials];
		float* cumsum = new float[inputSize];
		unsigned int* candidatyeIds = new unsigned int[localTrials];
		float** d_distanceToCandidates = new float* [localTrials];
		float** h_distanceToCandidates = new float* [localTrials];

		for (int i = 0; i < localTrials; i++)
		{
			gpuErrChk(cudaMalloc((void**)&d_distanceToCandidates[i], inputSize * sizeof(float)));
			h_distanceToCandidates[i] = new float[inputSize];
		}

		for (int i = 1; i < clusterSize; i++)
		{
			for (int j = 0; j < localTrials; j++)
				randVals[j] = ((float)rand() / (float)RAND_MAX) * currentPot;

			cumsum[0] = h_closestDistSq[0];
			for (int j = 1; j < inputSize; j++)
				cumsum[j] = cumsum[j - 1] + h_closestDistSq[j];

			for (int j = 0; j < localTrials; j++)
			{
				unsigned int idx = 0;
				for (int k = 1; k < inputSize; k++)
				{
					if (cumsum[k - 1] < randVals[j] && cumsum[k] > randVals[j])
					{
						idx = k;
						break;
					}
				}
				candidatyeIds[j] = idx > (inputSize - 1) ? (inputSize - 1) : idx;
			}

			unsigned int bestCandidate = -1;
			currentPot = FLT_MAX;
			for (int j = 0; j < localTrials; j++)
			{
				device_Euclidean_Squared_distances << <blocks, threads >> > (d_input, d_distanceToCandidates[j], h_input[candidatyeIds[j]], inputSize);

				device_minimum << <blocks, threads >> > (d_distanceToCandidates[j], d_closestDistSq, inputSize);
				gpuErrChk(cudaDeviceSynchronize());

				gpuErrChk(cudaMemcpy(h_distanceToCandidates[j], d_distanceToCandidates[j], inputSize * sizeof(float), cudaMemcpyDeviceToHost));

				float candidatesPot = 0.;
				for (int k = 0; k < inputSize; k++)
					candidatesPot += h_distanceToCandidates[j][k];

				if (candidatesPot < currentPot)
				{
					bestCandidate = j;
					currentPot = candidatesPot;
				}
			}

			gpuErrChk(cudaMemcpy(d_closestDistSq, d_distanceToCandidates[bestCandidate], inputSize * sizeof(float), cudaMemcpyDeviceToDevice));
			bestCandidate = candidatyeIds[bestCandidate];

			centers[i] = h_input[bestCandidate];
			indices[i] = bestCandidate;
		}

		gpuErrChk(cudaFree(d_closestDistSq));
		for (int i = 0; i < localTrials; i++)
		{
			gpuErrChk(cudaFree(d_distanceToCandidates[i]));
			SAFE_DELETE_ARRAY(h_distanceToCandidates[i]);
		}
		SAFE_DELETE_ARRAY(h_input);
		SAFE_DELETE_ARRAY(h_closestDistSq);
		SAFE_DELETE_ARRAY(randVals);
		SAFE_DELETE_ARRAY(cumsum);
		SAFE_DELETE_ARRAY(candidatyeIds);
		SAFE_DELETE_ARRAY(d_distanceToCandidates);
		SAFE_DELETE_ARRAY(h_distanceToCandidates);
	}

	memcpy(clusters, centers, sizeof(float) * clusterSize);

	SAFE_DELETE_ARRAY(centers);
	SAFE_DELETE_ARRAY(indices);

	float* h_sum;// = new float[clusterSize];
	unsigned int* h_cnt;// = new unsigned int[clusterSize];

	float* d_clusters;
	unsigned char* d_label;

	float* d_sum;
	unsigned int* d_cnt;

	gpuErrChk(cudaMalloc((void**)&d_clusters, clusterSize * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_label, inputSize));
	gpuErrChk(cudaMalloc((void**)&d_sum, clusterSize * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_cnt, clusterSize * sizeof(unsigned int)));
	gpuErrChk(cudaMallocHost((void**)&h_sum, clusterSize * sizeof(float)));
	gpuErrChk(cudaMallocHost((void**)&h_cnt, clusterSize * sizeof(unsigned int)));

	int nIter = 0;
	while (nIter < 300)
	{
		std::sort(&clusters[0], &clusters[clusterSize - 1]);

		gpuErrChk(cudaMemcpy(d_clusters, &clusters[0], clusterSize * sizeof(float), cudaMemcpyHostToDevice));

		gpuErrChk(cudaMemset(d_sum, 0, clusterSize * sizeof(float)));
		gpuErrChk(cudaMemset(d_cnt, 0, clusterSize * sizeof(unsigned int)));

		device_KMeansStep1 << <blocks, threads >> > (d_input, d_clusters, d_label, inputSize, clusterSize);
		device_KMeansStep2 << <blocks, threads >> > (d_input, d_label, d_sum, d_cnt, inputSize);

		gpuErrChk(cudaDeviceSynchronize());

		gpuErrChk(cudaMemcpy(h_sum, d_sum, clusterSize * sizeof(float), cudaMemcpyDeviceToHost));
		gpuErrChk(cudaMemcpy(h_cnt, d_cnt, clusterSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		float* h_mean = new float[clusterSize];
		for (int i = 0; i < clusterSize; i++)
		{
			h_mean[i] = h_sum[i] / h_cnt[i];
		}

		int same = 0;
		for (int i = 0; i < clusterSize; i++)
		{
			if (clusters[i] == h_mean[i])
				same++;
		}

		memcpy(&clusters[0], &h_mean[0], clusterSize * sizeof(float));

		delete[] h_mean;
		h_mean = nullptr;

		nIter++;

		if (same == clusterSize)
			break;
	}

	gpuErrChk(cudaMemcpy(d_output, d_label, inputSize, cudaMemcpyDeviceToDevice));

	gpuErrChk(cudaFree(d_clusters));
	gpuErrChk(cudaFree(d_label));
	gpuErrChk(cudaFree(d_sum));
	gpuErrChk(cudaFree(d_cnt));
	gpuErrChk(cudaFreeHost(h_sum));
	gpuErrChk(cudaFreeHost(h_cnt));

	SAFE_DELETE_ARRAY(clusters);
}





void canny(const char* d_input, char* d_output, const int shape_x, const int shape_y, const int shape_z)
{
	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	int count = shape_x * shape_y * shape_z;

	float h_gaussianKernel[25] = { 0.002969016248,	0.01330620867,	0.02193822962,	0.01330620867,	0.002969016248,
									0.01330620867,	0.05963429446,	0.0983203313,	0.05963429446,	0.01330620867,
									0.02193822962,	0.0983203313,	0.1621028241,	0.0983203313,	0.02193822962,
									0.01330620867,	0.05963429446,	0.0983203313,	0.05963429446,	0.01330620867,
									0.002969016248,	0.01330620867,	0.02193822962,	0.01330620867,	0.002969016248 };

	float* d_gaussianKernel;
	gpuErrChk(cudaMalloc((void**)&d_gaussianKernel, 25 * sizeof(float)));
	gpuErrChk(cudaMemcpy(d_gaussianKernel, h_gaussianKernel, 25 * sizeof(float), cudaMemcpyHostToDevice));

	char* d_blur;
	gpuErrChk(cudaMalloc((void**)&d_blur, count));

	device_gaussianBlur<char> << <gridDim, blockDim >> > (d_input, d_blur, d_gaussianKernel, shape_x, shape_y, shape_z, 5, 5);

	char* d_magnitude;
	char* d_angle;
	gpuErrChk(cudaMalloc((void**)&d_magnitude, count));
	gpuErrChk(cudaMalloc((void**)&d_angle, count));

	device_sobel << <gridDim, blockDim >> > (d_blur, d_magnitude, d_angle, shape_x, shape_y, shape_z);

	char* d_edge;
	gpuErrChk(cudaMalloc((void**)&d_edge, count));

	device_edge_thin << <gridDim, blockDim >> > (d_magnitude, d_angle, d_edge, shape_x, shape_y, shape_z);

	unsigned char* d_edge2;
	gpuErrChk(cudaMalloc((void**)&d_edge2, count));

	device_edge_thin << <gridDim, blockDim >> > (d_edge, d_edge2, shape_x, shape_y, shape_z, 25, 51);

	unsigned char* d_hystIters;
	gpuErrChk(cudaMalloc((void**)&d_hystIters, count));

	cudaMemcpy(d_hystIters, d_edge2, count, cudaMemcpyDeviceToDevice);

	for (int i = 0; i < 2; i++)
	{
		device_hysteresis << <gridDim, blockDim >> > (d_hystIters, shape_x, shape_y, shape_z, i == 2 - 1);
	}

	gpuErrChk(cudaMemcpy(d_output, d_hystIters, count, cudaMemcpyDeviceToDevice));

	gpuErrChk(cudaDeviceSynchronize());

	gpuErrChk(cudaFree(d_blur));
	gpuErrChk(cudaFree(d_magnitude));
	gpuErrChk(cudaFree(d_angle));
	gpuErrChk(cudaFree(d_edge));
	gpuErrChk(cudaFree(d_edge2));
	gpuErrChk(cudaFree(d_hystIters));
	gpuErrChk(cudaFree(d_gaussianKernel));
}
void getLargestComponent(const char* d_in, char* const d_out, const int shape_x, const int shape_y, const int shape_z)
{
	int cnt = shape_x * shape_y * shape_z;

	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	int* d_label;
	gpuErrChk(cudaMalloc((void**)&d_label, cnt * sizeof(int)));

	device_InitLabeling << <gridDim, blockDim >> > (d_in, d_label, shape_x, shape_y, shape_z);
	device_MergeLabel << <gridDim, blockDim >> > (d_label, shape_x, shape_y, shape_z);
	device_PathCompressionLabel << <gridDim, blockDim >> > (d_label, shape_x, shape_y, shape_z);

	gpuErrChk(cudaDeviceSynchronize());

	int* label = new int[shape_x * shape_y * shape_z];

	gpuErrChk(cudaMemcpy(&label[0], d_label, cnt * sizeof(int), cudaMemcpyDeviceToHost));

	std::map<int, int> mapDict;
	int itLabel = 1;
	for (int i = 0; i < shape_x * shape_y * shape_z; i++)
	{
		if (label[i] != 0)
		{
			auto iterMapFind = mapDict.find(label[i]);
			if (iterMapFind == mapDict.end())
			{
				mapDict[label[i]] = itLabel;
				label[i] = itLabel++;
			}
			else
			{
				label[i] = iterMapFind->second;
			}
		}
	}
	int numLabels = mapDict.size();

	std::cout << "numLabels: " << numLabels << std::endl;
	int range[2] = { 1, numLabels + 1 };
	int* d_range;
	int* d_sum;

	gpuErrChk(cudaMalloc((void**)&d_range, 2 * sizeof(int)));
	gpuErrChk(cudaMalloc((void**)&d_sum, numLabels * sizeof(int)));

	gpuErrChk(cudaMemset(d_sum, 0, numLabels * sizeof(int)));

	gpuErrChk(cudaMemcpy(d_label, &label[0], cnt * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_range, &range[0], 2 * sizeof(int), cudaMemcpyHostToDevice));

	device_SumLabeledArray << <gridDim, blockDim >> > (d_label, d_range, d_sum, shape_x, shape_y, shape_z);

	gpuErrChk(cudaDeviceSynchronize());

	int* sum = new int[numLabels];

	gpuErrChk(cudaMemcpy(&sum[0], d_sum, numLabels * sizeof(int), cudaMemcpyDeviceToHost));

	int largest_component = 0;
	int max_index = 0;
	for (int i = 0; i < numLabels; i++)
	{
		if (sum[i] > largest_component)
		{
			largest_component = sum[i];
			max_index = i;
		}
	}
	max_index += 1;
	std::cout << "maxIndex: " << max_index << std::endl;
	std::cout << "largestComponent: " << largest_component << std::endl;

	device_GetLargestComponentArray << <gridDim, blockDim >> > (d_label, d_out, max_index, shape_x, shape_y, shape_z);

	gpuErrChk(cudaDeviceSynchronize());

	SAFE_DELETE_ARRAY(label);
	SAFE_DELETE(sum);

	gpuErrChk(cudaFree(d_label));
	gpuErrChk(cudaFree(d_range));
	gpuErrChk(cudaFree(d_sum));
}


void findBorder(const char* d_stack, const char* d_la, char* d_output, const int shape_x, const int shape_y, const int shape_z)
{
	int count = shape_x * shape_y * shape_z;

	float h_gaussianKernel[25] = { 0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625,
											0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625,
											0.0234375 , 0.09375   , 0.140625  , 0.09375   , 0.0234375,
											0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625,
											0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625 };

	float* d_gaussianKernel;
	gpuErrChk(cudaMalloc((void**)&d_gaussianKernel, 25 * sizeof(float)));
	gpuErrChk(cudaMemcpy(d_gaussianKernel, h_gaussianKernel, 25 * sizeof(float), cudaMemcpyHostToDevice));

	char h_struct33[9] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
	char h_struct333[27] = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };

	char* d_structure33;
	char* d_structure333;
	gpuErrChk(cudaMalloc((void**)&d_structure33, 9));
	gpuErrChk(cudaMalloc((void**)&d_structure333, 27));
	gpuErrChk(cudaMemcpy(d_structure33, h_struct33, 9, cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_structure333, h_struct333, 27, cudaMemcpyHostToDevice));

	char* d_blur;
	gpuErrChk(cudaMalloc((void**)&d_blur, count));

	char* d_blur2;
	char* d_blurTmp;
	gpuErrChk(cudaMalloc((void**)&d_blur2, count));
	gpuErrChk(cudaMalloc((void**)&d_blurTmp, count));

	gpuErrChk(cudaMemcpy(d_blur2, d_stack, count, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < 3; i++)
	{
		gaussianBlur<char>(d_blur2, d_blurTmp, d_gaussianKernel, shape_x, shape_y, shape_z, 5, 5);

		gpuErrChk(cudaMemcpy(d_blur2, d_blurTmp, count, cudaMemcpyDeviceToDevice));
	}

	gpuErrChk(cudaMemcpy(d_blur, d_blurTmp, count, cudaMemcpyDeviceToDevice));

	char* d_erosion;
	gpuErrChk(cudaMalloc((void**)&d_erosion, count));

	binaryErosion(d_la, d_structure333, d_erosion, shape_x, shape_y, shape_z, 1, 3);

	char* d_largest;
	gpuErrChk(cudaMalloc((void**)&d_largest, count));

	getLargestComponent(d_erosion, d_largest, shape_x, shape_y, shape_z);

	gpuErrChk(cudaFree(d_erosion));

	char* d_dilation;
	gpuErrChk(cudaMalloc((void**)&d_dilation, count));

	binaryDilation(d_largest, d_structure333, d_dilation, shape_x, shape_y, shape_z, 1, 3);

	gpuErrChk(cudaFree(d_largest));

	char* d_canny;
	gpuErrChk(cudaMalloc((void**)&d_canny, count));

	canny(d_blur, d_canny, shape_x, shape_y, shape_z);

	char* d_cannyDilation;
	gpuErrChk(cudaMalloc((void**)&d_cannyDilation, count));

	binaryDilation2D(d_canny, d_structure33, d_cannyDilation, shape_x, shape_y, shape_z, 1, 3);

	char* d_tmp;
	gpuErrChk(cudaMalloc((void**)&d_tmp, count));

	{
		const int tnum = 1024;

		dim3 threads(tnum);
		dim3 blocks(count / tnum);

		device_findborder_tmp << <blocks, threads >> > (d_dilation, d_cannyDilation, d_tmp, count);
	}

	gpuErrChk(cudaFree(d_dilation));

	char* d_largest2;
	gpuErrChk(cudaMalloc((void**)&d_largest2, count));

	getLargestComponent(d_tmp, d_largest2, shape_x, shape_y, shape_z);

	char* d_dilation2;
	gpuErrChk(cudaMalloc((void**)&d_dilation2, count));

	binaryDilation(d_largest2, d_structure333, d_dilation2, shape_x, shape_y, shape_z, 1, 4);

	gpuErrChk(cudaMemcpy(d_blur2, d_dilation2, count, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < 3; i++)
	{
		gaussianBlur<char>(d_blur2, d_blurTmp, d_gaussianKernel, shape_x, shape_y, shape_z, 5, 5);

		gpuErrChk(cudaMemcpy(d_blur2, d_blurTmp, count, cudaMemcpyDeviceToDevice));
	}

	gpuErrChk(cudaDeviceSynchronize());

	gpuErrChk(cudaMemcpy(d_output, d_blur2, count, cudaMemcpyDeviceToDevice));

	gpuErrChk(cudaFree(d_gaussianKernel));
	gpuErrChk(cudaFree(d_structure33));
	gpuErrChk(cudaFree(d_structure333));
	gpuErrChk(cudaFree(d_blur));
	gpuErrChk(cudaFree(d_blur2));
	gpuErrChk(cudaFree(d_blurTmp));
	gpuErrChk(cudaFree(d_canny));
	gpuErrChk(cudaFree(d_cannyDilation));
	gpuErrChk(cudaFree(d_tmp));
	gpuErrChk(cudaFree(d_largest2));
	gpuErrChk(cudaFree(d_dilation2));
}

void _SaveToFile(void* input,
	const std::string& filename, const unsigned int& bytes)
{
	gLogInfo << "--------------------\n";
	gLogInfo << "Saving output data from binary file\n";
	gLogInfo << "\t" << filename << "\n";
	{
		PreciseCpuTimer mt; mt.start();

		std::ofstream out(filename, std::ios::out | std::ios::binary);
		out.write((char*)input, bytes);
		out.close();
		mt.stop();
		gLogInfo << "\t[Elapsed time: " << mt.milliseconds() << "ms]\n";
	}
	gLogInfo << "--------------------\n";
}
//
//template<class T>
//std::unique_ptr<T[]> DuplicateArrayDeviceToHost(const T* data, const int bytes) {
//    auto result = std::make_unique<T[]>(bytes/(sizeof(T)));
//    gpuErrChk(cudaMemcpy(result.get(), data, bytes, cudaMemcpyDeviceToHost));
//    return std::move(result);
//}
