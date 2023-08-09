#pragma once
#include "common.h"
#include "kernel.cuh"


__global__ void device_toBits(
    const float* __restrict__ in,
    volatile char* __restrict__ out,
    const float threshold,
    const int stride)
{
    const int gid = (threadIdx.x + blockDim.x * blockIdx.x) * stride + 1;
    const int bid = gid / stride;
//    const int bit_pos = bid % (1 << 3);
    
    //const int lid = lane_id();
    //const int wid = warp_id();

//    extern __shared__ char sd[];
    
    //const int val = in[gid] > threshold ? true : false;
//    sd[threadIdx.x] = in[gid] > threshold ? 1 << (7 - bit_pos): 0;
//     __syncthreads();
// 
//     sd[threadIdx.x] |= __shfl_xor_sync(-1, sd[threadIdx.x], 1);
//     sd[threadIdx.x] |= __shfl_xor_sync(-1, sd[threadIdx.x], 2);
//     sd[threadIdx.x] |= __shfl_xor_sync(-1, sd[threadIdx.x], 4);
//     __syncthreads();
//    if (bit_pos == 0) {
//        out[bid / (1 << 3)] = sd[threadIdx.x];
//    }
   out[bid] = in[gid] > threshold ? 1 : 0;
    
};

//
//void binaryDilation2D(const char* d_input, const char* d_structure, char* d_output, const int shape_x, const int shape_y, const int shape_z, const int structure_radius, const int iterations)
//{
//	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
//	dim3 blockDim(8, 8, 8);
//
//	char* d_arr;
//	gpuErrChk(cudaMalloc((void**)&d_arr, shape_x * shape_y * shape_z));
//
//	gpuErrChk(cudaMemcpy(d_arr, d_input, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));
//
//	for (int i = 0; i < iterations; i++)
//	{
//		gpuErrChk(cudaMemset(d_output, 0, shape_x * shape_y * shape_z));
//
//		device_binary_dilation2D << <gridDim, blockDim >> > (d_arr, d_output, d_structure, shape_x, shape_y, shape_z, structure_radius);
//
//		gpuErrChk(cudaDeviceSynchronize());
//
//		gpuErrChk(cudaMemcpy(d_arr, d_output, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));
//	}
//}

void logicalAnd(const char* d_inputA, const char* d_inputB, char* d_output, const int inputSize)
{
	const int tnum = 1024;

	dim3 threads(tnum);
	dim3 blocks(inputSize / tnum);

	device_logical_and << <blocks, threads >> > (d_inputA, d_inputB, d_output, inputSize);
}

void logicalOr(const char* d_inputA, const char* d_inputB, char* d_output, const int inputSize)
{
	const int tnum = 1024;

	dim3 threads(tnum);
	dim3 blocks(inputSize / tnum);

	device_logical_or << <blocks, threads >> > (d_inputA, d_inputB, d_output, inputSize);
}


extern "C"
void toBits(const float* d_in, char* const h_out, const unsigned long & cnt, const int stride)
{
    char* d_out;
    const int tnum = 1024;
    gpuErrChk(cudaMalloc((void**)&d_out, cnt/ (1 << 3)));
	

    const float threshold = 0.0f;
    dim3 threads(tnum);
    // n c h w
    dim3 blocks(cnt / tnum);
    device_toBits
        << < blocks, threads, sizeof(char)* tnum >> >
        (d_in, d_out, threshold, stride);
	gpuErrChk(cudaDeviceSynchronize());


    gpuErrChk(cudaMemcpy(h_out, d_out, cnt / (1 << 3), cudaMemcpyDeviceToHost));

    gpuErrChk(cudaFree(d_out));

}

//extern "C"
//void getLargestComponent(const char* d_in, char* const d_out, const int shape_x, const int shape_y, const int shape_z)
//{
//	int cnt = shape_x * shape_y * shape_z;
//
//	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
//	dim3 blockDim(8, 8, 8);
//
//	int* d_label;
//	gpuErrChk(cudaMalloc((void**)&d_label, cnt * sizeof(int)));
//
//	device_InitLabeling << <gridDim, blockDim >> > (d_in, d_label, shape_x, shape_y, shape_z);
//	device_MergeLabel << <gridDim, blockDim >> > (d_label, shape_x, shape_y, shape_z);
//	device_PathCompressionLabel << <gridDim, blockDim >> > (d_label, shape_x, shape_y, shape_z);
//
//	gpuErrChk(cudaDeviceSynchronize());
//
//	int* label = new int[shape_x * shape_y * shape_z];
//
//	gpuErrChk(cudaMemcpy(&label[0], d_label, cnt * sizeof(int), cudaMemcpyDeviceToHost));
//
//	std::map<int, int> mapDict;
//	int itLabel = 1;
//	for (int i = 0; i < shape_x * shape_y * shape_z; i++)
//	{
//		if (label[i] != 0)
//		{
//			auto iterMapFind = mapDict.find(label[i]);
//			if (iterMapFind == mapDict.end())
//			{
//				mapDict[label[i]] = itLabel;
//				label[i] = itLabel++;
//			}
//			else
//			{
//				label[i] = iterMapFind->second;
//			}
//		}
//	}
//	int numLabels = mapDict.size();
//
//	std::cout << "numLabels: " << numLabels << std::endl;
//	int range[2] = { 1, numLabels + 1 };
//	int* d_range;
//	int* d_sum;
//
//	gpuErrChk(cudaMalloc((void**)&d_range, 2 * sizeof(int)));
//	gpuErrChk(cudaMalloc((void**)&d_sum, numLabels * sizeof(int)));
//
//	gpuErrChk(cudaMemset(d_sum, 0, numLabels * sizeof(int)));
//
//	gpuErrChk(cudaMemcpy(d_label, &label[0], cnt * sizeof(int), cudaMemcpyHostToDevice));
//	gpuErrChk(cudaMemcpy(d_range, &range[0], 2 * sizeof(int), cudaMemcpyHostToDevice));
//
//	device_SumLabeledArray << <gridDim, blockDim >> > (d_label, d_range, d_sum, shape_x, shape_y, shape_z);
//
//	gpuErrChk(cudaDeviceSynchronize());
//
//	int* sum = new int[numLabels];
//
//	gpuErrChk(cudaMemcpy(&sum[0], d_sum, numLabels * sizeof(int), cudaMemcpyDeviceToHost));
//
//	int largest_component = 0;
//	int max_index = 0;
//	for (int i = 0; i < numLabels; i++)
//	{
//		if (sum[i] > largest_component)
//		{
//			largest_component = sum[i];
//			max_index = i;
//		}
//	}
//	max_index += 1;
//	std::cout << "maxIndex: " << max_index << std::endl;
//	std::cout << "largestComponent: " << largest_component << std::endl;
//
//	GetLargestComponentArray << <gridDim, blockDim >> > (d_label, d_out, max_index, shape_x, shape_y, shape_z);
//
//	gpuErrChk(cudaDeviceSynchronize());
//
//	free(label);
//	free(sum);
//
//	gpuErrChk(cudaFree(d_label));
//	gpuErrChk(cudaFree(d_range));
//	gpuErrChk(cudaFree(d_sum));
//}

extern "C"
void getLargestComponentChar(char* d_inout, const Dims& iodim)
{
	if (iodim.nbDims != 3)
		return;

	const int num = volume(iodim);
	const int bytes = num * sizeof(int);

	Dims dim = iodim;
	
	dim3 gridDim(dim.d[0] / 8, dim.d[1] / 8, dim.d[2] / 8 + 1);
	dim3 blockDim(8, 8, 8);
	
	int* d_label;
	gpuErrChk(cudaMalloc((void**)&d_label, bytes));

	device_InitLabeling << <gridDim, blockDim >> > (d_inout, d_label, dim.d[0], dim.d[1], dim.d[2]);
	device_MergeLabel << <gridDim, blockDim >> > (d_label, dim.d[0], dim.d[1], dim.d[2]);
	device_PathCompressionLabel << <gridDim, blockDim >> > (d_label, dim.d[0], dim.d[1], dim.d[2]);

	gpuErrChk(cudaDeviceSynchronize());

	int* label = new int[num];

	//auto label_ = std::make_unique<int[]>(num);
	//auto label = label_.get();

	gpuErrChk(cudaMemcpy(label, d_label, bytes, cudaMemcpyDeviceToHost));

	std::map<int, int> mapDict;
	int itLabel = 1;

	for (int i = 0; i < num; i++)
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
	//	int numLabels = vecOutput.size();
	std::cout << "numLabels: " << numLabels << std::endl;
	int range[2] = { 1, numLabels + 1 };
	int* d_range;
	int* d_sum;

	gpuErrChk(cudaMalloc((void**)&d_range, 2 * sizeof(int)));
	gpuErrChk(cudaMalloc((void**)&d_sum, numLabels * sizeof(int)));

	gpuErrChk(cudaMemset(d_sum, 0, numLabels * sizeof(int)));

	gpuErrChk(cudaMemcpy(d_label, &label[0], bytes, cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_range, &range[0], 2 * sizeof(int), cudaMemcpyHostToDevice));

	device_SumLabeledArray << <gridDim, blockDim >> > (d_label, d_range, d_sum, dim.d[0], dim.d[1], dim.d[2]);

	gpuErrChk(cudaDeviceSynchronize());

	//int* sum = new int[numLabels];
	auto sum_ = std::make_unique<int[]>(numLabels);
	auto sum = sum_.get();
	gpuErrChk(cudaMemcpy(sum, d_sum, numLabels * sizeof(int), cudaMemcpyDeviceToHost));

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
	
	/*char* d_result;
	gpuErrChk(cudaMalloc((void**)&d_result, num));
	gpuErrChk(cudaMemset(d_result, 0, num));*/

	device_GetLargestComponentArray << <gridDim, blockDim >> > (d_label, d_inout, max_index, dim.d[0], dim.d[1], dim.d[2]);

	gpuErrChk(cudaDeviceSynchronize());

	//gpuErrChk(cudaMemcpy(&h_out[0], d_result, num, cudaMemcpyDeviceToHost));

	delete[] label;
	//free(sum);

	//gpuErrChk(cudaFree(d_image));
	gpuErrChk(cudaFree(d_label));
	gpuErrChk(cudaFree(d_range));
	gpuErrChk(cudaFree(d_sum));
	//gpuErrChk(cudaFree(d_result));
}

//void skeletonize(char* d_arr, int shape_x, int shape_y, int shape_z)
//{
//	char* d_markers;
//	int* d_count;
//
//	size_t size = shape_x * shape_y * shape_z * sizeof(char);
//
//	gpuErrChk(cudaMalloc((void**)&d_markers, size));
//	gpuErrChk(cudaMalloc((void**)&d_count, sizeof(int)));
//
//	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
//	dim3 blockDim(8, 8, 8);
//
//	int processed = 0, count = 0;
//
//	do {
//		count = 0;
//		processed = 0;
//		gpuErrChk(cudaMemset(d_markers, 0, size));
//		gpuErrChk(cudaMemset(d_count, 0, sizeof(int)));
//
//		device_Skeletonize2DStep1 << <gridDim, blockDim >> > (d_arr, d_markers, d_count, shape_x, shape_y, shape_z);
//
//		gpuErrChk(cudaDeviceSynchronize());
//
//		gpuErrChk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
//
//		std::cout << "first : " << count << std::endl;
//		processed = (count > 0);
//
//		if (processed)
//		{
//			device_BlackToWhite << <gridDim, blockDim >> > (d_arr, d_markers, shape_x, shape_y, shape_z);
//			gpuErrChk(cudaDeviceSynchronize());
//		}
//
//		gpuErrChk(cudaMemset(d_markers, 0, size));
//		gpuErrChk(cudaMemset(d_count, 0, sizeof(int)));
//
//		device_Skeletonize2DStep2 << <gridDim, blockDim >> > (d_arr, d_markers, d_count, shape_x, shape_y, shape_z);
//
//		gpuErrChk(cudaDeviceSynchronize());
//
//		gpuErrChk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
//		std::cout << "second : " << count << std::endl;
//		if (processed == 0)
//			processed = (count > 0);
//
//		if (processed)
//		{
//			device_BlackToWhite << <gridDim, blockDim >> > (d_arr, d_markers, shape_x, shape_y, shape_z);
//			gpuErrChk(cudaDeviceSynchronize());
//		}
//	} while (processed == 1);
//
//	gpuErrChk(cudaFree(d_markers));
//	gpuErrChk(cudaFree(d_count));
//}

extern "C"
void thresholding(const void* d_in, char* d_out, Dims & iodim, const nvinfer1::DataType & dtype, const float& threshold1, const float& threshold2)
{
	
	auto threshFunc1 = [threshold1] __device__(const auto& data) -> char
	{
		return data>threshold1;
	};
	
	auto threshFunc2 = [threshold1, threshold2] __device__(const auto& data) -> char
	{
		if (data < threshold1)
			return threshold1;
		else if (data > threshold2)
			return threshold2;
		else
			return data;
	};

	Dims odim = iodim;
	if (threshold2 == -1)
		odim.d[--odim.nbDims] = 0;

	const int trd= 1024;
	const int num = volume(odim);
	int stride = iodim.d[iodim.nbDims - 1];
	stride = (threshold2 == -1) ? stride : 1;

	dim3 threads(trd);
	dim3 blocks(num/ trd);

	if (threshold2 == -1)
	{
		if (stride == 1)
			device_customFunctor<char, char> << < blocks, threads >> > ((char*)d_in, d_out, threshFunc1, stride);
		else if (stride == 2)
		{
			//device_customFunctor<float, char> << < blocks, threads >> > ((float*)d_in, d_out, threshFunc1, stride);
			switch (dtype)
			{
			case nvinfer1::DataType::kFLOAT:
			{
				device_customFunctor<float, char> << < blocks, threads >> > ((float*)d_in, d_out, threshFunc1, stride);
				break;
			}
			case nvinfer1::DataType::kINT8:
			{
				device_customFunctor<char, char> << < blocks, threads >> > ((char*)d_in, d_out, threshFunc1, stride);
				break;
			}
			case nvinfer1::DataType::kINT32:
			{
				device_customFunctor<int, char> << < blocks, threads >> > ((int*)d_in, d_out, threshFunc1, stride);
				break;
			}
			case nvinfer1::DataType::kBOOL:
			{
				device_customFunctor<bool, char> << < blocks, threads >> > ((bool*)d_in, d_out, threshFunc1, stride);
				break;
			}
			default:
				break;
			}
		}
	}
	else
	{
		if (stride == 1)
			device_customFunctor<char, char> << < blocks, threads >> > ((char*)d_in, d_out, threshFunc2, stride);
		else if (stride == 2)
		{
			//device_customFunctor<float, char> << < blocks, threads >> > ((float*)d_in, d_out, threshFunc1, stride);
			switch (dtype)
			{
			case nvinfer1::DataType::kFLOAT:
			{
				device_customFunctor<float, char> << < blocks, threads >> > ((float*)d_in, d_out, threshFunc2, stride);
				break;
			}
			case nvinfer1::DataType::kINT8:
			{
				device_customFunctor<char, char> << < blocks, threads >> > ((char*)d_in, d_out, threshFunc2, stride);
				break;
			}
			case nvinfer1::DataType::kINT32:
			{
				device_customFunctor<int, char> << < blocks, threads >> > ((int*)d_in, d_out, threshFunc2, stride);
				break;
			}
			case nvinfer1::DataType::kBOOL:
			{
				device_customFunctor<bool, char> << < blocks, threads >> > ((bool*)d_in, d_out, threshFunc2, stride);
				break;
			}
			default:
				break;
			}
		}
	}

	iodim = odim;
}

extern "C"
void call_binary_dilation(
	char* d_inout,
	const char* h_structure,
	const Dims& dim,
	const int structure_radius,
	const int iter,
	const cudaStream_t & stream)
{
	if (dim.nbDims != 3)
		return;

	const int num = volume(dim);
	const int bytes = num * sizeof(unsigned char);
	dim3 gridDim(dim.d[0] / 8, dim.d[1] / 8, dim.d[2] / 8 + 1);
	dim3 blockDim(8, 8, 8);

	char* d_buffer;
	gpuErrChk(cudaMalloc((void**)&d_buffer, bytes));

	char* d_structure;
	const int structSize = sizeof(char) *
		(structure_radius * 2 + 1) * (structure_radius * 2 + 1) * (structure_radius * 2 + 1);
	gpuErrChk(cudaMalloc((void**)&d_structure, structSize));
	gpuErrChk(cudaMemcpyAsync(d_structure, h_structure, structSize, cudaMemcpyHostToDevice,stream));

	for (int i = 0; i < iter; i++)
	{
		/*gpuErrChk(cudaStreamSynchronize(stream));
		if (i % 2 == 1)
			binary_dilation << <gridDim, blockDim, 0, stream >> > (d_buffer, d_inout, d_structure, dim.d[0], dim.d[1], dim.d[2], structure_radius);
		else*/
		device_binary_dilation << <gridDim, blockDim, 0, stream >> > (d_inout, d_buffer, d_structure, dim.d[0], dim.d[1], dim.d[2], structure_radius);
		gpuErrChk(cudaMemcpyAsync(d_inout, d_buffer, bytes, cudaMemcpyDeviceToDevice, stream));
	}

	/*if (iter % 2 == 1)
		gpuErrChk(cudaMemcpyAsync(d_inout, d_buffer, bytes, cudaMemcpyDeviceToDevice,stream));*/

	gpuErrChk(cudaStreamSynchronize(stream));
	gpuErrChk(cudaFree(d_buffer));
	gpuErrChk(cudaFree(d_structure));

}
extern "C"
void call_binary_erosion(
	char* d_inout,
	const char* h_structure,
	const Dims & dim,
	const int structure_radius,
	const int iter,
	const cudaStream_t & stream)
{
	if (dim.nbDims != 3)
		return;

	const int num = volume(dim);
	const int bytes = num * sizeof(char);
	dim3 gridDim(dim.d[0] / 8, dim.d[1] / 8, dim.d[2] / 8 + 1);
	dim3 blockDim(8, 8, 8);

	char* d_buffer;
	gpuErrChk(cudaMalloc((void**)&d_buffer, bytes));

	char* d_structure;
	const int structSize = sizeof(char) *
		(structure_radius * 2 + 1) * (structure_radius * 2 + 1) * (structure_radius * 2 + 1);
	gpuErrChk(cudaMalloc((void**)&d_structure, structSize));
	gpuErrChk(cudaMemcpyAsync(d_structure, h_structure, structSize, cudaMemcpyHostToDevice, stream));

	for (int i = 0; i < iter; i++)
	{
		/*gpuErrChk(cudaStreamSynchronize(stream));
		if (i % 2 == 1)
			binary_erosion << <gridDim, blockDim, 0, stream >> > (d_buffer, d_inout, d_structure, dim.d[0], dim.d[1], dim.d[2], structure_radius);
		else*/
		device_binary_erosion << <gridDim, blockDim, 0, stream >> > (d_inout, d_buffer, d_structure, dim.d[0], dim.d[1], dim.d[2], structure_radius);
		gpuErrChk(cudaMemcpyAsync(d_inout, d_buffer, bytes, cudaMemcpyDeviceToDevice, stream));
	}

	//if (iter % 2 == 1)
	//	gpuErrChk(cudaMemcpyAsync(d_inout, d_buffer, bytes, cudaMemcpyDeviceToDevice, stream));

	gpuErrChk(cudaStreamSynchronize(stream));
	gpuErrChk(cudaFree(d_buffer));
	gpuErrChk(cudaFree(d_structure));

}
extern "C"
void test(char* const out, const Dims & dim)
{
	/*unsigned int* test_value = nullptr;
	size_t* test_idx = nullptr;

	auto close = []__device__(const char* in)
	{
		const int x = threadIdx.x;
		const int y = blockIdx.x;
		const int gid = x + y * blockDim.x;

		char now = in[gid];
		if (now == 0)
		{
			for (auto _x = -1; _x <= 1 && now == 0; _x++)
			{
				const int X = x + _x;
				if (X > -1 && X < blockDim.x)
				{
					for (auto _y = -1; _y <= 1 && now == 0; _y++)
					{
						const int Y = y + _y;
						if (Y > -1 && Y < gridDim.x)
						{
							now |= in[X + Y * blockDim.x];
						}
					}
				}
			}
		}
		return in[gid] ^ now;
	};
	
	constexpr int threads = 1 << 9;
	constexpr int snum = 1 << 4;
	std::vector<cudaStream_t> cs(snum);

	const int pitch = dim.d[0] * dim.d[1];
	const int num = pitch * dim.d[2];
	const int blocks = num / threads;

	char* d_buffer = nullptr;
	gpuErrChk(cudaMalloc((void**)&d_buffer, num*sizeof(char)));


	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamCreateWithFlags(&_cs, cudaStreamNonBlocking));
	}

	for (int d = 0, stride = 0, si = 0;
		d < dim.d[2];
		d++, si = (++si == snum) ? 0 : si, stride += pitch)
	{
		device_1arrayFunctor<char> << < blocks, threads, 0, cs[si] >> > (in + stride, d_buffer + stride, close);
	}
	gpuErrChk(cudaDeviceSynchronize());

	convexFill(out, dim, cs.data());

	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamDestroy(_cs));
	}*/
}
extern "C"
void fill_holes(const char* d_in, char* d_out, const Dims& dim, const Plane& p)
{
	
	int w = 0, h = 0, d = 0;
	int num = 0;
	int bytes = 0;
	const int /*trd = 1 << 9, */snum = 1 << 4;
	dim3 threads;
	dim3 blocks;

	char* d_initBorder = nullptr;
	char* d_mask = nullptr;
	char* d_buffer = nullptr;
	//char* d_bufModif = nullptr;
	//char* d_bufReduc = nullptr;
	bool* h_condition = nullptr;
	std::vector<cudaStream_t> cs(snum);
	//std::unique_ptr<bool[]> condition;
	auto logicalNot = []__device__(const char& x) { return !x; };
	auto dil = []__device__(const char * in, const char * mask)
	{
		const int x = threadIdx.x;
		const int y = blockIdx.x;
		const int gid = x + y * blockDim.x;

		char now = in[gid];
		if (mask[gid] && now == 0)
		{
#pragma unroll (3)
			for (auto _x = -1; _x <= 1; _x++)
			{
				const int X = x + _x;
				if (X > -1 && X < blockDim.x)
				{
					if (now != 0) break;
#pragma unroll (3)
					for (auto _y = -1; _y <= 1; _y++)
					{
						const int Y = y + _y;
						if (Y > -1 && Y < gridDim.x)
						{
							if (now != 0) break;
							now |= in[X + Y * blockDim.x];
						}
					}
				}
			}
			//out[gid] = now;
		}
		return now;
	};
		

	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamCreateWithFlags(&_cs, cudaStreamNonBlocking));
	}

	switch (p)
	{
	case Plane::XY:
		gLogInfo << "Set up binary fill holes XY\n";
		w = dim.d[0]; h = dim.d[1]; d = dim.d[2];
		break;
	case Plane::YZ:
		gLogInfo << "Set up binary fill holes YZ\n";
		w = dim.d[1]; h = dim.d[2]; d = dim.d[0];
		break;
	case Plane::XZ:
		gLogInfo << "Set up binary fill holes XZ\n";
		w = dim.d[0]; h = dim.d[2]; d = dim.d[1];
		break;
	}
		
	num = w * h;
	bytes = num * sizeof(char);
	threads.x = w;
	blocks.x = h;
	//condition = std::make_unique<bool[]>(d);

	gpuErrChk(cudaMalloc((void**)&d_initBorder, bytes));
	gpuErrChk(cudaMalloc((void**)&d_mask, bytes* d));
	gpuErrChk(cudaMalloc((void**)&d_buffer, bytes* d));
	//gpuErrChk(cudaMalloc((void**)&d_bufModif, bytes* d));
	//gpuErrChk(cudaMalloc((void**)&d_bufReduc, bytes* d));

	gpuErrChk(cudaMallocHost((void**)&h_condition, d * sizeof(bool)));
	gpuErrChk(cudaMemset(d_buffer, 0, bytes* d));
	{
		auto pivot = std::make_unique<char[]>(bytes);
		auto h_initBotder = pivot.get();
		for (int i = 0, _endRowFirst = num - w, j = 0, _endCol = w - 1;
			i < num;
			i++, j = i % w)
		{
			if (i < w || (j == _endCol) || (j == 0) || (i >= _endRowFirst))
			{
				*(h_initBotder + i) = 1;
			}
		}
		gpuErrChk(cudaMemcpy(d_initBorder, h_initBotder, bytes, cudaMemcpyHostToDevice));
	}
	//auto cond = condition.get();

	int _start = -1;
	for (int i = 0, wd = w * d, stride = 0, si = 0;
		i < d;
		i++, stride +=num)
	{
		switch (p)
		{
		case Plane::XY:
			if (i == 0)
			{
				cudaMemcpy(d_out, d_in, bytes * d, cudaMemcpyDeviceToDevice);
				gpuErrChk(cudaDeviceSynchronize());
			}
			break;
		case Plane::YZ:
			device_YZplane<char> << <blocks, threads, 0, cs[si] >> > (d_in, d_out + stride, i, w, wd);
			break;
		case Plane::XZ:
			device_XZplane<char> << <blocks, threads, 0, cs[si] >> > (d_in, d_out + stride, i, w, wd);
			break;
		}

		isModified<char>(d_out + stride, d_mask + stride, num, h_condition[i], cs[si]);
		if (_start == -1 && h_condition[i])
		{
			_start = i;
		}
		device_customFunctor<char, char> << <blocks, threads, 0, cs[si] >> > (d_out + stride, d_mask + stride, logicalNot, 1);
		si = (++si == snum) ? 0 : si;
	}
	gpuErrChk(cudaDeviceSynchronize());

	//iter prepare
	for (auto k = 0, l = 0; k < 1<<9; k++, l++)//max iter
	{
		
		bool any = false;
		for (int i = _start, stride = _start * num, si = 0;
			i < d;
			i++, stride += num)
		{
			if (h_condition[i])
			{
				device_2arrayFunctor<char> << < blocks, threads, 0, cs[si] >> >((k == 0 ? d_initBorder : (d_out + stride)), d_mask + stride, d_buffer + stride, dil);
				if (l == 1 << 6)
				{
					isModified<char>(d_out + stride, d_buffer + stride, num, h_condition[i], cs[si]);
					if (h_condition[i])
					{
						cudaMemcpyAsync(d_out + stride, d_buffer + stride, bytes, cudaMemcpyDeviceToDevice, cs[si]);
						any = true;
					}
					else
					{
						device_customFunctor<char, char> << <blocks, threads, 0, cs[si] >> > (d_buffer + stride, d_buffer + stride, logicalNot, 1);
					}
				}
				else
				{
					cudaMemcpyAsync(d_out + stride, d_buffer + stride, bytes, cudaMemcpyDeviceToDevice, cs[si]);
					any = true;
				}
				si = (++si == snum) ? 0 : si;
			}
		}
		l = (l == 1 << 6) ? 0 : l;

		if (!any)
		{
			gLogInfo << "\tbinary fill holes iter loop fin \n";
			break;
		}
		//gLogInfo << "\tbinary fill holes iter[" << k << "]\r";
	}

	
	if (p == Plane::XY)
	{
		cudaMemcpy(d_out, d_buffer, bytes * d, cudaMemcpyDeviceToDevice);
	}
	else
	{
		for (int i = 0, wd = w * d, stride = 0, si = 0;
			i < d;
			i++, stride +=num)
		{
			switch (p)
			{
			case Plane::YZ:
				device_YZplane_rev<char> << <blocks, threads, 0, cs[si] >> > (d_buffer + stride, d_out, i, w, wd);
				break;
			case Plane::XZ:
				device_XZplane_rev<char> << <blocks, threads, 0, cs[si] >> > (d_buffer + stride, d_out, i, w, wd);
				break;
			}
			si = (++si == snum) ? 0 : si;
		}
	}
	gpuErrChk(cudaDeviceSynchronize());
	for (auto& _cs : cs) 
	{
		gpuErrChk(cudaStreamDestroy(_cs));
	}
	gpuErrChk(cudaFree(d_initBorder));
	gpuErrChk(cudaFree(d_mask));
	gpuErrChk(cudaFree(d_buffer));
	//gpuErrChk(cudaFree(d_bufModif));
	//gpuErrChk(cudaFree(d_bufReduc));
	gpuErrChk(cudaFreeHost(h_condition));
	gLogInfo << "\t\tDone\n";

}

extern "C"
void merge(char* const* d_in, char* d_out, const Merge & merge, const Dims & dim)
{

	constexpr int threads = 1 << 10;
	const int blocks = volume(dim) / threads;

	auto mergeFunc = [merge] __device__(const auto * a, const auto * b, const auto * c)
	{
		const int gid = threadIdx.x + blockIdx.x * blockDim.x;
		if (merge == Merge::SUM)
		{
			return a[gid] + b[gid] + c[gid];
		}
		else if (merge == Merge::MEAN)
		{
			return (a[gid] + b[gid] + c[gid]) / 3;
		}
	};

	device_3arrayFunctor<char, char, char, char> << <blocks, threads >> > (d_in[0], d_in[1], d_in[2], d_out, mergeFunc);

}

extern "C"
int2 FindRAIndex(char* d_inout, const Dims & dim)
{
	const int snum = 1 << 4;
	int pitch = dim.d[0] * dim.d[1];
	int num = pitch * dim.d[2];
	int raStart, raEnd, raRange = 0;
	int startIdx, endIdx;
	char* d_mask = nullptr;
	char* d_clone = nullptr;
	bool* h_condition = nullptr;
	
	int2* d_mnmx;
	int2* h_mnmx;

	std::vector<cudaStream_t> cs(snum);

	gLogInfo << "Set up Find RA index\n";
	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamCreateWithFlags(&_cs, 0));
	}

	gpuErrChk(cudaMalloc((void**)&d_clone, num * sizeof(char)));

	gpuErrChk(cudaMalloc((void**)&d_mask, num * sizeof(char)));
	gpuErrChk(cudaMallocHost((void**)&h_condition, dim.d[2] * sizeof(bool)));
	
	gpuErrChk(cudaMalloc((void**)&d_mnmx, sizeof(int2) * dim.d[2]));
	gpuErrChk(cudaMallocHost((void**)&h_mnmx, sizeof(int2) * dim.d[2]));

	
	gpuErrChk(cudaMemsetAsync(d_mnmx, 0, sizeof(int2) * dim.d[2], cs[2]));
	gpuErrChk(cudaMemcpyAsync(d_clone, d_inout, sizeof(char) * num, cudaMemcpyDeviceToDevice, cs[3]));
	
	
	auto boundIdx = [&](const char* in, int& start, int& end, const int i0, const int j0) ->void
	{
		gpuErrChk(cudaMemsetAsync(d_mask, 0, num * sizeof(char), cs[0]));
		gpuErrChk(cudaMemsetAsync(h_condition, false, sizeof(bool) * dim.d[2], cs[0]));
		gpuErrChk(cudaStreamSynchronize(cs[0]));
		
		start = -1;
		end = -1;

		for (int i = i0, stride = i0 * pitch, stride_r = j0 * pitch, j = j0;
			i < dim.d[2] && ((start == -1) || (end == -1));
			i++, stride += pitch, stride_r -= pitch, j--)
		{
			if (start == -1)
			{
				isModified<char>(in + stride, d_mask + stride, pitch, h_condition[i], cs[0]);
				if (h_condition[i])
					start = i;
			}
			if (end == -1)
			{
				isModified<char>(in + stride_r, d_mask + stride_r, pitch, h_condition[j], cs[1]);
				if (h_condition[j])
					end = j;
			}
		}

		gLogInfo << "startIdx : " << start << "\tendIdx : " << end << '\n';
	};
	boundIdx(d_inout, raStart, raEnd,0, dim.d[2] - 1);
	
	gpuErrChk(cudaStreamSynchronize(cs[2]));
	raRange = raEnd - raStart + 1;
	for (int d = raStart, si = 0, stride = pitch * raStart;
		d <= raEnd;
		d++, si = (++si == snum) ? 0 : si, stride += pitch)
	{
		device_verticThreshold << <dim.d[1], dim.d[0], sizeof(int2)* dim.d[0], cs[si] >> > (d_inout + stride, d_mnmx + d);
	}
	gpuErrChk(cudaDeviceSynchronize());
	cudaMemcpy(h_mnmx, d_mnmx, sizeof(int2) * dim.d[2], cudaMemcpyDeviceToHost);

	{
		int thd1 = 0, thd2 = 0;

		int rng = raRange * 0.2, deno1 = 0, deno2 = 0;
		for (int idx1 = raStart, idx2 = raEnd - rng;
			(idx1 < raStart + rng) || (idx2 < raEnd); idx1++, idx2++)
		{
			if (idx1 < raStart + rng)
			{
				thd1 += h_mnmx[idx1].y - h_mnmx[idx1].x;
				deno1++;
			}
			if (idx2 < raEnd)
			{
				thd2 += h_mnmx[idx2].y - h_mnmx[idx2].x;
				deno2++;
			}
		}
		thd1 /= deno1;
		thd2 /= deno2;
		gLogInfo << "thd1 : " << thd1<< "\tthd2 : " << thd2<< '\n';
		{
			auto threshFunc = [] __device__(const auto & data) -> char
			{
				return 0;
			};
			
			for (int idx1 = raStart, idx2 = raEnd - rng, si = 0;
				(idx1 < raStart + rng) || (idx2 < raEnd + 1); idx1++, idx2++)
			{
				if (idx1 < raStart + rng)
				{
					if ((h_mnmx[idx1].y - h_mnmx[idx1].x) <= thd1)
					{
						//const int stride = idx1 * pitch;
						//device_customFunctor<char, char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_inout + stride, d_mask + stride, threshFunc, 1);
						cudaMemsetAsync(d_clone + idx1 * pitch, 0, sizeof(char)* pitch, cs[si]);
						si = (++si == snum) ? 0 : si;
					}
				}

				if (idx2 < raEnd + 1)
				{
					if ((h_mnmx[idx2].y - h_mnmx[idx2].x) <= thd2)
					{
						//const int stride = idx2 * pitch;
						//device_customFunctor<char, char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_inout + stride, d_mask + stride, threshFunc, 1);
						cudaMemsetAsync(d_clone + idx2 * pitch, 0, sizeof(char)* pitch, cs[si]);
						si = (++si == snum) ? 0 : si;
					}
				}
			}
		}
		
		getLargestComponentChar(d_clone, dim);
	}

	{
		boundIdx(d_clone, startIdx, endIdx, 0, dim.d[2] - 1);

		int _startIdx, _endIdx;
		boundIdx(d_clone, _endIdx, _startIdx, endIdx, startIdx);
		startIdx = _max(raStart,_startIdx - 7);
		endIdx = _min(raEnd,_endIdx + 5);

		for (int idx1 = startIdx, idx2 = endIdx, si = 0;
			(idx1 >= 0) || (idx2 <= dim.d[2]); idx1--, idx2++)
		{
			if (idx1 >= 0)
			{
				const int stride = idx1 * pitch;
				cudaMemsetAsync(d_inout + idx1 * pitch, 0, sizeof(char)* pitch, cs[si]);
				si = (++si == snum) ? 0 : si;
			}
			if (idx2 <= dim.d[2])
			{
				cudaMemsetAsync(d_inout + idx2 * pitch, 0, sizeof(char)* pitch, cs[si]);
				si = (++si == snum) ? 0 : si;
			}
		}

	}
		
	gpuErrChk(cudaDeviceSynchronize());
	//cudaMemcpy(d_inout, d_clone, sizeof(char)* num, cudaMemcpyDeviceToDevice);

	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamDestroy(_cs));
	}
	gpuErrChk(cudaFree(d_mask));
	gpuErrChk(cudaFree(d_clone));
	gpuErrChk(cudaFree(d_mnmx));
	gpuErrChk(cudaFreeHost(h_condition));
	gpuErrChk(cudaFreeHost(h_mnmx));
	
	
	gLogInfo << "\t\tDone\n";
	return make_int2(startIdx, endIdx);
}

extern "C"
void ConvexRV(const char* d_ra, const char* d_rv, char* d_out, const int2 & idx, const Dims & dim)
{
	const int snum = 1 << 4;
	const int threads = 1 << 10;
	//const int w = dim.d[0], h = dim.d[1];
	int pitch = dim.d[0] * dim.d[1];
	int num = pitch * dim.d[2];
	
	std::vector<cudaStream_t> cs(snum);

	char* d_merge;
	char* d_boundary;
	char* d_buffer;
	auto logicalOr = []__device__(const char* x, const char* y) { 
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		return x[gid] | y[gid];
	};
	/*auto logicalAnd = []__device__(const char* x, const char* y) { 
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		return x[gid] & y[gid];
	};*/
	auto threshFunc = [] __device__(const char* x, const char* y) -> char
	{
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		if (x[gid] > 0)
			return 0;
		else
			return y[gid];
	};

	auto ero = []__device__(const char* in)
	{
		const int x = threadIdx.x;
		const int y = blockIdx.x;
		const int gid = x + y * blockDim.x;

		char now = in[gid];
		if (now != 0)
		{
//#pragma unroll (3)
			for (auto _x = -1; _x <= 1 && now; _x++)
			{
				const int X = x + _x;
				if (X > -1 && X < blockDim.x)
				{
//#pragma unroll (3)
					for (auto _y = -1; _y <= 1 && now; _y++)
					{
						const int Y = y + _y;
						if (Y > -1 && Y < gridDim.x)
						{
							if (in[X + Y * blockDim.x])
								continue;
							now = 0;
						}
						else
						{
							now = 0;
						}
					}
				}
				else
				{
					now = 0;
				}
			}
			//out[gid] = now;
		}
		return now;
	};

	gLogInfo << "Set up Convex RV\n";
	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamCreateWithFlags(&_cs, 0));
	}

	gpuErrChk(cudaMalloc((void**)&d_merge, num * sizeof(char)));
	gpuErrChk(cudaMalloc((void**)&d_boundary, num * sizeof(char)));
	gpuErrChk(cudaMalloc((void**)&d_buffer, num * sizeof(char)));
	
	gpuErrChk(cudaMemcpyAsync(d_out, d_ra, num * sizeof(char), cudaMemcpyDeviceToDevice, cs[2]));
	gpuErrChk(cudaStreamSynchronize(cs[2]));
	
	{
		auto h_structure = std::make_unique<char[]>(3 * 3 * 3);
		memset(h_structure.get(), 1, 3 * 3 * 3);
	
		//unsigned char* d_merge_ = nullptr;
		//unsigned char* d_boundary_ = nullptr;
		char* d_ra_in = nullptr;

		//gpuErrChk(cudaMalloc((void**)&d_merge_, num * sizeof(char)));
		//gpuErrChk(cudaMalloc((void**)&d_boundary_, num * sizeof(char)));
		gpuErrChk(cudaMalloc((void**)&d_ra_in, num * sizeof(char)));
		cudaMemcpyAsync(d_ra_in, d_ra, num * sizeof(char), cudaMemcpyDeviceToDevice, cs[2]);

		device_2arrayFunctor << <num / threads, threads, 0, cs[0] >> > (d_ra, d_rv, d_merge, logicalOr);
		//device_2arrayFunctor << <num / threads, threads, 0, cs[1] >> > (d_ra, d_rv, d_boundary, logicalAnd);
		//cudaMemcpyAsync(d_merge_, d_merge, num * sizeof(char), cudaMemcpyDeviceToDevice, cs[0]);
		//cudaMemcpyAsync(d_boundary_, d_boundary, num * sizeof(char), cudaMemcpyDeviceToDevice, cs[1]);
		
		
		/*call_binary_erosion(d_merge_, h_structure.get(), dim, 1, 2, cs[0]);*/
		call_binary_erosion(d_merge, h_structure.get(), dim, 1, 2, cs[0]);
		//call_binary_dilation(d_boundary_, h_structure.get(), dim, 1, 5, cs[1]);
		call_binary_erosion(d_ra_in, h_structure.get(), dim, 1, 2, cs[2]);

		//cudaMemcpyAsync(d_out, d_ra_in, num * sizeof(char), cudaMemcpyDeviceToDevice);

		//device_2arrayFunctor<char> << < num / threads, threads, 0, cs[0] >> > ((char*)d_merge_, d_merge, d_merge, threshFunc);
		//cudaMemcpyAsync(d_boundary, d_boundary_, num * sizeof(char), cudaMemcpyDeviceToDevice, cs[1]);
		
		device_2arrayFunctor << < num / threads, threads, 0, cs[2] >> > (d_ra_in, d_out, d_boundary, threshFunc);
		////gpuErrChk(cudaStreamSynchronize(cs[2]));
		//


		gpuErrChk(cudaStreamSynchronize(cs[0]));
		device_2arrayFunctor<char> << < num / threads, threads, 0, cs[0] >> > (d_merge, d_boundary, d_buffer, threshFunc);
		//gpuErrChk(cudaStreamSynchronize(cs[2]));
		//device_2arrayFunctor<char> << < num / threads, threads, 0, cs[1] >> > (d_merge, d_boundary, d_boundary, threshFunc);
		////gpuErrChk(cudaStreamSynchronize(cs[1]));
		//cudaMemcpyAsync(d_out, d_boundary, num * sizeof(char), cudaMemcpyDeviceToDevice,cs[0]);
		//gpuErrChk(cudaFree(d_merge_));
		//gpuErrChk(cudaFree(d_boundary_));
		gpuErrChk(cudaFree(d_ra_in));
	}
	cudaMemcpy(d_out, d_buffer, num * sizeof(char), cudaMemcpyDeviceToDevice);

	for (int i = 0, si = 0; i < 2; i++, si = (++si == snum) ? 0 : si)
	{
		const int i0 = idx.x + i;
		const int i1 = idx.y + (2 - i);

		//cudaMemcpyAsync(d_merge + i0 * pitch, d_out + i0 * pitch, sizeof(char)* pitch, cudaMemcpyDeviceToDevice, cs[si]);
		if (i0 > 0)
		{
			device_1arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_buffer + i0 * pitch, d_merge + i0 * pitch, ero);
			device_1arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_merge + i0 * pitch, d_boundary + i0 * pitch, ero);
			device_2arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_boundary + i0 * pitch, d_buffer + i0 * pitch, d_out + i0 * pitch, threshFunc);
			si++;
		}

		//cudaMemcpyAsync(d_boundary + i1 * pitch, d_out + i1 * pitch, sizeof(char)* pitch, cudaMemcpyDeviceToDevice, cs[si]);
		if (i1 < dim.d[2])
		{
			device_1arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_buffer + i1 * pitch, d_merge + i1 * pitch, ero);
			device_1arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_merge + i1 * pitch, d_boundary + i1 * pitch, ero);
			device_2arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_boundary + i1 * pitch, d_buffer + i1 * pitch, d_out + i1 * pitch, threshFunc);
		}
	}
	gpuErrChk(cudaDeviceSynchronize());
	gpuErrChk(cudaFree(d_merge));
	gpuErrChk(cudaFree(d_boundary));
	gpuErrChk(cudaFree(d_buffer));
	
	gLogInfo << "\t\tDone\n";
	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamDestroy(_cs));
	}
}

extern "C"
void ConvexRVAW(const char* d_rv_aw, const char* d_rv_nd, const char* d_lw, char* d_out, int& st_idx, const Dims & dim)
{
	constexpr int threads = 1 << 9;
	constexpr int snum = 1 << 4;
	std::vector<cudaStream_t> cs(snum);
	
	const int pitch = dim.d[0] * dim.d[1];
	const int num = pitch * dim.d[2];
	const int blocks = num / threads;
	const int _limit = 100;

	char* d_buffer = nullptr;
	unsigned int* iidx = nullptr;
	size_t* inum = nullptr;

	gpuErrChk(cudaMalloc((void**)&iidx, sizeof(unsigned int) * num));
	gpuErrChk(cudaMalloc((void**)&inum, sizeof(size_t) * dim.d[2]));
	gpuErrChk(cudaMalloc((void**)&d_buffer, sizeof(char) * num));

	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamCreateWithFlags(&_cs, cudaStreamNonBlocking));
	}

	auto close = []__device__(const char* in)
	{
		const int x = threadIdx.x;
		const int y = blockIdx.x;
		const int gid = x + y * blockDim.x;

		char now = in[gid];
		if (now == 0)
		{
#pragma unroll (3)
			for (int _x = -1; _x <= 1 && now == 0; _x++)
			{
				const int X = x + _x;
				if (X > -1 && X < blockDim.x)
				{
#pragma unroll (3)
					for (auto _y = -1; _y <= 1 && now == 0; _y++)
					{
						const int Y = y + _y;
						if (Y > -1 && Y < gridDim.x)
						{
							now |= in[X + Y * blockDim.x];
						}
					}
				}
			}
		}
		return in[gid] ^ now;
	};
	auto dil = []__device__(const char* in)
	{
		const int x = threadIdx.x;
		const int y = blockIdx.x;
		const int gid = x + y * blockDim.x;

		char now = in[gid];
		if (now == 0)
		{
#pragma unroll (3)
			for (int _x = -1; _x <= 1; _x++)
			{
				const int X = x + _x;
				if (X > -1 && X < blockDim.x)
				{
					if (now != 0) break;
#pragma unroll (3)
					for (int _y = -1; _y <= 1; _y++)
					{
						const int Y = y + _y;
						if (Y > -1 && Y < gridDim.x)
						{
							if (now != 0) break;

							now |= in[X + Y * blockDim.x];
						}
					}
				}
			}
			//out[gid] = now;
		}
		return now;
	};
	auto subConv = []__device__(const char* x, const char* y) -> char
	{
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		if (y[gid] > 0)
			return 0;
		return x[gid];
	};
	auto cusfunc = [_limit] __device__(const auto * rvaw, const auto * rvnd, const auto * count)
	{
		const int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
		char result = 0;
		if (count[blockIdx.y] < 100)
			result = rvnd[gid] + rvaw[gid];
		else
			result = rvaw[gid];
		return result > 0 ? 1 : result;
	};

	for (int d = 0, stride = 0, si = 0;
		d < dim.d[2];
		d++, si = (++si == snum) ? 0 : si, stride += pitch)
	{
		device_1arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_rv_aw + stride, d_out + stride, close);
	}
	gpuErrChk(cudaDeviceSynchronize());
	//char* d_buffer= nullptr;
	

	convexFill<char>(d_out, dim, cs.data());

	fill_holes(d_out, d_buffer, dim, Plane::XY);

	{
		const int structure_radius = 1;
		const int structSize = (structure_radius * 2 + 1) * (structure_radius * 2 + 1) * (structure_radius * 2 + 1);
		auto h_structure = std::make_unique<char[]>(structSize);
		memset(h_structure.get(), 1, structSize);
		call_binary_dilation(d_out, h_structure.get(), dim);
	}
	
	device_2arrayFunctor << <num / threads, threads>> > (d_rv_nd, d_out, d_buffer, subConv);

	gpuErrChk(cudaMemsetAsync(iidx, 0, sizeof(unsigned int)* num,cs[0]));
	gpuErrChk(cudaMemsetAsync(inum, 0, sizeof(size_t)* dim.d[2], cs[1]));


	extractIndex(d_buffer, iidx, inum, dim, cs.data());

	dim3 blocks2(dim.d[1], dim.d[2]);
	
	device_3arrayFunctor<char, char, char, size_t> << <blocks2, dim.d[0] >> > (d_rv_aw, d_rv_nd, inum, d_out, cusfunc);
	
	gpuErrChk(cudaMemcpy(d_buffer, d_lw, sizeof(char)* num, cudaMemcpyDeviceToDevice));

	getLargestComponentChar(d_buffer, dim);
	gpuErrChk(cudaMemsetAsync(iidx, 0, sizeof(unsigned int)* num, cs[0]));
	gpuErrChk(cudaMemsetAsync(inum, 0, sizeof(size_t)* dim.d[2], cs[1]));

	{
		size_t* h_inum = nullptr;
		gpuErrChk(cudaMallocHost((void**)&h_inum, sizeof(size_t)* dim.d[2]));
		extractIndex(d_buffer, iidx, inum, dim, cs.data());
		cudaMemcpy(h_inum, inum, sizeof(size_t)* dim.d[2], cudaMemcpyDeviceToHost);
		for (int i=0;i<dim.d[2];i++)
		{
			if (h_inum[i] > 0)
			{
				st_idx = i;
				break;
			}
		}
		gpuErrChk(cudaFreeHost(h_inum));
	}

	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamDestroy(_cs));
	}

	gpuErrChk(cudaFree(d_buffer));
	gpuErrChk(cudaFree(iidx));
	gpuErrChk(cudaFree(inum));
}

extern "C"
void ConvexRAAW(const char* d_rv_aw, const char* d_rv_nd, const char* d_ra_aw, char* d_out, char* d_out2, const int st_idx, const Dims & dim)
{
	constexpr int threads = 1 << 9;
	constexpr int snum = 1 << 4;
	std::vector<cudaStream_t> cs(snum);

	const int pitch = dim.d[0] * dim.d[1];
	const int num = pitch * dim.d[2];
	const int blocks = num / threads;
	const int _limit = 100;

	char* d_buffer = nullptr;
	char* d_buffer2 = nullptr;
	//char* d_buffer3 = nullptr;
	unsigned int* iidx = nullptr;
	size_t* inum = nullptr;
	
	dim3 blocks2(dim.d[1], dim.d[2]);

	gpuErrChk(cudaMalloc((void**)&iidx, sizeof(unsigned int) * num));
	gpuErrChk(cudaMalloc((void**)&inum, sizeof(size_t) * dim.d[2]));
	gpuErrChk(cudaMalloc((void**)&d_buffer, sizeof(char) * num));
	gpuErrChk(cudaMalloc((void**)&d_buffer2, sizeof(char) * num));
	//gpuErrChk(cudaMalloc((void**)&d_buffer3, sizeof(char) * num));

	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamCreateWithFlags(&_cs, cudaStreamNonBlocking));
	}

	auto close = []__device__(const char* in)
	{
		const int x = threadIdx.x;
		const int y = blockIdx.x;
		const int gid = x + y * blockDim.x;

		char now = in[gid];
		if (now == 0)
		{
#pragma unroll (3)
			for (int _x = -1; _x <= 1 && now == 0; _x++)
			{
				const int X = x + _x;
				if (X > -1 && X < blockDim.x)
				{
#pragma unroll (3)
					for (auto _y = -1; _y <= 1 && now == 0; _y++)
					{
						const int Y = y + _y;
						if (Y > -1 && Y < gridDim.x)
						{
							now |= in[X + Y * blockDim.x];
						}
					}
				}
			}
		}
		return in[gid] ^ now;
	};
	auto dil = []__device__(const char* in)
	{
		const int x = threadIdx.x;
		const int y = blockIdx.x;
		const int gid = x + y * blockDim.x;

		char now = in[gid];
		if (now == 0)
		{
#pragma unroll (3)
			for (int _x = -1; _x <= 1; _x++)
			{
				const int X = x + _x;
				if (X > -1 && X < blockDim.x)
				{
					if (now != 0) break;
#pragma unroll (3)
					for (int _y = -1; _y <= 1; _y++)
					{
						const int Y = y + _y;
						if (Y > -1 && Y < gridDim.x)
						{
							if (now != 0) break;

							now |= in[X + Y * blockDim.x];
						}
					}
				}
			}
			//out[gid] = now;
		}
		return now;
	};
	auto logicalOr = []__device__(const char* x, const char* y) {
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		return x[gid] | y[gid];
	};
	auto threshFunc = [] __device__(const char* x, const char* y) -> char
	{
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		if (x[gid] > 0)
			return 0;
		else
			return y[gid];
	};
	auto cusFunc1 = [st_idx] __device__(const char* x) -> char
	{
		const int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
		if (blockIdx.y < st_idx + 1)
			return 0;
		return x[gid];
	};
	auto cusFunc2 = [st_idx] __device__(const char* x, const char* y, const char* z) -> char
	{
		const int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
		if (blockIdx.y < st_idx + 1)
			return 0;
		if (z[gid] > 0)
			return 0;
		char result = x[gid] - y[gid];
		return result < 0 ? 0 : result;
	};

	for (int d = 0, stride = 0, si = 0;
		d < dim.d[2];
		d++, si = (++si == snum) ? 0 : si, stride += pitch)
	{
		device_1arrayFunctor<char> << < dim.d[1], dim.d[0], 0, cs[si] >> > (d_ra_aw + stride, d_out + stride, close);
	}
	gpuErrChk(cudaDeviceSynchronize());
	//char* d_buffer= nullptr;


	convexFill<char>(d_out, dim, cs.data());

	//buf: ra_conv_npy
	fill_holes(d_out, d_buffer, dim, Plane::XY);
	
	//gpuErrChk(cudaMemcpyAsync(d_buffer2, d_buffer, sizeof(char)* num, cudaMemcpyDeviceToDevice, cs[0]));

	device_2arrayFunctor << <num / threads, threads>> > (d_buffer, d_rv_nd, d_out, logicalOr);

	{
		const int structure_radius = 1;
		const int structSize = (structure_radius * 2 + 1) * (structure_radius * 2 + 1) * (structure_radius * 2 + 1);
		auto h_structure = std::make_unique<char[]>(structSize);
		memset(h_structure.get(), 1, structSize);
		gpuErrChk(cudaMemcpyAsync(d_out2, d_out, sizeof(char)* num, cudaMemcpyDeviceToDevice,cs[0]));

		call_binary_erosion(d_out2, h_structure.get(), dim, structure_radius, 2, cs[0]);
		
		//buf2: out
		//out2 : merge_in
		//out : merge
		device_2arrayFunctor << < num / threads, threads, 0, cs[0] >> > (d_out2, d_out, d_buffer2, threshFunc);


		call_binary_dilation(d_buffer, h_structure.get(), dim, structure_radius, 2, cs[1]);

		cudaStreamSynchronize(cs[0]);
		//buf1: out(ra_conv_npy)
		//buf2 : merge
		//out : ra_conv_npy
		device_2arrayFunctor << < num / threads, threads, 0, cs[1] >> > (d_out, d_buffer2, d_buffer, threshFunc);

		//rv_npy_erode
		gpuErrChk(cudaMemcpyAsync(d_buffer2, d_rv_nd, sizeof(char)* num, cudaMemcpyDeviceToDevice, cs[0]));
		call_binary_erosion(d_buffer2, h_structure.get(), dim, structure_radius, 1, cs[0]);
		
	}
	device_1arrayFunctor << <blocks2, dim.d[0], 0, cs[1] >> > (d_rv_nd, d_out2, cusFunc1);
	device_3arrayFunctor<char, char, char, char> << <blocks2, dim.d[0], 0, cs[0] >> > (d_rv_nd, d_buffer2, d_buffer, d_out, cusFunc2);
	//gpuErrChk(cudaMemcpy(d_out2, d_rv_nd, sizeof(char)* num, cudaMemcpyDeviceToDevice));
	getLargestComponentChar(d_out, dim);

	for (auto& _cs : cs)
	{
		gpuErrChk(cudaStreamDestroy(_cs));
	}

	gpuErrChk(cudaFree(d_buffer));
	cudaFree(iidx);
	cudaFree(inum);
	cudaFree(d_buffer);
	cudaFree(d_buffer2);
}

extern "C"
void gaussianBlurChar(char* d_inout, const float* h_kernel, const int shape_kernel_x, const int shape_kernel_y, const Dims & dim)
{
	//char* buffer;
	//float* d_kernel;
	//const int num = volume(dim);
	//const dim3 gridDim(dim.d[0] / 8 + 1, dim.d[1] / 8 + 1, dim.d[2] / 8 + 1);
	//const dim3 blockDim(8, 8, 8);

	//gpuErrChk(cudaMalloc((void**)&buffer, num * sizeof(char)));
	//gpuErrChk(cudaMemcpy(buffer, d_inout, num * sizeof(char), cudaMemcpyDeviceToDevice));
	//gpuErrChk(cudaMemcpy(d_kernel, h_kernel, shape_kernel_x * shape_kernel_y * sizeof(float), cudaMemcpyDeviceToDevice));

	//device_gaussianBlur<char> << <gridDim, blockDim >> > (buffer, d_inout, d_kernel, dim.d[0], dim.d[1], dim.d[2], 5, 5);

	//cudaFree(buffer);
	//cudaFree(d_kernel);
}
extern "C"
void gaussianBlurFloat(float* d_inout, const float* h_kernel, const int shape_kernel_x, const int shape_kernel_y, const Dims & dim)
{
	//float* buffer;
	//float* d_kernel;
	//const int num = volume(dim);
	//const dim3 gridDim(dim.d[0] / 8 + 1, dim.d[1] / 8 + 1, dim.d[2] / 8 + 1);
	//const dim3 blockDim(8, 8, 8);
	//
	//gpuErrChk(cudaMalloc((void**)&buffer, num * sizeof(float)));
	//gpuErrChk(cudaMemcpy(buffer, d_inout, num * sizeof(float), cudaMemcpyDeviceToDevice));
	//gpuErrChk(cudaMemcpy(d_kernel, h_kernel, shape_kernel_x * shape_kernel_y * sizeof(float), cudaMemcpyDeviceToDevice));
	//
	//device_gaussianBlur<float> << <gridDim, blockDim >> > (buffer, d_inout, d_kernel, dim.d[0], dim.d[1], dim.d[2], 5, 5);

	//cudaFree(buffer);
	//cudaFree(d_kernel);
}
extern "C"
void seperateLV(const char* d_voxel, const char* d_vmark, char* d_lv_chamber, char* d_lv_wallvolume, const Dims& dim)
{
	const int tnum = 1024;
	const int num = volume(dim);
	const int bytes = num * sizeof(char);

	dim3 threads(tnum);
	dim3 blocks(num / tnum);

	const float h_gaussianKernel[25] = { 
		0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625,
		0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625,
		0.0234375 , 0.09375   , 0.140625  , 0.09375   , 0.0234375,
		0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625,
		0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625 
	};

	float* d_gaussianKernel;
	char* d_temp;
	//char* d_blur;

	auto logicalAnd = []__device__(const char* x, const char* y) {
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		return x[gid] & y[gid];
	};

	float mean = getMarkedMean(d_voxel, d_vmark, num);
	float stdDev = getMarkedStdDevitation(d_voxel, d_vmark, num, mean);

	gpuErrChk(cudaMalloc((void**)&d_temp, bytes));
	//gpuErrChk(cudaMalloc((void**)&d_blur, bytes));
	gpuErrChk(cudaMalloc((void**)&d_gaussianKernel, 25 * sizeof(float)));
	gpuErrChk(cudaMemcpy(d_gaussianKernel, h_gaussianKernel, 25 * sizeof(float), cudaMemcpyHostToDevice));

	char h_struct333[27] = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };

	//char* d_structure333;
	//gpuErrChk(cudaMalloc((void**)&d_structure333, 27));
	//gpuErrChk(cudaMemcpy(d_structure333, h_struct333, 27, cudaMemcpyHostToDevice));

	

	// lv[voxel < (mean + 0.5*std)] = 0
	{
		
		float threshold = mean + (0.5 * stdDev);

		gpuErrChk(cudaMemcpy(d_temp, d_vmark, bytes, cudaMemcpyDeviceToDevice));
		device_seperateStep2 << <blocks, threads >> > (d_temp, d_voxel, threshold, num);
	}

	
	{
		dim3 gridDim(dim.d[0] / 8 + 1, dim.d[1] / 8 + 1, dim.d[2] / 8 + 1);
		dim3 blockDim(8, 8, 8);

		device_gaussianBlur<char> << <gridDim, blockDim >> > (d_temp, d_lv_chamber, d_gaussianKernel, dim.d[0], dim.d[1], dim.d[2], 5, 5);
	}
	//gaussianBlur<char>(d_temp, d_blur, d_gaussianKernel, shape_x, shape_y, shape_z, 5, 5);


	//char* d_largest;
	//gpuErrChk(cudaMalloc((void**)&d_largest, shape_x * shape_y * shape_z));

	getLargestComponentChar(d_lv_chamber, dim);

	//char* d_dilation;
	//gpuErrChk(cudaMalloc((void**)&d_dilation, bytes));

	call_binary_dilation(d_lv_chamber, h_struct333, dim, 1, 5);
	//binaryDilation(d_largest, d_structure333, d_dilation, shape_x, shape_y, shape_z, 1, 5);

	device_2arrayFunctor << <blocks, threads >> > (d_vmark, d_lv_chamber, d_lv_chamber, logicalAnd);
	gpuErrChk(cudaMemcpy(d_lv_wallvolume, d_vmark, bytes, cudaMemcpyDeviceToDevice));
	//gpuErrChk(cudaMemcpy(d_lv_chamber, d_dilation, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));
	//gpuErrChk(cudaMemcpy(d_lv_chamber, d_blur, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));
	//gpuErrChk(cudaMemcpy(d_lv_wallvolume, d_vmark, shape_x * shape_y * shape_z, cudaMemcpyDeviceToDevice));

	gpuErrChk(cudaFree(d_gaussianKernel));
	//gpuErrChk(cudaFree(d_structure333));
	gpuErrChk(cudaFree(d_temp));
	//gpuErrChk(cudaFree(d_blur));
	//gpuErrChk(cudaFree(d_largest));
	//gpuErrChk(cudaFree(d_dilation));
}

extern"C"
void generateLV_LVWALL(char* d_inout, const char* d_lv, const Dims & dim)
{
	char h_structBall777[343] = {
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
					1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
					1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
					1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
					0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
					0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
					1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
					1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
					1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float h_gaussianKernel[25] = {
		0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625,
		0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625,
		0.0234375 , 0.09375   , 0.140625  , 0.09375   , 0.0234375,
		0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625,
		0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625 };
	const int threads = 1 << 10;
	const int num = volume(dim);
	const dim3 grids(dim.d[0] / 8 + 1, dim.d[1] / 8 + 1, dim.d[2] / 8 + 1);
	const dim3 blocks(8, 8, 8);

	char* d_buffer;
	float* d_kernel;

	auto logicalOr = []__device__(const char* x, const char* y) {
		const int gid = threadIdx.x + blockDim.x * blockIdx.x;
		return x[gid] | y[gid];
	};

	gpuErrChk(cudaMalloc((void**)&d_buffer, num * sizeof(char)));
	gpuErrChk(cudaMalloc((void**)&d_kernel, 25* sizeof(float)));
	//gpuErrChk(cudaMemcpy(d_buffer, d_inout, num * sizeof(char), cudaMemcpyDeviceToDevice));
	gpuErrChk(cudaMemcpy(d_kernel, h_gaussianKernel, 25 * sizeof(float), cudaMemcpyHostToDevice));

	call_binary_erosion(d_inout, h_structBall777, dim, 3, 5);
	getLargestComponentChar(d_inout, dim);
	call_binary_dilation(d_inout, h_structBall777, dim, 3, 5);

	device_gaussianBlur<char> << <grids, blocks>> > (d_inout, d_buffer, d_kernel, dim.d[0], dim.d[1], dim.d[2], 5, 5);
	//gaussianBlurChar(d_inout, h_gaussianKernel, 5, 5, dim);

	device_2arrayFunctor << <num / threads, threads >> > (d_buffer, d_lv, d_inout, logicalOr);

	cudaFree(d_buffer);
	cudaFree(d_kernel);
}
extern "C"
void laLine(const char* d_ct_aw, const char* d_la_aw, const char* d_la_nd, const char* d_lvwall, char* d_la_chamber, char* d_la_line, const int shape_x, const int shape_y, const int shape_z)
{
	const int count = shape_x * shape_y * shape_z;

	char h_structOne333[27] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	char* d_structureOne333;
	gpuErrChk(cudaMalloc((void**)&d_structureOne333, 27));
	gpuErrChk(cudaMemcpy(d_structureOne333, h_structOne333, 27, cudaMemcpyHostToDevice));

	char* d_or;
	gpuErrChk(cudaMalloc((void**)&d_or, count));

	logicalOr(d_la_aw, d_la_nd, d_or, count);

	char* d_la_border;
	gpuErrChk(cudaMalloc((void**)&d_la_border, count));

	findBorder(d_ct_aw, d_or, d_la_border, shape_x, shape_y, shape_z);

	char* d_and;
	gpuErrChk(cudaMalloc((void**)&d_and, count));
	logicalAnd(d_or, d_la_border, d_and, count);

	gpuErrChk(cudaMemcpy(d_la_chamber, d_and, count, cudaMemcpyDeviceToDevice));

	char* d_dilation;
	gpuErrChk(cudaMalloc((void**)&d_dilation, count));
	binaryDilation(d_and, d_structureOne333, d_dilation, shape_x, shape_y, shape_z, 1, 1);

	char* d_tmp;
	gpuErrChk(cudaMalloc((void**)&d_tmp, count));

	{
		const int tnum = 1024;

		dim3 threads(tnum);
		dim3 blocks(count / tnum);

		device_la_line_tmp << <blocks, threads >> > (d_dilation, d_and, d_tmp, count);
	}

	char* d_transposeXZ;
	char* d_transposeYZ;
	gpuErrChk(cudaMalloc((void**)&d_transposeXZ, count));
	gpuErrChk(cudaMalloc((void**)&d_transposeYZ, count));

	{
		dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
		dim3 blockDim(8, 8, 8);

		device_transposeXZ << < gridDim, blockDim >> > (d_tmp, d_transposeXZ, shape_x, shape_y, shape_z);
		device_transposeYZ << <gridDim, blockDim >> > (d_tmp, d_transposeYZ, shape_x, shape_y, shape_z);
	}

	skeletonize(d_tmp, shape_x, shape_y, shape_z);
	skeletonize(d_transposeXZ, shape_z, shape_y, shape_x);
	skeletonize(d_transposeYZ, shape_x, shape_z, shape_y);

	char* d_sagi_border;
	char* d_coro_border;
	gpuErrChk(cudaMalloc((void**)&d_sagi_border, count));
	gpuErrChk(cudaMalloc((void**)&d_coro_border, count));

	{
		dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
		dim3 blockDim(8, 8, 8);

		device_transposeXZ << <gridDim, blockDim >> > (d_transposeXZ, d_sagi_border, shape_z, shape_y, shape_x);
		device_transposeYZ << <gridDim, blockDim >> > (d_transposeYZ, d_coro_border, shape_x, shape_z, shape_y);

		device_la_line_merge << <gridDim, blockDim >> > (d_tmp, d_sagi_border, d_coro_border, d_lvwall, d_la_line, shape_x, shape_y, shape_z);
	}
	gpuErrChk(cudaDeviceSynchronize());
	
	gpuErrChk(cudaFree(d_structureOne333));
	gpuErrChk(cudaFree(d_or));
	gpuErrChk(cudaFree(d_la_border));
	gpuErrChk(cudaFree(d_and));
	gpuErrChk(cudaFree(d_dilation));
	gpuErrChk(cudaFree(d_tmp));
	gpuErrChk(cudaFree(d_transposeXZ));
	gpuErrChk(cudaFree(d_transposeYZ));
	gpuErrChk(cudaFree(d_sagi_border));
	gpuErrChk(cudaFree(d_coro_border));
}

extern "C"
void exp_left_cardiac_tmp(const char* d_input, const char* d_mark, char* d_output, const int inputSize)
{
	const int tnum = 1024;

	dim3 threads(tnum);
	dim3 blocks(inputSize / tnum);

	device_exp_left_cardiac_tmp << <blocks, threads >> > (d_input, d_mark, d_output, inputSize);

	gpuErrChk(cudaDeviceSynchronize());
}

extern "C"
void arota_exp(char* inout, const Dims& dim)
{
	constexpr int threads = 1 << 10;

	char* d_buffer;
	//char* d_buffer2;
	const int num = volume(dim);
	const int blocks = num / threads;
	const int rIdx = dim.d[0] * dim.d[1] * int(dim.d[2] / 3);
	const int structure_radius = 1;
	const int structSize = (structure_radius * 2 + 1) * (structure_radius * 2 + 1) * (structure_radius * 2 + 1);
	auto h_structure = std::make_unique<char[]>(structSize);
	memset(h_structure.get(), 1, structSize);


	gpuErrChk(cudaMalloc((void**)&d_buffer, sizeof(char) * num));
	//gpuErrChk(cudaMalloc((void**)&d_buffer2, sizeof(char) * num));

	Dims _dim = dim;
	_dim.d[2] = int(_dim.d[2] / 3);
	//gpuErrChk(cudaMemset(d_buffer, 0, sizeof(char) * num));
	gpuErrChk(cudaMemcpy(d_buffer, inout, sizeof(char) * num, cudaMemcpyDeviceToDevice));

	//getLargestComponentChar(d_buffer, dim);
	auto bin = [rIdx]__device__(const char* in) ->unsigned char {
		const int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
		//if (rIdx <= gid)
		//	return 0;
		return (in[gid] == 0) ? 0 : 1;
	};
	//gpuErrChk(cudaMemset(inout, 0, sizeof(char)* num));
	device_1arrayFunctor<char> << <blocks, threads>> > (d_buffer, inout, bin);
	gpuErrChk(cudaDeviceSynchronize());
	getLargestComponentChar(inout, dim);
	gpuErrChk(cudaDeviceSynchronize());
	call_binary_erosion(inout, h_structure.get(), dim);
	gpuErrChk(cudaDeviceSynchronize());
	getLargestComponentChar(inout, dim);
	gpuErrChk(cudaDeviceSynchronize());
	call_binary_dilation(inout, h_structure.get(), dim);
	gpuErrChk(cudaDeviceSynchronize());
	getLargestComponentChar(inout, dim);
	gpuErrChk(cudaDeviceSynchronize());
	//gpuErrChk(cudaMemcpy(inout, d_buffer2, sizeof(char)* volume(dim), cudaMemcpyDeviceToDevice));
	//auto result = NumbericLabelPair(inout, dim);
	cudaFree(d_buffer);
}

void Histogram(const float* d_input, float* d_output, float* h_binOutput, unsigned int binSize, float rangeMin, float rangeMax, const unsigned int inputSize)
{
	float range = rangeMax - rangeMin;
	for (int i = 0; i < binSize - 1; i++)
	{
		h_binOutput[i] = rangeMin + ((range / (float)(binSize - 1)) * i);
	}
	h_binOutput[binSize - 1] = rangeMax;

	const int thNum = 1024;

	dim3 threads(thNum);
	dim3 blocks(inputSize / thNum + 1);

	float* d_bin;
	gpuErrChk(cudaMalloc((void**)&d_bin, inputSize * sizeof(float)));

	gpuErrChk(cudaMemcpy(d_bin, h_binOutput, binSize * sizeof(float), cudaMemcpyHostToDevice));

	device_histogram << <blocks, threads >> > (d_input, d_bin, d_output, inputSize, binSize);
	gpuErrChk(cudaDeviceSynchronize());

	gpuErrChk(cudaFree(d_bin));
}

//extern"C"
// flag 0: LALV, 1: Coronary [9/30/2021 Jeon]
void adwin(const float* d_hu, void* d_out, const int shape_x, const int shape_y, const int shape_z, const int histWing, const int flag)
{
	const int len = shape_x * shape_y * shape_z;

	dim3 gridDim(shape_x / 8 + 1, shape_y / 8 + 1, shape_z / 8 + 1);
	dim3 blockDim(8, 8, 8);

	float* d_temp;
	gpuErrChk(cudaMalloc((void**)&d_temp, len * sizeof(float)));

	device_transposeXY<float> << <gridDim, blockDim >> > (d_hu, d_temp, shape_y, shape_x, shape_z);

	float h_gaussianKernel[25] = { 0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902,
									0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621,
									0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823,
									0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621,
									0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902 };

	float* d_gaussianKernel;
	gpuErrChk(cudaMalloc((void**)&d_gaussianKernel, 25 * sizeof(float)));
	gpuErrChk(cudaMemcpy(d_gaussianKernel, h_gaussianKernel, 25 * sizeof(float), cudaMemcpyHostToDevice));

	float* d_blur;
	gpuErrChk(cudaMalloc((void**)&d_blur, len * sizeof(float)));

	gaussianBlur<float>(d_temp, d_blur, d_gaussianKernel, shape_x, shape_y, shape_z, 5, 5);

	//float* h_stack = new float[len];

	//gpuErrChk(cudaMemcpy(h_stack, d_blur, len * sizeof(float), cudaMemcpyDeviceToHost));

	float* d_vec;
	gpuErrChk(cudaMalloc((void**)&d_vec, shape_x * shape_y * sizeof(float)));
	gpuErrChk(cudaMemcpy(d_vec, &d_blur[int((shape_z / 2) * shape_x * shape_y)], shape_x * shape_y * sizeof(float), cudaMemcpyDeviceToDevice));

	//SAFE_DELETE_ARRAY(h_stack);

	unsigned char* d_label;
	gpuErrChk(cudaMalloc((void**)&d_label, shape_x * shape_y));

	const int thNum = 1024;
	{
		dim3 threads(thNum);
		dim3 blocks(shape_x * shape_y / thNum);

		device_ChkThreshold << <blocks, threads >> > (d_vec, d_label, -400, 1400, shape_x * shape_y);

		gpuErrChk(cudaDeviceSynchronize());
	}

	float* h_vec = new float[shape_x * shape_y];
	unsigned char* h_label = new unsigned char[shape_x * shape_y];

	gpuErrChk(cudaMemcpy(h_vec, d_vec, shape_x * shape_y * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrChk(cudaMemcpy(h_label, d_label, shape_x * shape_y, cudaMemcpyDeviceToHost));

	gpuErrChk(cudaFree(d_temp));
	gpuErrChk(cudaFree(d_gaussianKernel));
	gpuErrChk(cudaFree(d_vec));
	gpuErrChk(cudaFree(d_label));

	int nComp = 11;

	std::vector<float> vecVec;
	float lower = FLT_MAX;
	float upper = -FLT_MAX;
	for (int i = 0; i < shape_x * shape_y; i++)
	{
		if (h_label[i] != 0)
		{
			vecVec.emplace_back(h_vec[i]);

			if (h_vec[i] < lower)
				lower = h_vec[i];
			if (h_vec[i] > upper)
				upper = h_vec[i];
		}
	}

	SAFE_DELETE_ARRAY(h_vec);
	SAFE_DELETE_ARRAY(h_label);

	gpuErrChk(cudaMalloc((void**)&d_vec, vecVec.size() * sizeof(float)));
	gpuErrChk(cudaMemcpy(d_vec, &vecVec[0], vecVec.size() * sizeof(float), cudaMemcpyHostToDevice));

	unsigned char* d_kmeans;
	gpuErrChk(cudaMalloc((void**)&d_kmeans, vecVec.size()));

	KMean(d_vec, d_kmeans, lower, upper, nComp, vecVec.size());

	float* h_mu = new float[nComp];
	float* h_sigma = new float[nComp];
	float* h_prior = new float[nComp];

	float* d_lkh;
	float* d_pred;
	{
		dim3 threads(thNum);
		dim3 blocks(vecVec.size() / thNum + 1);

		float* d_sum;
		unsigned int* d_cnt;
		gpuErrChk(cudaMalloc((void**)&d_sum, nComp * sizeof(float)));
		gpuErrChk(cudaMalloc((void**)&d_cnt, nComp * sizeof(unsigned int)));
		gpuErrChk(cudaMemset(d_sum, 0, nComp * sizeof(float)));
		gpuErrChk(cudaMemset(d_cnt, 0, nComp * sizeof(unsigned int)));

		device_sumLabel << <blocks, threads >> > (d_vec, d_kmeans, d_sum, d_cnt, vecVec.size());

		gpuErrChk(cudaDeviceSynchronize());

		float* h_sum = new float[nComp];
		unsigned int* h_cnt = new unsigned int[nComp];
		gpuErrChk(cudaMemcpy(&h_sum[0], d_sum, nComp * sizeof(float), cudaMemcpyDeviceToHost));
		gpuErrChk(cudaMemcpy(&h_cnt[0], d_cnt, nComp * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		for (int i = 0; i < nComp; i++)
		{
			h_mu[i] = h_sum[i] / h_cnt[i];
		}

		float* d_mu;
		gpuErrChk(cudaMalloc((void**)&d_mu, nComp * sizeof(float)));
		gpuErrChk(cudaMemcpy(d_mu, h_mu, nComp * sizeof(float), cudaMemcpyHostToDevice));

		float* d_expdev;
		gpuErrChk(cudaMalloc((void**)&d_expdev, vecVec.size() * sizeof(float)));

		device_exp2dev << <blocks, threads >> > (d_vec, d_kmeans, d_mu, d_expdev, vecVec.size());

		float* d_sumExpdev;
		unsigned int* d_cntExpdev;
		gpuErrChk(cudaMalloc((void**)&d_sumExpdev, nComp * sizeof(float)));
		gpuErrChk(cudaMalloc((void**)&d_cntExpdev, nComp * sizeof(unsigned int)));
		gpuErrChk(cudaMemset(d_sumExpdev, 0, nComp * sizeof(float)));
		gpuErrChk(cudaMemset(d_cntExpdev, 0, nComp * sizeof(unsigned int)));

		device_sumLabel << <blocks, threads >> > (d_expdev, d_kmeans, d_sumExpdev, d_cntExpdev, vecVec.size());

		gpuErrChk(cudaDeviceSynchronize());

		float* h_sumExpdev = new float[nComp];
		unsigned int* h_cntExpdev = new unsigned int[nComp];

		gpuErrChk(cudaMemcpy(h_sumExpdev, d_sumExpdev, nComp * sizeof(float), cudaMemcpyDeviceToHost));
		gpuErrChk(cudaMemcpy(h_cntExpdev, d_cntExpdev, nComp * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		gpuErrChk(cudaMalloc((void**)&d_lkh, nComp * vecVec.size() * sizeof(float)));
		gpuErrChk(cudaMalloc((void**)&d_pred, vecVec.size() * sizeof(float)));
		gpuErrChk(cudaMemset(d_pred, 0, vecVec.size() * sizeof(float)));
		for (int i = 0; i < nComp; i++)
		{
			h_sigma[i] = sqrt(h_sumExpdev[i] / h_cntExpdev[i]);

			h_prior[i] = h_cnt[i] / (float)vecVec.size();

			device_gaussianDist << <blocks, threads >> > (d_vec, &d_lkh[i * vecVec.size()], d_pred, h_mu[i], h_sigma[i], h_prior[i], vecVec.size());

			gpuErrChk(cudaDeviceSynchronize());
		}

		gpuErrChk(cudaFree(d_sum));
		gpuErrChk(cudaFree(d_cnt));
		gpuErrChk(cudaFree(d_mu));
		gpuErrChk(cudaFree(d_expdev));
		gpuErrChk(cudaFree(d_sumExpdev));
		gpuErrChk(cudaFree(d_cntExpdev));

		SAFE_DELETE_ARRAY(h_sum);
		SAFE_DELETE_ARRAY(h_cnt);
		SAFE_DELETE_ARRAY(h_sumExpdev);
		SAFE_DELETE_ARRAY(h_cntExpdev);
	}

	float* d_clip;
	gpuErrChk(cudaMalloc((void**)&d_clip, vecVec.size() * sizeof(float)));

	{
		dim3 threads(thNum);
		dim3 blocks(vecVec.size() / thNum + 1);

		device_clip << <blocks, threads >> > (d_pred, d_clip, 1.e-32, FLT_MAX, vecVec.size());

		gpuErrChk(cudaDeviceSynchronize());
	}

	float* d_prior;
	gpuErrChk(cudaMalloc((void**)&d_prior, nComp * sizeof(float)));
	gpuErrChk(cudaMemcpy(d_prior, h_prior, nComp * sizeof(float), cudaMemcpyHostToDevice));

	float* d_post;
	float* d_prev;
	float* d_prevMu;
	float* d_muSumPost;
	float* d_muSumPostVec;
	float* d_sigmaSumPost;
	float* d_sigmaSumPostVec;
	float* d_mseSum;

	gpuErrChk(cudaMalloc((void**)&d_post, nComp * vecVec.size() * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_prev, vecVec.size() * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_prevMu, nComp * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_muSumPost, nComp * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_muSumPostVec, nComp * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_sigmaSumPost, nComp * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_sigmaSumPostVec, nComp * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&d_mseSum, sizeof(float)));

	float* h_muSumPost = new float[nComp];
	float* h_muSumPostVec = new float[nComp];
	float* h_sigmaSumPost = new float[nComp];
	float* h_sigmaSumPostVec = new float[nComp];

	int nIter = 0;
	while (nIter < 200)
	{
		dim3 threads(thNum, 1);
		dim3 blocks(vecVec.size() / thNum + 1, nComp);

		gpuErrChk(cudaMemcpy(d_pred, d_clip, vecVec.size() * sizeof(float), cudaMemcpyDeviceToDevice));

		device_post << <blocks, threads >> > (d_lkh, d_pred, d_prior, d_post, vecVec.size(), nComp);

		gpuErrChk(cudaMemcpy(d_prev, d_pred, vecVec.size() * sizeof(float), cudaMemcpyDeviceToDevice));
		gpuErrChk(cudaMemcpy(d_prevMu, h_mu, nComp * sizeof(float), cudaMemcpyHostToDevice));

		gpuErrChk(cudaMemset(d_pred, 0, vecVec.size() * sizeof(float)));
		gpuErrChk(cudaMemset(d_muSumPost, 0, nComp * sizeof(float)));
		gpuErrChk(cudaMemset(d_muSumPostVec, 0, nComp * sizeof(float)));
		gpuErrChk(cudaMemset(d_sigmaSumPost, 0, nComp * sizeof(float)));
		gpuErrChk(cudaMemset(d_sigmaSumPostVec, 0, nComp * sizeof(float)));

		device_muSum << <blocks, threads >> > (d_post, d_vec, d_muSumPost, d_muSumPostVec, vecVec.size(), nComp);
		device_sigmaSum << <blocks, threads >> > (d_post, d_vec, d_prevMu, d_sigmaSumPost, d_sigmaSumPostVec, vecVec.size(), nComp);

		gpuErrChk(cudaDeviceSynchronize());

		gpuErrChk(cudaMemcpy(h_muSumPost, d_muSumPost, nComp * sizeof(float), cudaMemcpyDeviceToHost));
		gpuErrChk(cudaMemcpy(h_muSumPostVec, d_muSumPostVec, nComp * sizeof(float), cudaMemcpyDeviceToHost));
		gpuErrChk(cudaMemcpy(h_sigmaSumPost, d_sigmaSumPost, nComp * sizeof(float), cudaMemcpyDeviceToHost));
		gpuErrChk(cudaMemcpy(h_sigmaSumPostVec, d_sigmaSumPostVec, nComp * sizeof(float), cudaMemcpyDeviceToHost));

		dim3 threads1(thNum);
		dim3 blocks1(vecVec.size() / thNum + 1);

		for (int i = 0; i < nComp; i++)
		{
			h_mu[i] = h_muSumPostVec[i] / h_muSumPost[i];
			h_sigma[i] = sqrt(h_sigmaSumPostVec[i] / h_sigmaSumPost[i]);

			h_prior[i] = h_muSumPost[i] / (float)vecVec.size();

			device_gaussianDist << <blocks1, threads1 >> > (d_vec, &d_lkh[i * vecVec.size()], d_pred, h_mu[i], h_sigma[i], h_prior[i], vecVec.size());
		}

		{
			device_clip << <blocks1, threads1 >> > (d_pred, d_clip, 1.e-32, FLT_MAX, vecVec.size());

			gpuErrChk(cudaMemset(d_mseSum, 0, sizeof(float)));
			device_mseSum << <blocks1, threads1 >> > (d_clip, d_prev, d_mseSum, vecVec.size());

			gpuErrChk(cudaDeviceSynchronize());

			float h_mseSum;
			gpuErrChk(cudaMemcpy(&h_mseSum, d_mseSum, sizeof(float), cudaMemcpyDeviceToHost));

			float h_mse = sqrtf(h_mseSum);

			if (h_mse < 1.e-4)
				break;
		}
		nIter++;
	}

	float* h_probs = new float[nComp];
	float maxProb = 0.;
	for (int i = 0; i < nComp; i++)
	{
		h_probs[i] = 1 / (sqrt(2 * M_PI * (h_sigma[i] * h_sigma[i])));
		if (h_probs[i] > maxProb)
			maxProb = h_probs[i];
	}

	int minI = INT_MAX, maxI = INT_MIN;
	int threshold = 0;
	int tempMin = INT_MAX, tempMax = INT_MIN;
	for (int i = 0; i < nComp; i++)
	{
		if (h_mu[i] < minI)
		{
			if (h_mu[i] > -4.e2 && h_sigma[i] < 5.e1 && h_probs[i] > 0.1 * maxProb)
			{
				minI = int(h_mu[i] - 3. * h_sigma[i] - histWing);
			}
			tempMin = h_mu[i] - 3.* h_sigma[i];
		}

		if (h_mu[i] > maxI)
		{
			if (h_mu[i] < 1.3e3 && h_sigma[i] < 1.e2 && h_probs[i] > 0.1 * maxProb)
			{
				maxI = int(h_mu[i] + 3. * h_sigma[i] + histWing);
				threshold = int(h_mu[i] - 3. * h_sigma[i]);
			}
			tempMax = h_mu[i] + 3.* h_sigma[i];
		}

		std::cout << "mu[" << i << "] : " << h_mu[i] << std::endl;
		std::cout << "sigma[" << i << "] : " << h_sigma[i] << std::endl;
	}

	if (minI == INT_MAX)
		minI = tempMin;
	if (maxI == INT_MIN)
		maxI = tempMax;

	// 	minI = -220;
	// 	maxI = 811;

	std::cout << "minI : " << minI << std::endl;
	std::cout << "maxI : " << maxI << std::endl;

	if (flag == 0) // Left Cardiac
	{
		dim3 threads(thNum);
		dim3 blocks(shape_x * shape_y * shape_z / thNum + 1);

		unsigned char* d_result;
		gpuErrChk(cudaMalloc((void**)&d_result, len));

		device_filter << <blocks, threads >> > (d_blur, d_result, minI, maxI, shape_x * shape_y * shape_z);

		gpuErrChk(cudaMemcpy((char*)d_out, d_result, len, cudaMemcpyDeviceToDevice));

		gpuErrChk(cudaDeviceSynchronize());

		gpuErrChk(cudaFree(d_result));
	}
	else if (flag == 1) // Coronary
	{
		const unsigned int binSize = 128;
		float bins[binSize];

		float* d_hist;
		gpuErrChk(cudaMalloc((void**)&d_hist, vecVec.size() * sizeof(float)));

		Histogram(d_vec, d_hist, bins, binSize, minI, maxI + 1, vecVec.size());

		float* h_hist = new float[vecVec.size()];
		gpuErrChk(cudaMemcpy(h_hist, d_hist, vecVec.size() * sizeof(float), cudaMemcpyDeviceToHost));

		gpuErrChk(cudaFree(d_hist));

		threshold = (255.0f * (threshold - bins[0]) / (bins[binSize - 1] - bins[0]));
		//threshold = 197;
		int minIdx = 0;
		float minValue = INT_MAX;// , maxValue = -minValue;
		for (int i = 0; i < binSize; i++)
		{
			auto val = fabsf(bins[i] - threshold);
			if (val < minValue)
			{
				minValue = val;
				minIdx = i;
			}
			//else
			//	maxValue = val;
		}
		float histSumPart = 0, histSumWhole = 0;
		//for (int i = 1; i < minIdx-1; i++)
		//	histSumPart += h_hist[i];
		//histSumWhole = histSumPart;
		//for (int i = minIdx; i < vecVec.size()-1; i++)
		//	histSumWhole += h_hist[i];
		histSumWhole = vecVec.size();
		for (int i = 0; i < vecVec.size(); i++)
		{
			if (h_hist[i] == 0 || h_hist[i] >= 127)
			{
				//printf("%f\n", h_hist[i]);
				histSumWhole--;
				continue;
			}
			if (h_hist[i] < minIdx)
				histSumPart++;
		}
		
		SAFE_DELETE_ARRAY(h_hist);

		float frac = (float)histSumPart / (float)histSumWhole;
		float slop0 = 255. * frac / (float)threshold;
		float slop1 = 255. * (1. - frac) / (255. - (float)threshold);

		{
			dim3 threads(thNum);
			dim3 blocks(shape_x * shape_y * shape_z / thNum + 1);

			float* d_filter;
			gpuErrChk(cudaMalloc((void**)&d_filter, shape_x * shape_y * shape_z * sizeof(float)));

			device_filter << <blocks, threads >> > (d_blur, d_filter, minI, maxI, shape_x * shape_y * shape_z);

			float* d_result;
			gpuErrChk(cudaMalloc((void**)&d_result, shape_x * shape_y * shape_z * sizeof(float)));

			device_piecewise << <blocks, threads >> > (d_filter, d_result, threshold, slop0, slop1, shape_x * shape_y * shape_z);

			gpuErrChk(cudaMemcpy((float*)d_out, d_result, len * sizeof(float), cudaMemcpyDeviceToDevice));

			gpuErrChk(cudaFree(d_filter));
			gpuErrChk(cudaFree(d_result));
		}
	}
	//gpuErrChk(cudaFree(d_temp));
	gpuErrChk(cudaFree(d_vec));
	gpuErrChk(cudaFree(d_kmeans));
	gpuErrChk(cudaFree(d_blur));

	gpuErrChk(cudaFree(d_lkh));
	gpuErrChk(cudaFree(d_pred));
	gpuErrChk(cudaFree(d_clip));
	gpuErrChk(cudaFree(d_prior));
	gpuErrChk(cudaFree(d_post));
	gpuErrChk(cudaFree(d_prev));
	gpuErrChk(cudaFree(d_prevMu));
	gpuErrChk(cudaFree(d_muSumPost));
	gpuErrChk(cudaFree(d_muSumPostVec));
	gpuErrChk(cudaFree(d_sigmaSumPost));
	gpuErrChk(cudaFree(d_sigmaSumPostVec));
	gpuErrChk(cudaFree(d_mseSum));


	SAFE_DELETE_ARRAY(h_mu);
	SAFE_DELETE_ARRAY(h_sigma);
	SAFE_DELETE_ARRAY(h_prior);
	SAFE_DELETE_ARRAY(h_muSumPost);
	SAFE_DELETE_ARRAY(h_muSumPostVec);
	SAFE_DELETE_ARRAY(h_sigmaSumPost);
	SAFE_DELETE_ARRAY(h_sigmaSumPostVec);
	SAFE_DELETE_ARRAY(h_probs);
}


extern "C"
void getInputs(const float* const h_in, void* const d_out, const Dims & dim, const int idx)
{
	const int threads = 1 << 10;
	const int num = volume(dim);
	float* buffer;
	float* d_temp;
	
	dim3 gridDim(dim.d[0] / 8 + 1, dim.d[1] / 8 + 1, dim.d[2] / 8 + 1);
	dim3 blockDim(8, 8, 8);

	auto clip = [=] __device__(const float val) ->char
	{
		if (val < 0) return 0;
		else if (val > 4095) return char(4095.0f / 16.0f);
		else return char(float(val) / 16.0f);
	};
	auto justSub = [=] __device__(const float val) ->float
	{
		return val - 1024.0f;
	};
	
	
	gpuErrChk(cudaMalloc((void**)&d_temp, num * sizeof(float)));
	gpuErrChk(cudaMalloc((void**)&buffer, num * sizeof(float)));

	gpuErrChk(cudaMemcpy(d_temp, h_in, num * sizeof(float), cudaMemcpyHostToDevice));
	
	


	if (idx == 0)
		device_customFunctor<float, char> << < num / threads, threads >> > (buffer, (char*)d_out, clip, 1);
	else
	{
		void* d_temp2;
		device_customFunctor<float, float> << < num / threads, threads >> > (d_temp, buffer, justSub, 1);
		cudaDeviceSynchronize();
		if (idx == 1)
		{
			gpuErrChk(cudaMalloc((void**)&d_temp2, num));
			adwin(buffer, d_temp2, dim.d[0], dim.d[1], dim.d[2], 0 , 0);
			cudaDeviceSynchronize();
			device_transposeXY<char> << <gridDim, blockDim >> > ((char*)d_temp2, (char*)d_out, dim.d[0], dim.d[1], dim.d[2]);
			cudaDeviceSynchronize();
		}
		else
		{
			gpuErrChk(cudaMalloc((void**)&d_temp2, num * sizeof(float)));
			adwin(buffer, d_temp2, dim.d[0], dim.d[1], dim.d[2], 50, 1);
			cudaDeviceSynchronize();
			device_transposeXY<float> << <gridDim, blockDim >> > ((float*)d_temp2, (float*)d_out, dim.d[0], dim.d[1], dim.d[2]);
			cudaDeviceSynchronize();
		}
		gpuErrChk(cudaFree(d_temp2));
	}

	gpuErrChk(cudaFree(d_temp));
	gpuErrChk(cudaFree(buffer));
}

extern "C"
void getPadded(const void* const d_in, float** const d_out, const Dims & dim, const int windowSize)
{
	const int stride = dim.d[0] * dim.d[1];
	const int num = dim.d[0] * dim.d[1] * (dim.d[2] + windowSize * 2);
	
	gpuErrChk(cudaMalloc((void**)d_out, sizeof(float) * num));
	
	dim3 blocks(dim.d[1], dim.d[2] + windowSize * 2);
	if(windowSize == 16)
		device_padded << <blocks, dim.d[0] >> > ((char*)d_in, *d_out, windowSize, dim.d[2]);
	else
		device_padded << <blocks, dim.d[0] >> > ((float*)d_in, *d_out, windowSize, dim.d[2]);
	

	cudaDeviceSynchronize();
}

extern "C"
void axial_copy(const int32_t * d_input, char* d_out, const Dims & dim)
{
	const int threads = 1 << 10;
	auto justRetype = [=] __device__(const int32_t val) ->unsigned char
	{
		return val;
	};
	device_customFunctor<int32_t,char> << <volume(dim) / threads, threads >> > (d_input, d_out, justRetype, 1);
}

extern "C"
void axial_copy2(const float * d_input, char* d_out, const Dims & dim)
{
	const int threads = 1 << 10;
	auto justRetype = [=] __device__(const float val) ->unsigned char
	{
		return val;
	};
	device_customFunctor<float, char> << <volume(dim) / threads, threads >> > (d_input, d_out, justRetype, 1);
}