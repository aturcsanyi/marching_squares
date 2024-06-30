#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct matrix
{
    matrix(int width, int height);
    int& element(int row, int col);
    void print();

    std::vector<int> data;
    unsigned long int n_rows;
    unsigned long int n_cols;
};
matrix::matrix(int width, int height)
    :data(width*height, -1), n_rows(height), n_cols(width)
{}

int& matrix::element(int row, int col)
{
    return data[row*n_cols + col];
}

void matrix::print()
{
    for (int i = 0; i < data.size(); i++)
    {
        std::cout << data[i] << " ";
        if (not(i % n_cols - (n_cols-1)))
        {
            std::cout << std::endl;
        }
    }
    std::cout << "--------------------------------" << std::endl;
    return;
}

int corner_value(int row, int col, matrix& input, int threshold)
{
    int result = 0;
    if (input.element(row, col) > threshold) result |= 8;
    if (input.element(row, col+1) > threshold) result |= 4;
    if (input.element(row+1, col+1) > threshold) result |= 2;
    if (input.element(row+1, col) > threshold) result |= 1;

    return result;
}

matrix image_crop_mx(unsigned char* data, int image_width, int image_height, int top_left_x, int top_left_y, int matrix_width, int matrix_height)
{
    matrix result(matrix_width, matrix_height);
    for (int row = 0; row < matrix_height; ++row)
    {
        for (int col = 0; col < matrix_width; ++col)
        {
            int idx = (top_left_y + row)*image_width + top_left_x + col;
            result.element(row, col) = data[idx];
        }
    }
    return result;
}

matrix image_mx(unsigned char* data, int image_width, int image_height)
{
    matrix result(image_width, image_height);
    result.data.assign(data, data + image_width*image_height);
    return result;
}

matrix contour_values(matrix& input, int threshold)
{
    matrix output(input.n_cols-1, input.n_rows-1);
    for (int row = 0; row < input.n_rows-1; row++)
    {
        for (int col = 0; col < input.n_cols-1; col++)
        {
            output.element(row, col) = corner_value(row, col, input, threshold);
        }
    }
    return output;
}

int compare(matrix& mx1, matrix& mx2)
{
    int mismatch_count = 0;
    for (auto it1 = mx1.data.begin(), it2 = mx2.data.begin(); it1 != mx1.data.end() && it2 != mx2.data.end(); ++it1, ++it2)
    {
        if (*it1 != *it2)
        {
            ++mismatch_count;
        }
    }
    return mismatch_count;
}


int main()
{
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.size() == 0){ std::cout << "No OpenCL platform detected\n"; return -1; }

        cl::Platform selected_platform{};
        cl::Device selected_device{};
        bool found_gpu_device = false;
        for(std::size_t i = 0; i<platforms.size(); ++i)
        {
            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);

            if(devices.size() == 0){ continue; }

            for(std::size_t j = 0; j<devices.size(); ++j)
            {
                // pick first device:
                if(j == 0)
                {
                    selected_platform = platforms[i];
                    selected_device = devices[j];
                    found_gpu_device = true;
                }
            }

            // skip other platforms if a GPU device is found:
            if(found_gpu_device){ break; }
        }

        std::cout << "Selected platform vendor: " << selected_platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
        std::cout << "Selected platform name:   " << selected_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cout << "Selected device name:     " << selected_device.getInfo<CL_DEVICE_NAME>() << "\n";

        // Actual program logic: create context and command queue:
        std::vector<cl_context_properties>cps{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(selected_platform()), 0};
        cl::Context context{ selected_device, cps.data() };
        // Enable profiling on the queue:
        cl::QueueProperties qps{cl::QueueProperties::Profiling};
        cl::CommandQueue queue{ context, selected_device, qps};

        // Load and compile kernel program:
        std::ifstream source{"./kernel.cl"};
        if( !source.is_open() ){ throw std::runtime_error{ std::string{"Error opening kernel file: kernel.cl"} }; }
        std::string source_string{ std::istreambuf_iterator<char>{ source },
                                   std::istreambuf_iterator<char>{} };
        cl::Program program{ context, source_string };
        program.build({selected_device});
        
        auto kernel_marching_squares = cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_int, cl::Buffer>(program, "marching_squares");
        auto kernel_draw_lines = cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl::Buffer>(program, "draw_lines");

        // Allocate and setup data buffers:

        // load image

        int x,y,n;
        unsigned char *data = stbi_load("../SicilyUTM.png", &x, &y, &n, 1 /*forcing 1 channel*/);
        if(!data)
        {
            std::cout << "Error: could not open input file\n";
            return -1;
        }
        else
        {
            std::cout << "Image opened successfully. Width x Height x Components = " << x << " x " << y << " x " << n << "\n";
        }
        auto input = image_crop_mx(data, x, y, x/2, y/2, 200, 200);
        //input.print();
        //auto input = image_mx(data, x, y);

        int level = 40;
        std::cout << "starting computation on input matrix (size: " << input.n_cols << "*" << input.n_rows << ")\n";

        // cpu implementation
        float dt0 = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto cpu_output = contour_values(input, level);
        auto t1 = std::chrono::high_resolution_clock::now();
        dt0 = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f;
        std::cout << "cpu version took " << dt0 << " ms\n";

        //cpu_output.print()

        // gpu

        matrix gpu_output(input.n_cols-1, input.n_rows-1);

        cl::Buffer buffer_input{ context, std::begin(input.data), std::end(input.data), true };  // true: read-only
        cl::Buffer buffer_output{ context, std::begin(gpu_output.data), std::end(gpu_output.data), false }; // false: read-write

        // Launch kernel:
        cl::NDRange global_threads = {gpu_output.n_cols, gpu_output.n_rows};
        cl::Event ev = kernel_marching_squares(cl::EnqueueArgs{queue, global_threads}, buffer_input, gpu_output.n_cols, gpu_output.n_rows, level, buffer_output);
        
        // Copy back results:
        cl::copy(queue, buffer_output, std::begin(gpu_output.data), std::end(gpu_output.data));

        int output_im_width = (gpu_output.n_cols)*5;
        int output_im_height = (gpu_output.n_rows)*5;

        cl::Buffer output{ context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, output_im_width * output_im_height * sizeof(unsigned char), };
        cl::Event ev2 = kernel_draw_lines(cl::EnqueueArgs{queue, global_threads}, buffer_output, gpu_output.n_cols, gpu_output.n_rows, output);

        std::vector<unsigned char> image(output_im_width*output_im_height);
        cl::copy(queue, output, std::begin(image), std::end(image));

        // Synchronize:
        queue.finish();

        int success = stbi_write_jpg("output.jpg", output_im_width, output_im_height, 1, &image[0], 100);
        
        float dt1 = 0.0f;
        cl_ulong t1_0 = ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        dt1 = (t1_0)*0.001f*0.001f;
        std::cout << "gpu version took " << dt1 << " ms\n";

        //std::cout << "gpu result:" << std::endl;
        //gpu_output.print();

        // Verify results:

        int diff = compare(cpu_output, gpu_output);

        if (not(diff))
        {
            std::cout << "SUCCESS: GPU result matches CPU reference\n";
        }
        else
        {
            std::cout << "FAILURE: GPU result does not match CPU reference, there were " << diff << "mismatches\n";
        }
        stbi_image_free(data);
    }
    catch(cl::BuildError& error) // If kernel failed to build
    {
        std::cout << "Build failed. Log:\n";
        for (const auto& log : error.getBuildLog())
        {
            std::cout << log.second;
        }
        return -1;
    }
    catch(cl::Error& e)
    {
        std::cout << "OpenCL error: " << e.what() << "\n";
        return -1;
    }
    catch(std::exception& e)
    {
        std::cerr << "C++ STL Error: " << e.what() << "\n";
        return -1;
    }
    
    return 0;
}