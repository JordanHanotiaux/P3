#include "matrix_opencl.hpp"
#include "mlp_sgd.cpp" // Note: Including .cpp is generally discouraged, prefer linking .o files. Included here to match original structure.
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cmath>   // For std::exp, std::log
#include <limits> // For std::numeric_limits

// Helper function to print a matrix (copies to host first)
void printMatrix(const std::string& label, const MatrixCL& mat) {
    std::cout << label << " (" << mat.numRows() << "x" << mat.numCols() << "):\n";
    try {
        std::vector<float> host_data = mat.copyToHost();
        for (int i = 0; i < mat.numRows(); ++i) {
            std::cout << "  [";
            for (int j = 0; j < mat.numCols(); ++j) {
                std::cout << " " << host_data[i * mat.numCols() + j];
            }
            std::cout << " ]\n";
        }
         std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error printing matrix: " << e.what() << std::endl;
    }
}

// Helper function for approximate float comparison
bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Helper function to verify matrix contents
bool verifyMatrix(const std::string& label, const MatrixCL& mat, const std::vector<float>& expected, float epsilon = 1e-5f) {
    std::cout << "Verifying " << label << "..." << std::endl;
    if (static_cast<size_t>(mat.numRows() * mat.numCols()) != expected.size()) {
        std::cerr << "Verification failed: Dimension mismatch for " << label << ". Got "
                  << mat.numRows() << "x" << mat.numCols() << ", expected " << expected.size() << " elements." << std::endl;
        return false;
    }
    try {
        std::vector<float> actual = mat.copyToHost();
        bool match = true;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!approxEqual(actual[i], expected[i], epsilon)) {
                std::cerr << "Verification failed for " << label << " at index " << i
                          << ". Got " << actual[i] << ", expected " << expected[i] << std::endl;
                match = false;
                // Don't break, report all mismatches if desired, or break here for efficiency
                 break;
            }
        }
        if (match) {
            std::cout << label << " verified successfully." << std::endl;
        } else {
             std::cout << label << " verification failed." << std::endl;
        }
        return match;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error verifying matrix " << label << ": " << e.what() << std::endl;
        return false;
    }
}


int main() {
    try {
        // 1. --- OpenCL Setup ---
        std::cout << "--- OpenCL Setup ---" << std::endl;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found." << std::endl;
            return 1;
        }
        cl::Platform platform = platforms.front();
        std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cout << "No GPU found, trying CPU..." << std::endl;
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty()) {
                std::cerr << "No OpenCL devices found." << std::endl;
                return 1;
            }
        }
        cl::Device device = devices.front();
        std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Context context(device);
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE); // Keep profiling enabled

        std::vector<cl::Device> devices_to_init = {device};
        try {
            MatrixCL::initializeKernels(context, devices_to_init);
            std::cout << "Kernel initialization successful." << std::endl;
        } catch (const std::exception& e) {
            // Catching std::exception here because initializeKernels wraps cl::Error
            std::cerr << "FATAL ERROR during kernel initialization: " << e.what() << std::endl;
            // If the error was a BuildError, the log should have been printed
            // by the loadAndBuildProgram function within initializeKernels.
            return 1;
        }


    } catch (const cl::BuildError& err) { // Catch specific build error first
        std::cerr << "OpenCL Build Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog()) {
            std::cerr << "Device " << pair.first.getInfo<CL_DEVICE_NAME>() << " Build Log:" << std::endl;
            std::cerr << pair.second << std::endl;
        }
        return 1;
    } catch (const cl::Error& err) { // Catch other OpenCL errors
        std::cerr << "OpenCL Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) { // Catch standard exceptions
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll OpenCL Matrix and MLP tests completed successfully." << std::endl;
    return 0;
}