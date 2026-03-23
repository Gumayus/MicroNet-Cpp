cmake_minimum_required(VERSION 3.10)
project(unet_verify)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 寻找 OpenCV
find_package(OpenCV REQUIRED)


include_directories(
    .                    
    ../                  
    ${OpenCV_INCLUDE_DIRS} # OpenCV 的路径
)

# 3. 编译目标
add_executable(unet_run main.cpp)

# 4. 链接库
target_link_libraries(unet_run ${OpenCV_LIBRARIES})
