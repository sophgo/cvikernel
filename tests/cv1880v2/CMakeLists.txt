# tests/cv1880v2/CMakeLists.txt

cmake_minimum_required(VERSION 3.5)

# 添加配置选项，控制是否使用真实硬件API实现
option(CV1880V2_USE_REAL_IMPL "Use real hardware API implementation for CV1880V2 tests" OFF)

if(CV1880V2_USE_REAL_IMPL)
  message(STATUS "Building CV1880V2 tests with real hardware API implementation")
  add_definitions(-DCV1880V2_USE_REAL_IMPL)
else()
  message(STATUS "Building CV1880V2 tests with simulated implementation")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# 获取所有以tiu_开头的测试源文件
file(GLOB CV1880V2_TEST_SOURCES "tiu_*.c")
message(STATUS "CV1880V2 Test sources: ${CV1880V2_TEST_SOURCES}")

# 添加测试函数
function(add_cvikernel_test test_file)
  get_filename_component(test_name ${test_file} NAME_WE)
  add_executable(${test_name} ${test_file})
  
  # 链接需要的库
  target_link_libraries(${test_name} m)
  
  # 不设置输出目录，使用默认目录
  # set_target_properties(${test_name} PROPERTIES
  #   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    
  # 注册测试
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# 使用循环添加所有测试
foreach(test_file ${CV1880V2_TEST_SOURCES})
  add_cvikernel_test(${test_file})
endforeach()

# 添加运行所有cv1880v2测试的目标
add_custom_target(run_cv1880v2_tests 
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS ${CV1880V2_TEST_TARGETS}
  COMMENT "Running all cv1880v2 tests...")

# 以下已由循环添加
# add_cvikernel_test(tiu_avgpool_cv1880v2.c)
# add_cvikernel_test(tiu_batchnorm_cv1880v2.c)
# add_cvikernel_test(tiu_relu_cv1880v2.c)
# add_cvikernel_test(tiu_softmax_cv1880v2.c)
# add_cvikernel_test(tiu_conv_cv1880v2.c)
# add_cvikernel_test(tiu_max_cv1880v2.c)
# add_cvikernel_test(tiu_maxpool_cv1880v2.c)
# add_cvikernel_test(tiu_min_cv1880v2.c)
# add_cvikernel_test(tiu_multiply_cv1880v2.c) 