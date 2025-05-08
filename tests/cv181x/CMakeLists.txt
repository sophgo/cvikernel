# tests/cv181x/CMakeLists.txt

cmake_minimum_required(VERSION 3.5)

# 添加配置选项，控制是否使用真实硬件API实现
option(CV181X_USE_REAL_IMPL "Use real hardware API implementation for CV181X tests" OFF)

if(CV181X_USE_REAL_IMPL)
  message(STATUS "Building CV181X tests with real hardware API implementation")
  add_definitions(-DCV181X_USE_REAL_IMPL)
else()
  message(STATUS "Building CV181X tests with simulated implementation")
endif()

# 收集所有测试源文件，测试文件名以tiu_开头
file(GLOB TEST_SOURCES "tiu_*.c")

# 打印找到的测试文件
message(STATUS "找到以下cv181x测试文件:")
foreach(src ${TEST_SOURCES})
    message(STATUS "  ${src}")
endforeach()

# 定义添加测试的函数
function(add_cvikernel_test test_source)
    # 提取测试名称（去除路径和扩展名）
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # 定义可执行目标名称，例如 tiu_add_cv181x_exe
    set(exe_name ${test_name}_exe)
    
    # 添加可执行文件
    add_executable(${exe_name} ${test_source})
    
    # 链接库
    if(CV181X_USE_REAL_IMPL)
        target_link_libraries(${exe_name} cvikernel m)
    else()
        target_link_libraries(${exe_name} m)
    endif()
    
    # 设置运行时路径（如果是UNIX系统）
    if(UNIX)
        set_target_properties(${exe_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
    
    # 注册测试
    add_test(NAME ${test_name} COMMAND ${exe_name})
    
    # 设置测试超时
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
    
    message(STATUS "添加cv181x测试: ${test_name}")
endfunction()

# 遍历所有测试源文件并添加测试
foreach(test_src ${TEST_SOURCES})
    add_cvikernel_test(${test_src})
endforeach()

# 添加一个运行cv181x所有测试的自定义目标
add_custom_target(run_cv181x_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -V -R "tiu_.*_cv181x"
    COMMENT "运行所有cv181x测试..."
)

