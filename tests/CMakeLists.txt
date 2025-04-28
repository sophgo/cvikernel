# tests/CMakeLists.txt

# 添加子目录 cv180x，让其中的 CMakeLists.txt 处理该目录下的测试文件
add_subdirectory(cv180x)

# 添加子目录 cv1880v2，让其中的 CMakeLists.txt 处理该目录下的测试文件
add_subdirectory(cv1880v2)

# 添加子目录 bm1822，让其中的 CMakeLists.txt 处理该目录下的测试文件
add_subdirectory(bm1822)

# 添加一个运行所有测试的自定义目标
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -V
    COMMENT "运行所有测试..."
) 