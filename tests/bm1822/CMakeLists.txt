# tests/bm1822/CMakeLists.txt

cmake_minimum_required(VERSION 3.5)

# Add configuration option to control whether to use real hardware API implementation
option(BM1822_USE_REAL_IMPL "Use real hardware API implementation for BM1822 tests" OFF)

if(BM1822_USE_REAL_IMPL)
  message(STATUS "Building BM1822 tests with real hardware API implementation")
  add_definitions(-DBM1822_USE_REAL_IMPL)
else()
  message(STATUS "Building BM1822 tests with simulation implementation")
endif()

# Collect all test source files, test files start with "tiu_"
file(GLOB TEST_SOURCES "tiu_*.c")

# Print found test files
message(STATUS "Found the following bm1822 test files:")
foreach(src ${TEST_SOURCES})
    message(STATUS "  ${src}")
endforeach()

# Define function to add tests
function(add_cvikernel_test test_source)
    # Extract test name (remove path and extension)
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Define executable target name, e.g., tiu_add_bm1822_exe
    set(exe_name ${test_name}_exe)
    
    # Add executable file
    add_executable(${exe_name} ${test_source})
    
    # Link libraries
    if(BM1822_USE_REAL_IMPL)
        target_link_libraries(${exe_name} cvikernel m)
    else()
        target_link_libraries(${exe_name} m)
    endif()
    
    # Set runtime path (if UNIX system)
    if(UNIX)
        set_target_properties(${exe_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
    
    # Register test
    add_test(NAME ${test_name} COMMAND ${exe_name})
    
    # Set test timeout
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
    
    message(STATUS "Added bm1822 test: ${test_name}")
endfunction()

# Iterate through all test source files and add tests
foreach(test_src ${TEST_SOURCES})
    add_cvikernel_test(${test_src})
endforeach()

# Add a custom target to run all bm1822 tests
add_custom_target(run_bm1822_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -V -R "tiu_.*_bm1822"
    COMMENT "Running all bm1822 tests..."
) 