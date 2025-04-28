#!/bin/bash
# This script is used to verify bm1822 test files #verify bm1822 test script

# Set test directory
TEST_DIR=$(dirname $0)
BUILD_DIR="$TEST_DIR/../../build"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No color

# Record failed tests
FAILED_TESTS=()

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p "$BUILD_DIR"
fi

# Change to build directory
cd "$BUILD_DIR"

# Run CMAKE configuration if not built
if [ ! -f "CMakeCache.txt" ]; then
    echo -e "${YELLOW}Running CMAKE configuration...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Debug
fi

# Build all bm1822 tests
echo -e "${YELLOW}Building bm1822 tests...${NC}"
make -j4

# Run all built tests
echo -e "${YELLOW}Running bm1822 tests...${NC}"

# Find all bm1822 test executables
TEST_EXES=$(find "$BUILD_DIR/tests/bm1822" -name "tiu_*_bm1822_exe" -type f)

# For each test, run and check the result
for test in $TEST_EXES; do
    test_name=$(basename $test)
    echo -e "${YELLOW}Running test: $test_name ${NC}"
    
    # Run the test and capture output and return code
    output=$($test 2>&1)
    return_code=$?
    
    # Check if the test passed
    if [ $return_code -eq 0 ] && [[ $output == *"All tests passed"* ]]; then
        echo -e "${GREEN}Test passed: $test_name ${NC}"
    else
        echo -e "${RED}Test failed: $test_name ${NC}"
        echo "$output"
        FAILED_TESTS+=("$test_name")
    fi
done

# Report results
echo ""
echo -e "${YELLOW}Test summary:${NC}"
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Failed tests: ${#FAILED_TESTS[@]}${NC}"
    for failed in "${FAILED_TESTS[@]}"; do
        echo -e "${RED}- $failed${NC}"
    done
    exit 1
fi 