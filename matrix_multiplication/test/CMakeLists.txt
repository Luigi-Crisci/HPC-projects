include(FetchContent)
FetchContent_Declare(gtest
  QUIET
  URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
)
# configure build of googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gtest)

# include_directories (${CMAKE_BINARY_DIR}/main) 
target_link_libraries(matrix_mult PRIVATE OpenMP::OpenMP_CXX)


add_executable(unit_tests
	test_matrix.cpp
)

target_link_libraries(
	unit_tests
	PRIVATE
		gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests
  PROPERTIES
    LABELS "unit"
  DISCOVERY_TIMEOUT  # how long to wait (in seconds) before crashing
    240
  )