add_executable(matrix_mult)

target_link_libraries(matrix_mult PRIVATE OpenMP::OpenMP_CXX)

target_sources(matrix_mult
  PUBLIC
	matrix_multiplication.cpp
)

target_include_directories(matrix_mult
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )



# install(TARGETS matrix_mult RUNTIME DESTINATION bin)