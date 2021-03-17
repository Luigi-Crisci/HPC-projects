#include <iostream>
#include <vector>
#include <errno.h>
#include <random>
#include <chrono>
#include <functional>

typedef std::vector<std::vector<int64_t>> Int_Matrix;

void print_matrix(Int_Matrix M)
{
	for (std::vector<int64_t> row : M)
	{
		for (int64_t val : row)
			std::cout << val << " | ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void initialize_matrix(Int_Matrix *M)
{
	int row = (*M).size();
	int column = (*M)[0].size();

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			(*M)[i][j] = rand() % 9 + 1;
		}
	}
}

Int_Matrix multiply_base(Int_Matrix A, Int_Matrix B)
{
	int result_row_num = A.size();
	int result_column_num = B[0].size();

	Int_Matrix C(result_row_num, std::vector<int64_t>(result_column_num));

	int row = A.size();
	int column = B[0].size();
	int row_2 = B.size();

	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			for (int k = 0; k < row_2; k++)
				C[i][j] += A[i][k] * B[k][j];
	return C;
}

Int_Matrix multiply_interchange(Int_Matrix A, Int_Matrix B)
{
	int result_row_num = A.size();
	int result_column_num = B[0].size();

	Int_Matrix C(result_row_num, std::vector<int64_t>(result_column_num));

	int row = A.size();
	int column = B[0].size();
	int row_2 = B.size();

	for (int i = 0; i < row; i++)
		for (int k = 0; k < row_2; k++)
			for (int j = 0; j < column; j++)
				C[i][j] += A[i][k] * B[k][j];
	return C;
}

Int_Matrix multiply_parallel_for(Int_Matrix A, Int_Matrix B)
{
	int result_row_num = A.size();
	int result_column_num = B[0].size();

	Int_Matrix C(result_row_num, std::vector<int64_t>(result_column_num));

	int row = A.size();
	int column = B[0].size();
	int row_2 = B.size();

#pragma omp parallel for
	for (int i = 0; i < row; i++)
		for (int k = 0; k < row_2; k++)
			for (int j = 0; j < column; j++)
				C[i][j] += A[i][k] * B[k][j];
	return C;
}

Int_Matrix multiply_tailing_parallel(Int_Matrix A, Int_Matrix B)
{
	int result_row_num = A.size();
	int result_column_num = B[0].size();

	Int_Matrix C(result_row_num, std::vector<int64_t>(result_column_num));

	int N = A.size();
	int s = 128;

#pragma omp parallel for
	for (int i = 0; i < N; i += s)
#pragma omp parallel for
		for (int j = 0; j < N; j += s)
			for (int kn = 0; kn < N; kn += s)
				for (int in = 0; in < s; in++)
					for (int k = 0; k < s; k++)
						for (int jn = 0; jn < s; jn++)
							C[i + in][j + jn] += A[i + in][k + kn] * B[k + kn][j + jn];
	return C;
}

Int_Matrix matrix_sum(Int_Matrix A, Int_Matrix B)
{
	Int_Matrix C(A.size(), std::vector<int64_t>(A[0].size()));
	#pragma omp parallel for
	for (long unsigned int i = 0; i < A.size(); i++)
		for (long unsigned int j = 0; j < A[0].size(); j++)
			C[i][j] = A[i][j] + B[i][j];
	return C;
}


void matrix_sum_in_position(Int_Matrix A, Int_Matrix B, Int_Matrix result, int x_start,int y_start)
{
	#pragma omp parallel for
	for (long unsigned int i = 0; i < A.size(); i++)
		for (long unsigned int j = 0; j < A[0].size(); j++)
			C[i + x_start][j + y_start] = A[i][j] + B[i][j];
	return C;
}

/**
 * @brief Nightmare method. It will hunt you until it's thirst for blood is satisfied
 * 
 * @param A 
 * @param B 
 * @param C 
 * @param D 
 * @return Int_Matrix 
 */
Int_Matrix concatenate_matrix(Int_Matrix A, Int_Matrix B, Int_Matrix C, Int_Matrix D)
{
	Int_Matrix result(A.size() * 2, std::vector<int64_t>(A.size() * 2));

	//Copy first matrix
	#pragma omp task
	for (long unsigned int i = 0; i < A.size(); i++){
		for (long unsigned int j = 0; j < A.size(); j++)
			result[i][j] = A[i][j];
	}
	//Copy second matrix
	#pragma omp task
	for (long unsigned int i = 0; i < A.size(); i++){
		for (long unsigned int j = A.size(), j_i = 0; j_i < A.size(); j++, j_i++)
			result[i][j] = B[i][j_i];
	}
	//Copy third matrix
	#pragma omp task
	for (long unsigned int i = A.size(), i_i = 0; i_i < A.size(); i++, i_i++){
		for (long unsigned int j = 0; j < A.size(); j++)
			result[i][j] = C[i_i][j];
	}
	//Copy fourth matrix
	#pragma omp task
	for (long unsigned int i = A.size(), i_i = 0; i_i < A.size(); i++, i_i++){
		for (long unsigned int j = A.size(), j_i = 0; j_i < A.size(); j++, j_i++)
			result[i][j] = D[i_i][j_i];
	}

	#pragma omp taskwait
	return result;
}

Int_Matrix multiply_interchange(Int_Matrix A, Int_Matrix B, int A_row_start, int A_column_start,
								int B_row_start, int B_column_start, int size)
{
	Int_Matrix C(size, std::vector<int64_t>(size));
	#pragma omp parallel for
	for (int i = 0; i < size; i++)
		for (int A_k = A_column_start, B_k = B_row_start; A_k < A_column_start + size; A_k++, B_k++)
			for (int j = 0; j < size;  j++)
				C[i][j] += A[i + A_row_start][A_k] * B[B_k][j + B_column_start];
	return C;
}

Int_Matrix recursive_tiling_multiplication(Int_Matrix A, Int_Matrix B, int A_row_start, int A_column_start,
										   int B_row_start, int B_column_start, int size)
{
	if (size < 1024)
		return multiply_interchange(A, B, A_row_start, A_column_start, B_row_start, B_column_start, size);

	int diff = size / 2;

	Int_Matrix l_l_h, l_l_l, l_r_h, l_r_l; //Left matrix
	Int_Matrix r_l_h, r_l_l, r_r_h, r_r_l; //Right matrix
	Int_Matrix l_h, l_l, r_h, r_l; //Submatrixs

	//First matrix
	#pragma omp task shared(l_l_h) if (size < 1024) depend(out:l_l_h)
	l_l_h = recursive_tiling_multiplication(A, B, A_row_start, A_column_start, B_row_start, B_column_start, diff);
	#pragma omp task shared(l_l_l) if (size < 1024) depend(out:l_l_l)
	l_l_l = recursive_tiling_multiplication(A, B, A_row_start + diff, A_column_start, B_row_start, B_column_start, diff);
	#pragma omp task shared(l_r_h) if (size < 1024) depend(out:l_r_h)
	l_r_h = recursive_tiling_multiplication(A, B, A_row_start, A_column_start, B_row_start, B_column_start + diff, diff);
	#pragma omp task shared(l_r_l) if (size < 1024) depend(out:l_r_l)
	l_r_l = recursive_tiling_multiplication(A, B, A_row_start + diff, A_column_start, B_row_start, B_column_start + diff, diff);

	//Second matrix
	#pragma omp task shared(r_l_h) if (size < 1024) depend(out:r_l_h)
	r_l_h = recursive_tiling_multiplication(A, B, A_row_start, A_column_start + diff, B_row_start + diff, B_column_start, diff);
	#pragma omp task shared(r_l_l) if (size < 1024) depend(out:r_l_l) 
	r_l_l = recursive_tiling_multiplication(A, B, A_row_start + diff, A_column_start + diff, B_row_start + diff, B_column_start, diff);
	#pragma omp task shared(r_r_h) if (size < 1024) depend(out:r_r_h)
	r_r_h = recursive_tiling_multiplication(A, B, A_row_start, A_column_start + diff, B_row_start + diff, B_column_start + diff, diff);
	#pragma omp task shared(r_r_l) if (size < 1024) depend(out:r_r_l)
	r_r_l = recursive_tiling_multiplication(A, B, A_row_start + diff, A_column_start + diff, B_row_start + diff, B_column_start + diff, diff);

	#pragma omp task shared(l_h) depend(in: l_l_h,r_l_h) depend(out: l_h)
	l_h = matrix_sum(l_l_h, r_l_h);
	#pragma omp task shared(l_l) depend(in: l_l_l,r_l_l) depend(out: l_l)
	l_l = matrix_sum(l_l_l, r_l_l);
	#pragma omp task shared(r_h) depend(in: l_r_h,r_r_h) depend(out: r_h)
	r_h = matrix_sum(l_r_h, r_r_h);
	#pragma omp task shared(r_l) depend(in: l_r_l,r_r_l) depend(out: r_l)
	r_l = matrix_sum(l_r_l, r_r_l);
	
	#pragma omp taskwait

	//Create one matrix
	return concatenate_matrix(l_h, r_h, l_l, r_l);
}

Int_Matrix recursive_tiling_multiplication_full(Int_Matrix A, Int_Matrix B)
{
	return recursive_tiling_multiplication(A, B, 0, 0, 0, 0, A.size());
}

Int_Matrix multiply_tailing_parallel_5_for(Int_Matrix A, Int_Matrix B)
{
	int result_row_num = A.size();
	int result_column_num = B[0].size();

	Int_Matrix C(result_row_num, std::vector<int64_t>(result_column_num));

	int N = A.size();
	int s = 128;

#pragma omp parallel for
	for (int i = 0; i < N; i += s)
#pragma omp parallel for
		for (int j = 0; j < N; j += s)
			for (int in = 0; in < s; in++)
				for (int k = 0; k < N; k++)
					for (int jn = 0; jn < s; jn++)
						C[i + in][j + jn] += A[i + in][k] * B[k][j + jn];
	return C;
}

Int_Matrix multiply_and_measure(Int_Matrix A, Int_Matrix B, std::function<Int_Matrix(Int_Matrix, Int_Matrix)> multiply_function, std::string test_name)
{
	std::cout << test_name << std::endl;
	Int_Matrix C;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	#pragma omp parallel
	{
		#pragma omp single
		{
			C = multiply_function(A, B);
		}
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time elapsed: " << (_Float64)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0 << " s" << std::endl;
	return C;
}

bool equals(Int_Matrix A, Int_Matrix B)
{
	for (long unsigned int i = 0; i < A.size(); i++)
	{
		for (long unsigned int j = 0; j < A.size(); j++)
		{
			if (A[i][j] != B[i][j])
				return false;
		}
	}
	return true;
}

int main(int argc, char const *argv[])
{
	if (argc < 4)
	{
		fprintf(stderr, "Too few parameters\n");
		exit(EXIT_FAILURE);
	}

	int row = strtol(argv[1], NULL, 10);
	int column = strtol(argv[2], NULL, 10);
	int column_2 = strtol(argv[3], NULL, 10);

	if (errno == EINVAL)
	{
		fprintf(stderr, "Wrong parameters, please provide 3 integers\n");
		exit(EXIT_FAILURE);
	}

	if (row <= 0 || column <= 0 || column_2 <= 0)
	{
		fprintf(stderr, "All values must be greater than 0\n");
		exit(EXIT_FAILURE);
	}

	srand(time(NULL));
	Int_Matrix A(row, std::vector<int64_t>(column)),
		B(column, std::vector<int64_t>(column_2)),
		C;

	initialize_matrix(&A);
	initialize_matrix(&B);

	// print_matrix(A);
	// print_matrix(B);

	// multiply_and_measure(A, B, multiply_base, "Basic Multiply");
	// multiply_and_measure(A, B, multiply_interchange, "Basic Multiply with interchange");
	Int_Matrix C1 = multiply_and_measure(A, B, multiply_parallel_for, "Parallel multiply");
	// multiply_and_measure(A, B, multiply_tailing_parallel, "Parallel Tailed");
	Int_Matrix C2 = multiply_and_measure(A, B, recursive_tiling_multiplication_full, "Recursive tiling");
	// multiply_and_measure(A, B, multiply_tailing_parallel_5_for, "Parallel Tailed with 5 for");

	if (equals(C1, C2))
		std::cout << "They are the same" << std::endl;
	else
		std::cout << "They are not the same" << std::endl;

	return 0;
}
