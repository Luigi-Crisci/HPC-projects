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

void multiply_and_measure(Int_Matrix A, Int_Matrix B, std::function<Int_Matrix(Int_Matrix, Int_Matrix)> multiply_function, std::string test_name)
{
	std::cout << test_name << std::endl;
	Int_Matrix C;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	C = multiply_function(A, B);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time elapsed: " << (_Float64)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0 << " s" << std::endl;
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

	multiply_and_measure(A, B,multiply_base,"Basic Multiply");
	multiply_and_measure(A, B,multiply_interchange,"Basic Multiply with interchange");
	multiply_and_measure(A, B, multiply_parallel_for, "Parallel multiply");
	multiply_and_measure(A, B, multiply_tailing_parallel, "Parallel Tailed");
	multiply_and_measure(A, B, multiply_tailing_parallel_5_for, "Parallel Tailed with 5 for");

	return 0;
}
