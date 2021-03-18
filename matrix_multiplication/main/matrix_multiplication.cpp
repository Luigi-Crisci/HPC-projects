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

Int_Matrix* multiply_parallel_for(Int_Matrix *A, Int_Matrix *B)
{
	int result_row_num = A->size();
	int result_column_num = (*B)[0].size();

	Int_Matrix *C = new Int_Matrix(result_row_num, std::vector<int64_t>(result_column_num));

	int row = A->size();
	int column = (*B)[0].size();
	int row_2 = B->size();

#pragma omp parallel for
	for (int i = 0; i < row; i++)
		for (int k = 0; k < row_2; k++)
			for (int j = 0; j < column; j++)
				(*C)[i][j] += (*A)[i][k] * (*B)[k][j];
	return C;
}

Int_Matrix* multiply_tailing_parallel(Int_Matrix *A, Int_Matrix *B)
{
	int result_row_num = (*A).size();
	int result_column_num = (*B)[0].size();

	Int_Matrix* C = new Int_Matrix(result_row_num, std::vector<int64_t>(result_column_num));

	int N = (*A).size();
	int s = 128;

#pragma omp parallel for
	for (int i = 0; i < N; i += s)
#pragma omp parallel for
		for (int j = 0; j < N; j += s)
			for (int kn = 0; kn < N; kn += s)
				for (int in = 0; in < s; in++)
					for (int k = 0; k < s; k++)
						for (int jn = 0; jn < s; jn++)
							(*C)[i + in][j + jn] += (*A)[i + in][k + kn] * (*B)[k + kn][j + jn];
	return C;
}

void multiply_tailing_parallel_in_place(Int_Matrix *A, Int_Matrix *B,Int_Matrix* C, int A_row_start, int A_column_start,
								int B_row_start, int B_column_start, int size)
{
	int s = 64;
	int k_size = A_column_start + s;

#pragma omp parallel for
	for (int i = 0; i <size; i += s)
#pragma omp parallel for
		for (int j = 0; j < size; j += s)
			for (int kn = 0; kn < size; kn += s)
				for (int in = 0; in < s; in++)
					for (int A_k = A_column_start, B_k = B_row_start; A_k < k_size; A_k++, B_k++)
					for (int jn = 0; jn < s; jn++)
							(*C)[i + in + A_row_start][j + jn + B_column_start] += (*A)[i + in + A_row_start][kn + A_k] * (*B)[kn + B_k][j + jn + B_column_start];
}


void multiply_interchange_in_place(Int_Matrix *A, Int_Matrix *B,Int_Matrix* C, int A_row_start, int A_column_start,
								int B_row_start, int B_column_start, int size)
{
	int k_size = A_column_start + size;

	#pragma omp parallel for
	for (int i = 0; i < size; i++)
		for (int A_k = A_column_start, B_k = B_row_start; A_k < k_size; A_k++, B_k++)
			for (int j = 0; j < size;  j++)
				(*C)[i + A_row_start][j + B_column_start] += (*A)[i + A_row_start][A_k] * (*B)[B_k][j + B_column_start];
	
}

void recursive_tiling_multiplication(Int_Matrix *A, Int_Matrix *B,Int_Matrix* C ,int A_row_start, int A_column_start,
										   int B_row_start, int B_column_start, int size)
{
	if (size <= 256){
		multiply_tailing_parallel_in_place(A, B,C, A_row_start, A_column_start, B_row_start, B_column_start, size);
		return;
	}

	int diff = size / 2;
	int a,b,c;
	//First matrix
	#pragma omp task //depend(out: a)
	recursive_tiling_multiplication(A, B,C, A_row_start, A_column_start, B_row_start, B_column_start, diff);
	#pragma omp task //depend(out: b)
	recursive_tiling_multiplication(A, B,C, A_row_start + diff, A_column_start, B_row_start, B_column_start, diff);
	#pragma omp task //depend(out: c)
	recursive_tiling_multiplication(A, B,C, A_row_start, A_column_start, B_row_start, B_column_start + diff, diff);
	// #pragma omp task 
	recursive_tiling_multiplication(A, B,C, A_row_start + diff, A_column_start, B_row_start, B_column_start + diff, diff);

	#pragma omp taskwait

	//Second matrix
	#pragma omp task //depend(in: a)
	recursive_tiling_multiplication(A, B,C, A_row_start, A_column_start + diff, B_row_start + diff, B_column_start, diff);
	#pragma omp task //depend(in: b)
	recursive_tiling_multiplication(A, B,C, A_row_start + diff, A_column_start + diff, B_row_start + diff, B_column_start, diff);
	#pragma omp task //depend(in: c)
	recursive_tiling_multiplication(A, B,C, A_row_start, A_column_start + diff, B_row_start + diff, B_column_start + diff, diff);
	// #pragma omp task 
	recursive_tiling_multiplication(A, B,C, A_row_start + diff, A_column_start + diff, B_row_start + diff, B_column_start + diff, diff);
	
	#pragma omp taskwait
	
}

Int_Matrix* recursive_tiling_multiplication_full(Int_Matrix *A, Int_Matrix *B)
{
	Int_Matrix* C = new Int_Matrix(A->size(), std::vector<int64_t>(A->size()));

	#pragma omp parallel
	{
		#pragma omp single
		{
			recursive_tiling_multiplication(A, B, C, 0, 0, 0, 0, (*A).size());
		}
	}
	return C;
}

Int_Matrix* multiply_tailing_parallel_in_place_wrapper(Int_Matrix* A, Int_Matrix* B){
	Int_Matrix *C = new Int_Matrix(A->size(),std::vector<int64_t>(A->size()));
	multiply_tailing_parallel_in_place(A,B,C,0,0,0,0,A->size());
	return C;
}

Int_Matrix* multiply_and_measure(Int_Matrix *A, Int_Matrix *B, std::function<Int_Matrix*(Int_Matrix*, Int_Matrix*)> multiply_function, std::string test_name)
{
	std::cout << test_name << std::endl;
	Int_Matrix *C;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	C = multiply_function(A, B);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time elapsed: " << (_Float64)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0 << " s" << std::endl;
	return C;
}

bool equals(Int_Matrix *A, Int_Matrix *B)
{
	for (long unsigned int i = 0; i < A->size(); i++)
	{
		for (long unsigned int j = 0; j < A->size(); j++)
		{
			// std::cout<<"A["<<i<<"]["<<j<<"] = "<<A->at(i).at(j)<<" - B["<<i<<"]["<<j<<"] = "<<B->at(i).at(j) << std::endl;
			if (A->at(i).at(j) != B->at(i).at(j))
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
	// Int_Matrix C1 = multiply_and_measure(A, B, multiply_interchange, "Basic Multiply with interchange");
	// Int_Matrix* C1 = multiply_and_measure(&A, &B, multiply_parallel_for, "Parallel multiply");
	// Int_Matrix *C2 = new Int_Matrix(row,std::vector<int64_t>(row));
	// multiply_interchange_in_place(&A,&B,C2,0,0,0,0,row);
	Int_Matrix* C1 =multiply_and_measure(&A, &B, multiply_tailing_parallel, "Parallel Tailed");
	// Int_Matrix* C2 = multiply_and_measure(&A,&B, multiply_tailing_parallel_in_place_wrapper, "Parallel Tiling genralized");
	Int_Matrix* C2 = 
	multiply_and_measure(&A,&B, recursive_tiling_multiplication_full, "Recursive tiling");
	// multiply_and_measure(A, B, multiply_tailing_parallel_5_for, "Parallel Tailed with 5 for");

	if (equals(C1, C2))
		std::cout << "They are the same" << std::endl;
	else
		std::cout << "They are not the same" << std::endl;

	delete C1;
	delete C2;
	return 0;
}
