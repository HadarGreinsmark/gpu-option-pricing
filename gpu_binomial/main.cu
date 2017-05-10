#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using namespace std;

#define check_err(cu_err) { cu_err_handler(cu_err, __FILE__, __LINE__); }

inline void cu_err_handler(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU error: '%s' at %s:%d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

//////// CPU dynamic programming implementation ////////

double cpu_dynprog_binomial_american_put(
		double stock_price,
		double strike_price,
		double expire,
		double volat,
		int num_steps,
		double risk_free_rate) {
	using namespace std;

	double dt = expire / num_steps;
	double up_factor = exp(volat * sqrt(dt));
	double down_factor = 1 / up_factor;
	double R = exp(risk_free_rate * dt);
	double up_prob = (R - down_factor) / (up_factor - down_factor);
	double *tree = new double[num_steps + 1];

	// Initialize end of tree at expire time
	for (int step = 0; step <= num_steps; ++step) {
		// Option value when exercising the option
		double exercise = strike_price - stock_price * pow(up_factor, 2 * step - num_steps);
		tree[step] = max(exercise, .0);
	}

	for (int step = num_steps - 1; step >= 0; --step) {
		for (int branch = 0; branch <= step; ++branch) {
			double binomial = 1 / R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
			double exercise = strike_price - stock_price * pow(up_factor, (double) 2 * branch - step);
			tree[branch] = max(binomial, exercise);
		}
	}

	double price = tree[0];
	delete[] tree;
	return price;
}

//////// GPU Reduce tree ////////

__global__ void tree_reduction(
		double* tree,
		double stock_price,
		double strike_price,
		int num_steps,
		double R,
		double up_factor,
		double up_prob) {

	for (int step = num_steps - 1; step >= 0; --step) {
		int branch = threadIdx.x;
		if (branch <= step) {
			double binomial = 1 / R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
			double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step);
			tree[branch] = max(binomial, exercise);
		}
		__syncthreads();
	}
}

double gpu1_binomial_american_put(
		double stock_price,
		double strike_price,
		double expire,
		double volat,
		int num_steps,
		double risk_free_rate) {
	double dt = expire / num_steps;
	double up_factor = exp(volat * sqrt(dt));
	double down_factor = 1 / up_factor;
	double R = exp((risk_free_rate) * dt);
	double up_prob = (R - down_factor) / (up_factor - down_factor);
	double *host_tree = new double[num_steps + 1];
	double *dev_tree;

	// Initialize end of host_tree at expire time
	for (int step = 0; step <= num_steps; ++step) {
		// Option value when exercising the option
		double exercise = strike_price - stock_price * pow(up_factor, 2 * step - num_steps);
		host_tree[step] = max(exercise, .0);
	}

	check_err(cudaMalloc((void** ) &dev_tree, (num_steps + 1) * sizeof(double)));
	check_err(cudaMemcpy(dev_tree, host_tree, (num_steps + 1) * sizeof(double), cudaMemcpyHostToDevice));

	tree_reduction<<<1, num_steps + 1>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);

	double price;
	check_err(cudaMemcpy(&price, &dev_tree[0], sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(dev_tree);
	delete[] host_tree;
	return price;
}

//////// Build tree on GPU directly ////////

__global__ void tree_builder(
		double* tree,
		double stock_price,
		double strike_price,
		int num_steps,
		double R,
		double up_factor,
		double up_prob) {

	// Initialize end of host_tree at expire time
	int branch = threadIdx.x;
	// Option value when exercising the option
	double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - num_steps);
	tree[branch] = max(exercise, .0);

	for (int step = num_steps - 1; step >= 0; --step) {
		__syncthreads();
		int branch = threadIdx.x;
		if (branch <= step) {
			double binomial = 1 / R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
			double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step);
			tree[branch] = max(binomial, exercise);
		}
	}
}

double gpu2_binomial_american_put(
		double stock_price,
		double strike_price,
		double expire,
		double volat,
		int num_steps,
		double risk_free_rate) {
	double dt = expire / num_steps;
	double up_factor = exp(volat * sqrt(dt));
	double down_factor = 1 / up_factor;
	double R = exp((risk_free_rate) * dt);
	double up_prob = (R - down_factor) / (up_factor - down_factor);
	double* dev_tree;
	double price;

	check_err(cudaMalloc((void** ) &dev_tree, (num_steps + 1) * sizeof(double)));

	tree_builder<<<1, num_steps + 1>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);

	check_err(cudaMemcpy(&price, &dev_tree[0], sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(dev_tree);

	return price;
}

//////// Build tree on GPU with shared memory ////////

__global__ void tree_builder_shared(
		double* dev_price,
		double stock_price,
		double strike_price,
		int num_steps,
		double R,
		double up_factor,
		double up_prob) {

	__shared__ double tree[1024];

	// Initialize end of host_tree at expire time
	int branch = threadIdx.x;
	if (branch <= num_steps) {
		// Option value when exercising the option
		double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - num_steps);
		tree[branch] = max(exercise, .0);
	}

	for (int step = num_steps - 1; step >= 0; --step) {
		__syncthreads();
		int branch = threadIdx.x;
		if (branch <= step) {
			double binomial = 1 / R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
			double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step);
			tree[branch] = max(binomial, exercise);
		}
	}

	*dev_price = tree[0];
}

double gpu3_binomial_american_put(
		double stock_price,
		double strike_price,
		double expire,
		double volat,
		int num_steps,
		double risk_free_rate) {
	double dt = expire / num_steps;
	double up_factor = exp(volat * sqrt(dt));
	double down_factor = 1 / up_factor;
	double R = exp((risk_free_rate) * dt);
	double up_prob = (R - down_factor) / (up_factor - down_factor);
	double price;
	double* dev_price;

	check_err(cudaMalloc((void** ) &dev_price, 1 * sizeof(double)));

	tree_builder_shared<<<1, num_steps + 1>>>(dev_price, stock_price, strike_price, num_steps, R, up_factor, up_prob);

	check_err(cudaMemcpy(&price, dev_price, sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(dev_price);

	return price;
}

//////// Build tree on GPU with texture memory ////////
/*
 texture<double> tex_tree1;
 texture<double> tex_tree2;

 __global__ void tree_builder_texture(
 double* tree1,
 double* tree2,
 double stock_price,
 double strike_price,
 int num_steps,
 double R,
 double up_factor,
 double up_prob) {



 // Initialize end of host_tree at expire time
 int branch = blockIdx.x;
 if (branch <= num_steps) {
 // Option value when exercising the option
 double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - num_steps);
 tree1[branch] = max(exercise, .0);
 }

 for (int step = num_steps - 1; step >= 0; --step) {
 int branch = blockIdx.x;
 if (branch <= step) {
 double binomial = 1 / R * (up_prob * tree1[branch + 1] + (1 - up_prob) * tree1[branch]);
 double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step);
 tree1[branch] = max(binomial, exercise);
 }
 __syncthreads();
 }
 }

 double gpu4_binomial_american_put(
 double stock_price,
 double strike_price,
 double expire,
 double volat,
 int num_steps,
 double risk_free_rate) {
 double dt = expire / num_steps;
 double up_factor = exp(volat * sqrt(dt));
 double down_factor = 1 / up_factor;
 double R = exp((risk_free_rate) * dt);
 double up_prob = (R - down_factor) / (up_factor - down_factor);
 double* dev_tree1;
 double* dev_tree2;
 double price;

 size_t tree_size = (num_steps + 1) * sizeof(double);
 check_err(cudaMalloc((void** ) &dev_tree1, tree_size));
 check_err(cudaMalloc((void** ) &dev_tree2, tree_size));
 check_err(cudaBindTexture(NULL, tex_tree1, dev_tree1, tree_size));
 check_err(cudaBindTexture(NULL, tex_tree2, dev_tree2, tree_size));

 tree_builder<<<num_steps, 1>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);

 check_err(cudaMemcpy(&price, &dev_tree[0], sizeof(double), cudaMemcpyDeviceToHost));
 cudaFree(dev_tree);



 check_err(cudaMalloc((void** ) &dev_price, 1 * sizeof(double)));

 tree_builder_shared<<<1, num_steps>>>(dev_price, stock_price, strike_price, num_steps, R, up_factor, up_prob);

 check_err(cudaMemcpy(&price, dev_price, sizeof(double), cudaMemcpyDeviceToHost));
 cudaUnbindTexture(tex_tree1);
 cudaUnbindTexture(tex_tree2);
 cudaFree(dev_tree1);
 cudaFree(dev_tree2);

 return price;
 }
 */

#if 1

//////// Build tree on with triangles and parallelograms ////////

enum BrickPos {
	CEIL_EDGE, INNER, FLOOR_EDGE, FINAL
};
const int NUM_STEPS = 1024;

/**
 * Aggregate one half brick, starting from the leaf nodes in the complete tree
 *
 * Needs to be launched with 'NUM_STEPS' threads
 */
template<BrickPos Pos>
__global__ void tree_builder_triangle(
		double stock_price,
		double strike_price,
		double R,
		double up_factor,
		double up_prob,
		int root_pos,
		double* out_climbing_edge,
		double* out_sinking_edge) {

	__shared__ double tree[NUM_STEPS];

	// Initialize end of host_tree at expire time
	int branch = threadIdx.x;
	// Option value when exercising the option
	double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - NUM_STEPS + root_pos);
tree[branch] = max(exercise, .0);

	for (int step = NUM_STEPS - 2; step >= 0; --step) {
		__syncthreads();

		int branch = threadIdx.x;
		if (branch <= step) {
			double binomial = 1 / R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
			double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step + root_pos);
			tree[branch] = max(binomial, exercise);

			if (Pos != FLOOR_EDGE) {
				out_sinking_edge[step] = tree[0];
			}
		}
	}

	if (Pos != CEIL_EDGE) {
		out_climbing_edge[threadIdx.x] = tree[threadIdx.x];
	}
}

template<BrickPos Pos>
__global__ void tree_builder_brick(
		double stock_price,
		double strike_price,
		double R,
		double up_factor,
		double up_prob,
		int root_pos,
		double* in_upper_edge,
		double* in_lower_edge,
		double* out_climbing_edge,
		double* out_sinking_edge) {

	// Use shared memory for speedup
	__shared__ double tree[NUM_STEPS + 1];
	tree[threadIdx.x] = in_lower_edge[threadIdx.x];

	// Create first triangle
	for (int step = NUM_STEPS - 1; step >= 0; --step) {

		// Take one value from the upper edge, each step backward
		if (threadIdx.x == NUM_STEPS - 1) { // This is the only thread that read tree[NUM_STEPS], so only wait for that thread
			tree[NUM_STEPS] = in_upper_edge[NUM_STEPS - step - 1];
		}

		__syncthreads(); // Fill all leafs, before writing over with the next step's leafs

		int branch = threadIdx.x;
		if (branch >= step) {
			double binomial = 1 / R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
			double exercise = strike_price
					- stock_price * pow(up_factor, 2 * branch - step - (NUM_STEPS - 1) + root_pos);
			tree[branch] = max(binomial, exercise);
		}
	}

	// Create second triangle
	for (int step = NUM_STEPS - 2; step >= 0; --step) {
		__syncthreads();

		int branch = threadIdx.x;
		if (branch <= step) {
			double binomial = 1 / R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
			double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step + root_pos);
			tree[branch] = max(binomial, exercise);

			if (Pos != FLOOR_EDGE && Pos != FINAL) {
				out_sinking_edge[step] = tree[0];
			}
		}
	}

	if (Pos == FINAL) {
		*out_climbing_edge = tree[0];
	} else if (Pos != CEIL_EDGE) {
		out_climbing_edge[threadIdx.x] = tree[threadIdx.x];
	}
}

double gpu4_binomial_american_put(
		double stock_price,
		double strike_price,
		double expire,
		double volat,
		int num_steps,
		double risk_free_rate) {
	double dt = expire / num_steps;
	double up_factor = exp(volat * sqrt(dt));
	double down_factor = 1 / up_factor;
	double R = exp((risk_free_rate) * dt);
	double up_prob = (R - down_factor) / (up_factor - down_factor);
	double price;

	double* edge1;
	double* edge2;
	double* dev_price;
	check_err(cudaMalloc((void** ) &edge1, NUM_STEPS * sizeof(double)));
	check_err(cudaMalloc((void** ) &edge2, NUM_STEPS * sizeof(double)));
	check_err(cudaMalloc((void** ) &dev_price, 1 * sizeof(double)));

	tree_builder_triangle<CEIL_EDGE> <<<1, NUM_STEPS>>>(stock_price, strike_price, R, up_factor, up_prob, NUM_STEPS,
	NULL, edge1);
	tree_builder_triangle<FLOOR_EDGE> <<<1, NUM_STEPS>>>(stock_price, strike_price, R, up_factor, up_prob, -NUM_STEPS,
			edge2, NULL);
	check_err(cudaStreamSynchronize(0));
	tree_builder_brick<FINAL> <<<1, NUM_STEPS>>>(stock_price, strike_price, R, up_factor, up_prob, 0, edge1, edge2,
			dev_price, NULL);

	check_err(cudaMemcpyAsync(&price, dev_price, 1 * sizeof(double), cudaMemcpyDefault));
	check_err(cudaStreamSynchronize(0));

	cudaFree(edge1);
	cudaFree(edge2);
	cudaFree(dev_price);

	return price;
}

#endif

void gpu_benchmark(const char* name, double (*to_invoke)(int), const int reruns) {
	cudaEvent_t start, end;
	check_err(cudaEventCreate(&start));
	check_err(cudaEventCreate(&end));

//printf("%-32s, %3.8f", name, to_invoke(100));

// Warm up
	for (int i = 0; i < reruns / 10; ++i) {
		to_invoke(100);
	}

	for (int var = 0; var <= 1000; var += 10) {
		check_err(cudaEventRecord(start, 0));
		for (int i = 0; i < reruns; ++i) {
			to_invoke(var);
		}

		check_err(cudaEventRecord(end, 0));
		check_err(cudaEventSynchronize(end));

		float duration;
		check_err(cudaEventElapsedTime(&duration, start, end));
		duration = duration / reruns * pow(10, 3);
		printf(", %6d", int(duration + 0.5)); // Microseconds
	}

	printf("\n");
	check_err(cudaEventDestroy(start));
	check_err(cudaEventDestroy(end));
}

void cpu_benchmark(const char* name, double (*to_invoke)(int), int reruns) {
	clock_t start;
	clock_t end;

//printf("%-32s, %3.8f", name, to_invoke(100));

// Warm up
	for (int i = 0; i < reruns / 10; ++i) {
		to_invoke(100);
	}

	for (int var = 0; var <= 1000; var += 10) {
		// Start test
		start = clock();
		for (int i = 0; i < reruns; ++i) {
			to_invoke(var);
		}

		end = clock();
		float duration = (double(end) - double(start)) / CLOCKS_PER_SEC / reruns * pow(10, 6);

		printf(", %6d", int(duration + 0.5)); // Microseconds
	}

	printf("\n");
}

double cpu(int indep_var) {
	return cpu_dynprog_binomial_american_put(20, 25, .5, 1, indep_var, 0.06);
}

double gpu1(int indep_var) {
	return gpu1_binomial_american_put(20, 25, .5, 1, indep_var, 0.06);
}

double gpu2(int indep_var) {
	return gpu2_binomial_american_put(20, 25, .5, 1, indep_var, 0.06);
}

double gpu3(int indep_var) {
	return gpu3_binomial_american_put(20, 25, .5, 1, indep_var, 0.06);
}

double gpu4(int indep_var) {
	return gpu4_binomial_american_put(20, 25, .5, 1, indep_var, 0.06);
}

int main() {
	printf("Compiled: %s %s\n", __DATE__, __TIME__);
	printf("cpu: %f\n", cpu(2048 - 1));
	printf("gpu: %f\n", gpu4(0));

	return;
	const int reruns = 100;
//const size_t num_step_tests = 11;
//int step_tests[num_step_tests] = { 1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };

	/*	printf("\"%-30s\", \"%-8s\"", "Implementation", "100 step");
	 for (int i = 0; i < num_step_tests; ++i) {
	 printf(", %6d", step_tests[i]);
	 }
	 printf("\n");*/

	cpu_benchmark("CPU dynprog", cpu, reruns);
	gpu_benchmark("GPU tree reduction", gpu1, reruns);
	gpu_benchmark("GPU tree build and reduction", gpu2, reruns);
	gpu_benchmark("GPU shared memory", gpu3, reruns);


	return 0;
}
