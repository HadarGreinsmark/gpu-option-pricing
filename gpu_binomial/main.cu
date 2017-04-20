#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <array>

using namespace std;

#define check_err(cu_err) { cu_err_handler(cu_err, __FILE__, __LINE__); }

inline void cu_err_handler(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU error: '%s' at %s:%d", cudaGetErrorString(err), file, line);
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

	tree_reduction<<<1, num_steps+1>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);

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

	tree_builder<<<1, num_steps+1>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);

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

	tree_builder_shared<<<1, num_steps+1>>>(dev_price, stock_price, strike_price, num_steps, R, up_factor, up_prob);

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

void gpu_benchmark(const char* name, double (*to_invoke)(int), int indep_var) {
	cudaEvent_t start, end;
	check_err(cudaEventCreate(&start));
	check_err(cudaEventCreate(&end));

	printf("Running: %s\n", name);
	printf("Function returns: %f\n", to_invoke(indep_var));

	// Warm up
	for (int i = 0; i < 100; ++i) {
		to_invoke(indep_var);
	}

	// Start test
	check_err(cudaEventRecord(start, 0));
	for (int i = 0; i < 1000; ++i) {
		to_invoke(indep_var);
	}

	check_err(cudaEventRecord(end, 0));
	check_err(cudaEventSynchronize(end));

	float duration;
	check_err(cudaEventElapsedTime(&duration, start, end));
	check_err(cudaEventDestroy(start));
	check_err(cudaEventDestroy(end));

	printf("Took %d ms\n\n", int(duration + 0.5));
}

void cpu_benchmark(const char* name, double (*to_invoke)(int), int indep_var) {
	clock_t start;
	clock_t end;

	printf("Running: %s\n", name);
	printf("Function returns: %f\n", to_invoke(indep_var));

	// Warm up
	for (int i = 0; i < 100; ++i) {
		to_invoke(indep_var);
	}

	// Start test
	start = clock();
	for (int i = 0; i < 1000; ++i) {
		to_invoke(indep_var);
	}

	end = clock();
	float duration = (double(end) - double(start)) * 1000 / CLOCKS_PER_SEC;
	printf("Took %d ms\n\n", int(duration + 0.5));
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

int main() {
	int num_steps[4] = {100, 200, 500, 1000};
	// TODO: Print CPU and GPU info
	for (int i = 0; i < 4; ++i) {
		printf("======== %d steps ========\n", num_steps[i]);
		cpu_benchmark("CPU dynprog", cpu, num_steps[i]);
		gpu_benchmark("GPU tree reduction", gpu1, num_steps[i]);
		gpu_benchmark("GPU tree build and reduction", gpu2, num_steps[i]);
		gpu_benchmark("GPU shared memory", gpu3, num_steps[i]);
	}
	return 0;
}
