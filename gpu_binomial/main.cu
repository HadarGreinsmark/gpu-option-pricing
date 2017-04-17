#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

#define check_err(cu_err) { cu_err_handler(cu_err, __FILE__, __LINE__); }

inline void cu_err_handler(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU error: '%s' at %s:%d", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

__global__ void tree_reduction(
		double* tree,
		double stock_price,
		double strike_price,
		int num_steps,
		double R,
		double up_factor,
		double up_prob) {

    for (int step = num_steps-1; step >= 0; --step) {
    	int branch = threadIdx.x;
    	if (branch <= step) {
            double binomial = 1/R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
            double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step);
            tree[branch] = max(binomial, exercise);
    	}
    	__syncthreads();
    }
}


double gpu1_binomial_american_put(double stock_price,
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
    double *host_tree = new double[num_steps+1];
    double *dev_tree;

    // Initialize end of host_tree at expire time
    for (int step = 0; step <= num_steps; ++step) {
        // Option value when exercising the option
        double exercise = strike_price - stock_price * pow(up_factor, 2 * step - num_steps);
        host_tree[step] = max(exercise, .0);
    }

    check_err(cudaMalloc((void**) &dev_tree, (num_steps+1) * sizeof(double)));
    check_err(cudaMemcpy(dev_tree, host_tree, (num_steps+1) * sizeof(double), cudaMemcpyHostToDevice));

    tree_reduction<<<1, num_steps>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);


    double price;
    check_err(cudaMemcpy(&price, &dev_tree[0], sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_tree);
    delete[] host_tree;
    return price;
}



// Build tree on GPU directly

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
    if (branch <= num_steps) {
        // Option value when exercising the option
        double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - num_steps);
        tree[branch] = max(exercise, .0);
    }

    for (int step = num_steps-1; step >= 0; --step) {
    	int branch = threadIdx.x;
    	if (branch <= step) {
            double binomial = 1/R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
            double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step);
            tree[branch] = max(binomial, exercise);
    	}
    	__syncthreads();
    }
}


double gpu2_binomial_american_put(double stock_price,
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

    check_err(cudaMalloc((void**) &dev_tree, (num_steps+1)*sizeof(double)));

    tree_builder<<<1, num_steps>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);

    check_err(cudaMemcpy(&price, &dev_tree[0], sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dev_tree);

    return price;
}



// Build tree on GPU directly

__global__ void tree_builder_shared(
		double* dev_price,
		double stock_price,
		double strike_price,
		int num_steps,
		double R,
		double up_factor,
		double up_prob) {

	__shared__ double tree[256];

    // Initialize end of host_tree at expire time
	int branch = threadIdx.x;
    if (branch <= num_steps) {
        // Option value when exercising the option
        double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - num_steps);
        tree[branch] = max(exercise, .0);
    }

    for (int step = num_steps-1; step >= 0; --step) {
    	int branch = threadIdx.x;
    	if (branch <= step) {
            double binomial = 1/R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
            double exercise = strike_price - stock_price * pow(up_factor, 2 * branch - step);
            tree[branch] = max(binomial, exercise);
    	}
    	__syncthreads();
    }
}


double gpu3_binomial_american_put(double stock_price,
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

    check_err(cudaMalloc((void**) &dev_price, 1*sizeof(double)));

    tree_builder_shared<<<1, num_steps>>>(dev_price, stock_price, strike_price, num_steps, R, up_factor, up_prob);

    check_err(cudaMemcpy(&price, dev_price, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dev_price);

    return price;
}



void benchmark_gpu(double (*to_invoke)()) {
	cudaEvent_t start, end;
	check_err(cudaEventCreate(&start));
	check_err(cudaEventCreate(&end));

	printf("Function returns: %f\n", to_invoke());

	printf("Warm up\n");
    for (int i = 0; i < 100; ++i) {
    	to_invoke();
    }

    printf("Start test\n");
    check_err(cudaEventRecord(start, 0));
    for (int i = 0; i < 1000; ++i) {
        to_invoke();
    }

    check_err(cudaEventRecord(end, 0));
    check_err(cudaEventSynchronize(end));

    float duration;
    check_err(cudaEventElapsedTime(&duration, start, end));
    check_err(cudaEventDestroy(start));
    check_err(cudaEventDestroy(end));

    printf("Took %d ms\n", int(duration + 0.5));
}

double gpu1() {
	return gpu1_binomial_american_put(20, 25, .5, 1, 200, 0.06);
}

double gpu2() {
	return gpu2_binomial_american_put(20, 25, .5, 1, 200, 0.06);
}

double gpu3() {
	return gpu3_binomial_american_put(20, 25, .5, 1, 200, 0.06);
}

int main() {
	benchmark_gpu(gpu1);
	benchmark_gpu(gpu2);
	benchmark_gpu(gpu3);

    return 0;
}
