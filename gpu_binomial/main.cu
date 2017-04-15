#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

#define cu_err_check(cu_err) { cu_err_handler(cu_err, __FILE__, __LINE__); }

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


double binomial_american_put(double stock_price,
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

    cu_err_check(cudaMalloc((void**) &dev_tree, (num_steps+1) * sizeof(double)));
    cu_err_check(cudaMemcpy(dev_tree, host_tree, (num_steps+1) * sizeof(double), cudaMemcpyHostToDevice));

    tree_reduction<<<1, num_steps>>>(dev_tree, stock_price, strike_price, num_steps, R, up_factor, up_prob);


    double price;
    cu_err_check(cudaMemcpy(&price, &dev_tree[0], sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_tree);
    delete[] host_tree;
    return price;
}
int main() {
    cout << binomial_american_put(20, 25, .5, 1, 20, 0.06) << endl;

    return 0;
}
