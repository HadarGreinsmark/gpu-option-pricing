#include <iostream>
#include <cmath>

using namespace std;

double binomial_american_put(double stock_price,
                             double strike_price,
                             double expire,
                             double volat,
                             int num_steps,
                             double risk_free_rate,
                             double dividend_yield) {
    double dt = expire / num_steps;
    double up_factor = exp(volat * sqrt(dt));
    double down_factor = 1 / up_factor;
    double R = exp((risk_free_rate - dividend_yield) * dt);
    double up_prob = (R - down_factor) / (up_factor - down_factor);
     double *tree = new double[num_steps+1];

    // Initialize end of tree at expire time
    for (int step = 0; step <= num_steps; ++step) {
        // Option value when exercising the option
        double exercise = strike_price - stock_price * pow(up_factor, 2 * step - num_steps);
        tree[step] = max(exercise, .0);
    }

    for (int step = num_steps-1; step >= 0; --step) {
        for (int branch = 0; branch <= step; ++branch) {
            double binomial = 1/R * (up_prob * tree[branch + 1] + (1 - up_prob) * tree[branch]);
            double exercise = strike_price - stock_price * pow(up_factor, (double) 2 * branch - step);
            tree[branch] = max(binomial, exercise);
        }
    }

    double price = tree[0];
    delete[] tree;
    return price;
}
int main() {
    cout << binomial_american_put(20, 25, .5, 1, 20, 0.06, 0) << endl;


    return 0;
}


