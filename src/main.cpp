#include <iostream>
#include <random>
#include <iomanip> // for consistent formatting

int main() {
    // Use mt19937 explicitly
    std::mt19937 rng(12345); // fixed seed
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Print first 10 numbers
    for(int i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(15)
                  << dist(rng) << "\n";
    }

    return 0;
}
