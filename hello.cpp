#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::string seed = (argc > 1) ? argv[1] : "no_seed";
    std::cout << "Hello from seed " << seed << std::endl;
    return 0;
}
