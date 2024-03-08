// CN snippet provided by Claude 3 on 20240307

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Function to calculate the modular exponentiation (a^b mod n)
static size_t modular_pow(size_t a, size_t b, size_t n) {
    size_t res = 1;
    a %= n;
    while (b > 0) {
        if (b % 2 == 1)
            res = (res * a) % n;
        a = (a * a) % n;
        b /= 2;
        // cout << a << " " << b << " " << n << endl;
    }
    return res;
}

// implement the Euclidean algorithm recursively for gcd
static size_t gcd(size_t a, size_t b) {
    if (a > b) {
        return gcd(a-b, b);
    } else if (b > a) {
        return gcd(a, b-a);
    }
    return a;
}

static void print_factors(vector<size_t> factors) {
    for(size_t f : factors) {
        cout << f << ".";
        };
}


static bool is_prime(size_t n) {
    bool is_prime = true;
        for (size_t j = 3; j * j <= n; j += 2) {
            if (n % j == 0) {
                is_prime = false;
                break;
            }
        }
    return is_prime;
}



// Function to check if a number is Carmichael
static bool is_carmichael(size_t n) {

    vector<size_t> factors = {};

    if (n <= 1)
        return false;

    for (size_t a = 2; a < n; a++) {
        size_t euclid = gcd(a, n);
        if (euclid == 1) {
            if (modular_pow(a, n-1, n) != 1)
                return false;
        } else if (is_prime(euclid) && !any_of(begin(factors), end(factors), [euclid](size_t y){return euclid == y; })) {
            factors.push_back(euclid);
        }
    }
    print_factors(factors);
    cout << " = " << n << endl;;
    return true;
}


int main() {
    size_t limit;
    cout << "Enter the limit: ";
    cin >> limit;

    vector<size_t> carmichael_numbers;

    for (size_t i = 3; i <= limit; i += 2) {
        if (!is_prime(i) && is_carmichael(i)) {
            carmichael_numbers.push_back(i);
            }
    }

    cout << "There are " << carmichael_numbers.size() << " Carmichael numbers below " << limit << ". They are: " << endl;
    for (size_t num : carmichael_numbers)
        cout << num << " ";
    cout << endl;

    return 0;
}
