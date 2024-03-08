// Claude 3 CN generator using Korselt's criteria

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Function to check if a number is prime
static bool isPrime(size_t n) {
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    for (size_t i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;

    return true;
}

// Function to calculate the value of (a^n) % n
static size_t modularExponentiation(size_t a, size_t n, size_t modulus) {
    size_t result = 1;
    a %= modulus;

    while (n > 0) {
        if (n & 1)
            result = (result * a) % modulus;
        a = (a * a) % modulus;
        n >>= 1;
    }

    return result;
}

// Function to check if a number is a Carmichael number
static bool isCarmichael(size_t n) {
    if (isPrime(n))
        return false;

    vector<size_t> divisors;
    for (size_t i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.push_back(i);
            if (i != n / i)
                divisors.push_back(n / i);
        }
    }

    for (size_t a = 2; a < n; a++) {
        if (isPrime(a)) {
            bool isCarmichael = true;
            for (size_t d : divisors) {
                if (modularExponentiation(a, d, n) != 1) {
                    isCarmichael = false;
                    break;
                }
            }
            if (isCarmichael)
                return true;
        }
    }

    return false;
}

int main() {
    size_t upperLimit;
    cout << "Enter the upper limit: ";
    cin >> upperLimit;

    vector<size_t> carmichaelNumbers;

    for (size_t i = 2; i <= upperLimit; i++) {
        if (isCarmichael(i))
            carmichaelNumbers.push_back(i);
    }

    if (carmichaelNumbers.empty())
        cout << "No Carmichael numbers found up to " << upperLimit << endl;
    else {
        cout << "Carmichael numbers up to " << upperLimit << ":" << endl;
        for (size_t n : carmichaelNumbers)
            cout << n << " ";
        cout << endl;
    }

    return 0;
}
