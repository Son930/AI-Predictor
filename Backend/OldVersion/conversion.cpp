#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

struct DataRow {
    double target;  // Target variable
    std::vector<double> features;  // All feature columns
};

// Function to convert CSV to LibSVM
void convertToLibSVM(const std::string& inputFile, const std::string& outputFile) {
    std::ifstream inFile(inputFile);
    std::ofstream outFile(outputFile);
    std::string line;

    // Read header (to determine the number of columns)
    std::getline(inFile, line);
    int num_features = std::count(line.begin(), line.end(), ','); // Count columns

    while (std::getline(inFile, line)) {
        std::stringstream ss(line);
        DataRow row;
        char delimiter;
        row.features.resize(num_features); // Adjust feature size

        // Read target (first column)
        ss >> row.target >> delimiter;

        // Read feature values
        for (int i = 0; i < num_features; i++) {
            ss >> row.features[i];
            if (i < num_features - 1) ss >> delimiter; // Skip delimiter except last column
        }

        // Write to LibSVM format: target 1:feature1 2:feature2 ...
        outFile << row.target;
        for (int i = 0; i < num_features; i++) {
            outFile << " " << (i + 1) << ":" << row.features[i];
        }
        outFile << "\n";
    }

    inFile.close();
    outFile.close();
}

int main() {
    convertToLibSVM("data.csv", "data.libsvm");
    std::cout << "Conversion to LibSVM complete!" << std::endl;
    return 0;
}
