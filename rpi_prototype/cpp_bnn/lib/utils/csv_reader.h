#ifndef csv_reader_h
#define csv_reader_h

#include "data_type.h"

#include <string>
#include <fstream>
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

template <typename DATA_T=float>
Matrix<DATA_T> read_csv(std::string &filename, std::vector<size_t> data_shape, size_t row_offset=0, size_t reserve=0, bool colname_present=false) {
    // data_shape: 2d data shape, for other dimension, reshape later.
    // Reads a CSV file into custom Matrix, ignoring col name if present
    Matrix<DATA_T> result(data_shape);

    //if (reserve > result.size()) {
    //    result.m_data.reserve(reserve);
    //}
    //result.resize(data_shape);
    
    // Create an input filestream
    std::ifstream csv_file(filename);

    // Make sure the file is open
    if(!csv_file.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;

    // Read the column names
    if(colname_present) {
        if(csv_file.good()) {
            // Extract the first line in the file
            std::getline(csv_file, line);

            /*
            // Create a stringstream from line
            std::stringstream ss(line);

            // Extract each column name
            while(std::getline(ss, colname, ',')){
                // Initialize and add <colname, int vector> pairs to result
                result.push_back({colname, std::vector<int> {}});
            }
            */
        }
    }

    // Read data, line by line
    size_t row_idx = 0;
    size_t skip_row_count = 0;
    while(std::getline(csv_file, line))
    {
        
        // skip if < offset
        if (skip_row_count < row_offset) {
            skip_row_count++;
            continue;
        }
        // else read data
        // Create a stringstream of the current line
        // std::stringstream ss(line);
        std::stringstream ss(line);

        size_t col_idx = 0;

        // Extract each value with delimeter ","
        //while(ss >> val){
        for (std::string val; std::getline(ss, val, ',');){
        
            // Add the current integer to the 'colIdx' column's values vector
            result.set(stof(val), row_idx, col_idx);
            col_idx++;
            //printf("col_index: %d\n", col_idx);

            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
        }
        row_idx++;
        //printf("row_index: %d\n", row_idx);
        //printf("skip_row_count: %d\n", skip_row_count);
        if (row_idx > (data_shape[0]-1)) {
            break;
        }
        
    }

    // Close file
    csv_file.close();

    return result;
}

template <typename DATA_T=float>
void read_csv(std::string &filename, Matrix<DATA_T> & result, std::vector<size_t> data_shape, size_t row_offset=0, bool colname_present=false) {
    // data_shape: 2d data shape, for other dimension, reshape later.
    // Reads a CSV file into custom Matrix, ignoring col name if present

    // Create an input filestream
    std::ifstream csv_file(filename);

    // Make sure the file is open
    if(!csv_file.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;

    // Read data, line by line
    size_t row_idx = 0;
    size_t skip_row_count = 0;
    while(std::getline(csv_file, line))
    {
        
        // skip if < offset
        if (skip_row_count < row_offset) {
            skip_row_count++;
            continue;
        }
        // else read data
        // Create a stringstream of the current line
        // std::stringstream ss(line);
        std::stringstream ss(line);

        size_t col_idx = 0;

        // Extract each value with delimeter ","
        //while(ss >> val){
        for (std::string val; std::getline(ss, val, ',');){
        
            // Add the current integer to the 'colIdx' column's values vector
            result.set(stof(val), row_idx, col_idx);
            col_idx++;

            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
        }
        row_idx++;
        if (row_idx > (data_shape[0]-1)) {
            break;
        }
        
    }

    // Close file
    csv_file.close();

    return;
}
#endif