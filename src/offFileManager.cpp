//
// Created by Tong Gao on 2019-04-27.
//

#include "offFileManager.h"

#include <iostream>
#include <fstream>


int read_off(string &filename, vector<vector<size_t>> &polygons, vector<Vector3D> &vertexPositions) {
  // Container holding last line read
  string readLine;
  // Containers for delimiter positions
  int delimiterPos_1, delimiterPos_2, delimiterPos_3, delimiterPos_4;

  // Open file for reading
  ifstream in(filename);

  // Check if file is in OFF format
  getline(in, readLine);
  if (readLine != "OFF" && readLine != "COFF") {
    cout << "The file to read is not in OFF format." << endl;
    return -1;
  }

  int nv, nf;
  // Read values for Nv and Nf
  getline(in, readLine);
  delimiterPos_1 = readLine.find(' ', 0);
  nv = stoi(readLine.substr(0, delimiterPos_1));

  delimiterPos_2 = readLine.find(' ', delimiterPos_1 + 1);
  nf = stoi(readLine.substr(delimiterPos_1, delimiterPos_2));

  // Read the vertices
  for (int n = 0; n < nv; ++n) {
    getline(in, readLine);
    delimiterPos_1 = readLine.find(' ', 0);
    double x = stod(readLine.substr(0, delimiterPos_1));
    delimiterPos_2 = readLine.find(' ', delimiterPos_1 + 1);
    double y = stod(readLine.substr(delimiterPos_1, delimiterPos_2));
    delimiterPos_3 = readLine.find(' ', delimiterPos_2 + 1);
    double z = stod(readLine.substr(delimiterPos_2, delimiterPos_3));
    vertexPositions.emplace_back(x, y, z);
  }

  // Read the facades
  for (int n = 0; n < nf; ++n) {
    getline(in, readLine);
    delimiterPos_1 = readLine.find(' ', 0);
    delimiterPos_2 = readLine.find(' ', delimiterPos_1+1);
    size_t v1 = stoi(readLine.substr(delimiterPos_1,delimiterPos_2));
    delimiterPos_3 = readLine.find(' ', delimiterPos_2+1);
    size_t v2 = stoi(readLine.substr(delimiterPos_2,delimiterPos_3));
    delimiterPos_4 = readLine.find(' ', delimiterPos_3+1);
    size_t v3 = stoi(readLine.substr(delimiterPos_3,delimiterPos_4));
    vector<size_t> v = {v1, v2, v3};
    polygons.push_back(v);
  }

  in.close();
  return 0;
}

int write_off(const string &filename, const vector<vector<size_t>> &polygons, const vector<Vector3D> &vertexPositions) {

  // Open file for writing
  ofstream out(filename);

  if (!out.is_open()) {
    cout << "Open file error!" << endl;
    return -1;
  }

  // Write first two lines
  out << "OFF" << endl;
  out << vertexPositions.size() << ' ' << polygons.size() << " 0" << endl;

  // Write vertex positions
  for (auto &vertex : vertexPositions) {
    out << vertex.x << ' ' << vertex.y << ' ' << vertex.z << endl;
  }

  // Write polygons
  for (auto &polygon : polygons) {
    out << polygon.size();
    for (auto &v : polygon) {
      out << ' ' << v;
    }
    out << endl;
  }

  out.close();
  return 0;
}

