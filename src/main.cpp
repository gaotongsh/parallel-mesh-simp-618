#include "meshEdit.h"
#include "cuda_simplify/cycleTimer.h"
#include "cuda_simplify/gpuSimplify.h"
#include "offFileManager.h"

#include <iostream>
#include <getopt.h>

#ifndef gid_t
typedef unsigned int gid_t;  // XXX Needed on some platforms, since gid_t is
                             // used in unistd.h but not always defined!
                             // (WARNING: May not be the right size!!)
#endif


using namespace std;
using namespace CMU462;

#define msg(s) cerr << "[MeshSimp] " << s << endl;

void usage(const char* binaryName) {
  printf("Usage: %s [options] <scenefile>\n", binaryName);
  printf("Program Options:\n");
  printf("  -r  <FLOAT>      Simplify mesh down to r times original vertices.\n");
  printf("  -c               Use CPU simplifier instead of GPU simplifier.\n");
  printf("  -h               Print this help message\n");
  printf("\n");
}

int main(int argc, char** argv) {
  int opt;
  float r = 0.5;
  bool useCpu = false;
  while ((opt = getopt(argc, argv, "h:r:c")) !=
         -1) {  // for each option...
    switch (opt) {
      case 'r':
        r = atof(optarg);
        break;
      case 'c':
        useCpu = true;
        break;
      default:
        usage(argv[0]);
        return 1;
    }
  }

  // print usage if no argument given
  if (optind >= argc) {
    usage(argv[0]);
    return 1;
  }

  string sceneFilePath = argv[optind];
  msg("Input scene file: " << sceneFilePath);

  vector<vector<size_t>> polygons, polygons_out;
  vector<Vector3D> vertexPositions, vertexPositions_out;
  int ret = read_off(sceneFilePath, polygons, vertexPositions);

  if (ret != 0) {
    msg("Error: parsing failed!")
    return ret;
  }
   
  if (!useCpu) {
    doCudaMeshSimplify(polygons, vertexPositions, polygons_out, vertexPositions_out, r);

    write_off("out.off", polygons_out, vertexPositions_out);
  } else {

    watchStart((char*)"CPU simplifier.");
    HalfedgeMesh mesh;
    mesh.build(polygons, vertexPositions);
    msg("Edge number: " << mesh.nEdges())

    watchTake((char*)"build edges");
    MeshResampler resampler;
    resampler.downsample(mesh, r);
    watchTake((char*)"Simp");
    msg("Edge number: " << mesh.nEdges())
    watchReport();
  }
  return 0;

}
