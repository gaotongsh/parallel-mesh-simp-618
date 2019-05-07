#ifndef CMU462_MESHEDIT_H
#define CMU462_MESHEDIT_H

#include "halfEdgeMesh.h"

using namespace std;

namespace CMU462 {

class MeshResampler {
 public:
  MeshResampler(){};
  ~MeshResampler() {}

  void downsample(HalfedgeMesh& mesh, float r);
};

}  // namespace CMU462

#endif  // CMU462_MESHEDIT_H
