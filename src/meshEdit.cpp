#include <float.h>
#include <assert.h>
#include "matrix3x3.h"
#include "meshEdit.h"
#include "mutablePriorityQueue.h"
#include "cuda_simplify/cycleTimer.h"

#include <set>
#include <algorithm>

namespace CMU462 {

set<EdgeIter> HalfedgeMesh::remove_tetrahedron(EdgeIter e) {

  auto h = e->halfedge(), h_twin = h->twin();
  auto v_keep = h->vertex(), v_delete = h_twin->vertex();

  vector<VertexIter> v_keep_neighbor, v_delete_neighbor, v_intersection;
  set<EdgeIter> e_delete;
  do {
    v_keep_neighbor.clear();
    v_delete_neighbor.clear();
    v_intersection.clear();
    auto hi = v_keep->halfedge();
    do {
      v_keep_neighbor.push_back(hi->twin()->vertex());
      hi = hi->twin()->next();
    } while (hi != v_keep->halfedge());
    hi = v_delete->halfedge();
    do {
      v_delete_neighbor.push_back(hi->twin()->vertex());
      hi = hi->twin()->next();
    } while (hi != v_delete->halfedge());

    sort(v_keep_neighbor.begin(), v_keep_neighbor.end());
    sort(v_delete_neighbor.begin(), v_delete_neighbor.end());
    set_intersection(v_keep_neighbor.begin(), v_keep_neighbor.end(),
                     v_delete_neighbor.begin(), v_delete_neighbor.end(),
                     back_inserter(v_intersection));
    if (v_intersection.size() <= 2)
      break;

    cerr << "Tetrahedron spotted! Adds: "
         << elementAddress(v_keep) << " and " << elementAddress(v_delete) << endl;

    auto v_to_erase_begin = v_intersection.front();
    for (auto& vi : v_intersection) {
      if (vi->degree() < v_to_erase_begin->degree())
        v_to_erase_begin = vi;
    }

    vector<VertexIter> v_to_erase;
    v_to_erase.push_back(v_to_erase_begin);

    for (Index i = 0; i < v_to_erase.size(); ++i) {
      auto hj = v_to_erase[i]->halfedge();
      do {
        e_delete.insert(hj->edge());

        auto v_neighbor = hj->twin()->vertex();
        if (v_neighbor != v_delete && v_neighbor != v_keep
            && find(v_intersection.begin(), v_intersection.end(), v_neighbor) == v_intersection.end()
            && find(v_to_erase.begin(), v_to_erase.end(), v_neighbor) == v_to_erase.end()) {
          v_to_erase.push_back(v_neighbor);
        }

        hj = hj->twin()->next();
      } while (hj != v_to_erase[i]->halfedge());
    }

    for (auto& vi : v_to_erase) {
      eraseVertex(vi);
    }

  } while (true);

  return e_delete;
}

VertexIter HalfedgeMesh::collapseEdge(EdgeIter e) {
  // This method should collapse the given edge and return an iterator to
  // the new vertex created by the collapse.

  remove_tetrahedron(e);

  auto h = e->halfedge(), h_twin = h->twin();
  auto v_keep = h->vertex(), v_delete = h_twin->vertex();

  // Collect
  // HalfEdges
  vector<HalfedgeIter> h_to_be_deleted;
  h_to_be_deleted.push_back(h);
  h_to_be_deleted.push_back(h_twin);

  vector<HalfedgeIter> h_delete_vec;
  auto hi = h_twin;
  while (hi->twin()->next() != h_twin) {
    hi = hi->twin()->next();
    h_delete_vec.push_back(hi);
  }
  auto h_0 = h_delete_vec.front(), h_0_twin = h_0->twin(),
       h_n = h_delete_vec.back(), h_n_twin = h_n->twin(),
       h_1 = h_0_twin->next();

  auto h_first = h;
  while (h_first->next() != h) {
    h_first = h_first->next();
  }
  auto h_last = h_twin->next();
  auto h_first_prev = h_1;
  while (h_first_prev->next() != h_0_twin) {
    h_first_prev = h_first_prev->next();
  }
  auto h_last_prev = h_n->next();
  while (h_last_prev->next() != h_n) {
    h_last_prev = h_last_prev->next();
  }
  auto h_last_next = h_n->next();

  // Vertices
  v_keep->position = (v_keep->position + v_delete->position) / 2;
  auto v_0 = h_0_twin->vertex(), v_n = h_n_twin->vertex();

  // Edges
  vector<EdgeIter> e_to_be_deleted;
  e_to_be_deleted.push_back(e);
  auto e_0 = h_0->edge(), e_n = h_n->edge();

  // Faces
  vector<FaceIter> f_to_be_deleted;
  auto f_0 = h->face(),
       f_1 = h_0_twin->face(),
       f_n = h_n->face(),
       f_n_p1 = h_twin->face();

  // Reassign
  if (h_0->next() == h_first) {
    h_to_be_deleted.push_back(h_0);
    h_to_be_deleted.push_back(h_0_twin);
    e_to_be_deleted.push_back(e_0);
    f_to_be_deleted.push_back(f_0);

    // HalfEdges
    // We have a corner case here: when the faces on either side of the edge have a degree of 3,
    // while the vertex to be deleted also has a degree of 3. We have to deal with this carefully.
    if (h_last->next() == h_n_twin && f_1 == f_n) {
      h_first->next() = h_last;
    } else {
      h_first->next() = h_1;
    }
    h_first->face() = f_1;
    h_first_prev->next() = h_first;

    // Vertices
    v_0->halfedge() = h_first;

    // Faces
    f_1->halfedge() = h_first;
  } else {
    h_first->next() = h_0;
    f_0->halfedge() = h_0;
  }
  for (auto& hj : h_delete_vec) {
    hj->vertex() = v_keep;
  }
  if (h_last->next() == h_n_twin) {
    h_to_be_deleted.push_back(h_n);
    h_to_be_deleted.push_back(h_n_twin);
    e_to_be_deleted.push_back(e_n);
    f_to_be_deleted.push_back(f_n_p1);

    // HalfEdges
    h_last->next() = h_last_next;
    h_last->face() = f_n;
    h_last_prev->next() = h_last;

    // Vertices
    v_n->halfedge() = h_last_next;

    // Faces
    f_n->halfedge() = h_last;
  } else {
    h_n_twin->next() = h_last;
    f_n_p1->halfedge() = h_last;
  }
  v_keep->halfedge() = h_last;

  // Delete
  for (auto& hj : h_to_be_deleted) {
    deleteHalfedge(hj);
  }
  deleteVertex(v_delete);
  for (auto& ej : e_to_be_deleted) {
    deleteEdge(ej);
  }
  for (auto& fj : f_to_be_deleted) {
    deleteFace(fj);
  }

  return v_keep;
}

FaceIter HalfedgeMesh::eraseVertex(VertexIter v) {
  // This method should replace the given vertex and all its neighboring
  // edges and faces with a single face, returning the new face.

  // Collect
  // HalfEdges
  auto h = v->halfedge(), ht = h->twin();
  vector<HalfedgeIter> h_to_be_deleted, h_outer_first, h_outer_last, h_outer_all;
  h_to_be_deleted.push_back(h);
  h_to_be_deleted.push_back(ht);
  h_outer_first.push_back(h->next());

  // Vertices
  vector<VertexIter> v_outer_vec;
  v_outer_vec.push_back(ht->vertex());

  // Edges
  vector<EdgeIter> e_to_be_deleted;
  e_to_be_deleted.push_back(h->edge());

  // Faces
  auto f0 = h->face();
  vector<FaceIter> f_to_be_deleted;

  // Iterations through remaining
  auto hp = ht, hi = hp->next();
  while (hi != h) {
    h_to_be_deleted.push_back(hi);
    h_to_be_deleted.push_back(hi->twin());
    h_outer_first.push_back(hi->next());

    v_outer_vec.push_back(hi->twin()->vertex());
    e_to_be_deleted.push_back(hi->edge());
    f_to_be_deleted.push_back(hi->face());

    auto hj = hi;
    while (hj->next() != hp) {
      hj = hj->next();
      h_outer_all.push_back(hj);
    }
    h_outer_last.push_back(hj);

    hp = hi->twin();
    hi = hp->next();
  }
  while (hi->next() != hp) {
    hi = hi->next();
    h_outer_all.push_back(hi);
  }
  h_outer_last.push_back(hi);

  // Reassign
  Size n = h_outer_first.size();
  for (Index i = 0; i < n; ++i) {
    h_outer_last[i]->next() = h_outer_first[i];
    v_outer_vec[i]->halfedge() = h_outer_first[i];
  }
  for (auto& hj : h_outer_all) {
    hj->face() = f0;
  }
  f0->halfedge() = h_outer_first.front();

  // Delete
  for (auto& hj : h_to_be_deleted) {
    deleteHalfedge(hj);
  }
  deleteVertex(v);
  for (auto& ej : e_to_be_deleted) {
    deleteEdge(ej);
  }
  for (auto& fj : f_to_be_deleted) {
    deleteFace(fj);
  }

  return f0;
}

EdgeRecord::EdgeRecord(EdgeIter& _edge) : edge(_edge) {
  // Compute the combined quadric from the edge endpoints.

  auto h = edge->halfedge();
  auto K = h->vertex()->quadric + h->twin()->vertex()->quadric;

  // -> Build the 3x3 linear system whose solution minimizes the quadric error
  //    associated with these two endpoints.

  double A_data[9] = { K(0,0), K(0,1), K(0,2),
                       K(1,0), K(1,1), K(1,2),
                       K(2,0), K(2,1), K(2,2) };
  auto A = Matrix3x3(A_data);
  auto b = Vector3D(-K(0,3), -K(1,3), -K(2,3));

  // -> Use this system to solve for the optimal position, and store it in
  //    EdgeRecord::optimalPoint.

  if (A.det() > 0.01) {
    optimalPoint = A.inv() * b;
  } else {
//    cerr << "detA is too small: " << A.det() << endl;
//    cerr << "inverse answer: " << A.inv() * b << endl;

    auto p0 = h->vertex()->position, p1 = h->twin()->vertex()->position;
    while ((p0 - p1).norm2() > 0.0001) {
      auto c0 = dot(p0, A * p0) - 2 * dot(b, p0),
           c1 = dot(p1, A * p1) - 2 * dot(b, p1);
      if (c0 > c1) {
        p0 = (p0 + p1) / 2;
      } else {
        p1 = (p0 + p1) / 2;
      }
    }
    optimalPoint = p0;

//    cerr << "actual answer: " << optimalPoint << endl << endl;
  }

  // -> Also store the cost associated with collapsing this edg in
  //    EdgeRecord::Cost.

  score = dot(optimalPoint, A * optimalPoint) - 2 * dot(b, optimalPoint) + K(3,3);
}

bool te(EdgeIter e) {
  auto h = e->halfedge(), h_twin = h->twin();
  auto v_keep = h->vertex(), v_delete = h_twin->vertex();

  vector<VertexIter> v_keep_neighbor, v_delete_neighbor, v_intersection;
  set<EdgeIter> e_delete;

    v_keep_neighbor.clear();
    v_delete_neighbor.clear();
    v_intersection.clear();
    auto hi = v_keep->halfedge();
    do {
      v_keep_neighbor.push_back(hi->twin()->vertex());
      hi = hi->twin()->next();
    } while (hi != v_keep->halfedge());
    hi = v_delete->halfedge();
    do {
      v_delete_neighbor.push_back(hi->twin()->vertex());
      hi = hi->twin()->next();
    } while (hi != v_delete->halfedge());

    sort(v_keep_neighbor.begin(), v_keep_neighbor.end());
    sort(v_delete_neighbor.begin(), v_delete_neighbor.end());
    set_intersection(v_keep_neighbor.begin(), v_keep_neighbor.end(),
                     v_delete_neighbor.begin(), v_delete_neighbor.end(),
                     back_inserter(v_intersection));
    return !(v_intersection.size() <= 2);
}

void MeshResampler::downsample(HalfedgeMesh& mesh, float r) {
  // Compute initial quadrics for each face by simply writing the plane equation
  // for the face in homogeneous coordinates. These quadrics should be stored
  // in Face::quadric

  for (auto f = mesh.facesBegin(); f != mesh.facesEnd(); ++f) {
    auto n = f->normal(), p = f->halfedge()->vertex()->position;
    auto v = Vector4D(n.x, n.y, n.z, -dot(n, p));
    f->quadric = outer(v, v);}

  // -> Compute an initial quadric for each vertex as the sum of the quadrics
  //    associated with the incident faces, storing it in Vertex::quadric

  for (auto v = mesh.verticesBegin(); v != mesh.verticesEnd(); ++v) {
    auto q = Matrix4x4();
    auto h = v->halfedge();
    do {
      q += h->face()->quadric;
      h = h->twin()->next();
    } while (h != v->halfedge());
    v->quadric = q;
  }

  // -> Build a priority queue of edges according to their quadric error cost,
  //    i.e., by building an EdgeRecord for each edge and sticking it in the
  //    queue.

  MutablePriorityQueue<EdgeRecord> queue;
  for (auto e = mesh.edgesBegin(); e != mesh.edgesEnd(); ++e) {
    e->record = EdgeRecord(e);
    queue.insert(e->record);
  }

  watchTake((char*)"Quadrics.");

  // -> Until we reach the target edge budget, collapse the best edge. Remember
  //    to remove from the queue any edge that touches the collapsing edge
  //    BEFORE it gets collapsed, and add back into the queue any edge touching
  //    the collapsed vertex AFTER it's been collapsed. Also remember to assign
  //    a quadric to the collapsed vertex, and to pop the collapsed edge off the
  //    top of the queue.

  
  Size currentSize = mesh.nFaces();
  Size target = currentSize * r;
  while (currentSize > target) { // This line is causing performance issues!!
    EdgeRecord er;

    do {
      // 1. Get the cheapest edge from the queue.
      er = queue.top();
      // 2. Remove the cheapest edge from the queue by calling pop().
      queue.pop();
    } while (te(er.edge));

    // 3. Compute the new quadric by summing the quadrics at its two endpoints.
    auto h = er.edge->halfedge();
    auto v0 = h->vertex(), v1 = h->twin()->vertex();
    auto newq = v0->quadric + v1->quadric;
    // 4. Remove any edge touching either of its endpoints from the queue.
    auto hi = v0->halfedge();
    do {
      queue.remove(hi->edge()->record);
      hi = hi->twin()->next();
    } while (hi != v0->halfedge());
    hi = v1->halfedge();
    do {
      queue.remove(hi->edge()->record);
      hi = hi->twin()->next();
    } while (hi != v1->halfedge());

    // [ADDITIONAL] Remove the tetrahedron edges.
    //auto e_deleted = mesh.remove_tetrahedron(er.edge);
    //for (auto& e : e_deleted) {
    //  queue.remove(e->record);
    //}

    // 5. Collapse the edge.
    auto newv = mesh.collapseEdge(er.edge);
    newv->position = er.optimalPoint;
    // 6. Set the quadric of the new vertex to the quadric computed in Step 3.
    newv->quadric = newq;
    // 7. Insert any edge touching the new vertex into the queue, creating new edge records for each of them.
    hi = newv->halfedge();
    do {
      auto e = hi->edge();
      e->record = EdgeRecord(e);
      queue.insert(e->record);
      hi = hi->twin()->next();
    } while (hi != newv->halfedge());

    currentSize -= 2;
    // if (currentSize != mesh.nFaces()) {
    //   printf("Error. Difference in Size (%d, %d).\n", currentSize, mesh.nFaces());
    // }
  }
}

}  // namespace CMU462
