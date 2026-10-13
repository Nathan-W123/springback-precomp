// Where this fits: see include/sparlab/elements/FaceGeometry.hpp

#include "sparlab/elements/FaceGeometry.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Quadrature.hpp"

#include <Eigen/Geometry>

#include <sstream>

namespace sparlab {
namespace {

void check_nodes(FaceShape shape, const Matrix& face_coords) {
  if (face_coords.cols() != face_shape_nodes(shape)) {
    std::ostringstream os;
    os << "a face of this shape has " << face_shape_nodes(shape) << " nodes, received "
       << face_coords.cols() << " coordinate columns";
    throw MeshError(os.str());
  }
}

}  // namespace

FaceShape face_shape_of(ElementType type) {
  switch (type) {
    case ElementType::Hex8: return FaceShape::Quad4;
  }
  throw MeshError("element type has no boundary-face shape registered");
}

int face_shape_nodes(FaceShape shape) {
  switch (shape) {
    case FaceShape::Quad4: return 4;
  }
  return 0;
}

std::vector<FacePoint> face_quadrature(FaceShape shape, int points) {
  if (points < 1 || points > 4) {
    std::ostringstream os;
    os << "face quadrature supports 1 to 4 points per direction, got " << points;
    throw ConfigError(os.str());
  }
  std::vector<FacePoint> rule;
  switch (shape) {
    case FaceShape::Quad4:
      for (const QuadraturePoint2D& g : gauss_legendre_square(points)) {
        rule.push_back({g.xi, g.eta, g.weight});
      }
      break;
  }
  return rule;
}

void face_shape_functions(FaceShape shape, Scalar s, Scalar t, Vector& n, Matrix& dn) {
  const int nf = face_shape_nodes(shape);
  n.resize(nf);
  dn.setZero(nf, 2);
  switch (shape) {
    case FaceShape::Quad4: {
      static constexpr Scalar c[4][2] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
      for (int a = 0; a < 4; ++a) {
        n(a) = 0.25 * (1.0 + s * c[a][0]) * (1.0 + t * c[a][1]);
        dn(a, 0) = 0.25 * c[a][0] * (1.0 + t * c[a][1]);
        dn(a, 1) = 0.25 * c[a][1] * (1.0 + s * c[a][0]);
      }
      return;
    }
  }
}

FaceGeometryPoint face_geometry(FaceShape shape, const Matrix& face_coords, Scalar s, Scalar t,
                                const Vector& n, const Matrix& dn) {
  (void)shape;
  (void)s;
  (void)t;
  if (face_coords.rows() != 3) throw MeshError("face coordinates must have three rows");
  const Matrix& x = face_coords;
  FaceGeometryPoint g;
  g.x = x * n;
  g.xs = x * dn.col(0);
  g.xt = x * dn.col(1);
  g.area = g.xs.cross(g.xt);
  g.jacobian = g.area.norm();
  return g;
}

Matrix element_face_coordinates(const Mesh& mesh, Index element, int local_face) {
  const std::vector<std::vector<int>>& faces = element_local_faces(mesh.element_type());
  if (local_face < 0 || local_face >= static_cast<int>(faces.size())) {
    std::ostringstream os;
    os << "local face " << local_face << " is outside the element's " << faces.size()
       << " faces";
    throw MeshError(os.str());
  }
  const std::vector<int>& fn = faces[static_cast<std::size_t>(local_face)];
  const Index* nodes = mesh.element_nodes(element);
  Matrix x(mesh.dim(), static_cast<Eigen::Index>(fn.size()));
  for (std::size_t a = 0; a < fn.size(); ++a) {
    x.col(static_cast<Eigen::Index>(a)) = mesh.coordinates().col(nodes[fn[a]]);
  }
  return x;
}

Vector face_shape_integrals(FaceShape shape, const Matrix& face_coords, int points) {
  check_nodes(shape, face_coords);
  Vector out = Vector::Zero(face_shape_nodes(shape));
  Vector n;
  Matrix dn;
  for (const FacePoint& p : face_quadrature(shape, points)) {
    face_shape_functions(shape, p.s, p.t, n, dn);
    const FaceGeometryPoint g = face_geometry(shape, face_coords, p.s, p.t, n, dn);
    out += (p.weight * g.jacobian) * n;
  }
  return out;
}

}  // namespace sparlab
