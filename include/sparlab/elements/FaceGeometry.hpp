/// \file FaceGeometry.hpp
/// \brief Shape functions, quadrature and outward area vectors of element
///        boundary faces.
///
/// A boundary face of a Hex8 is itself an isoparametric entity, a four-node
/// bilinear quadrilateral. The contact weights of the tools' surfaces are
/// integrals over such faces, written once here against the face's own
/// nodes.
///
/// **Orientation.** The face tables of Mesh.hpp wind every face so its
/// right-hand normal points out of the element. With face coordinates
/// \f$(s, t)\f$ and \f$x(s,t) = \sum_a N_a(s,t)\,x_a\f$, the outward area
/// vector per unit reference measure is \f$a = x_{,s} \times x_{,t}\f$, so
/// \f$\int_\Gamma f\,n\,d\Gamma = \int f\,a\,ds\,dt\f$, and \f$|a|\f$ is the
/// area element.
///
/// Where this fits: Simulations stage - integrals over Hex8 boundary faces that give each
///   tool-contact node its tributary area.
/// Called from: fem/RigidTool.cpp:ToolContact::ToolContact
/// Calls into: mesh/Mesh (element_local_faces), elements/Quadrature
/// Start reading at: face_shape_integrals, face_geometry
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/mesh/Mesh.hpp"

#include <vector>

namespace sparlab {

/// Boundary-face topologies.
enum class FaceShape {
  Quad4  ///< face of a Hex8: s, t in [-1, 1]
};

/// The boundary-face topology of an element type.
/// \throws MeshError for an element type without faces.
FaceShape face_shape_of(ElementType type);

/// Nodes of a face topology.
int face_shape_nodes(FaceShape shape);

/// One quadrature point on the reference face (weight without the area
/// element).
struct FacePoint {
  Scalar s = 0.0;
  Scalar t = 0.0;
  Scalar weight = 0.0;
};

/// Quadrature rule on a reference face, exact for polynomials of degree
/// `2 * points - 1` per direction (1 <= points <= 4).
std::vector<FacePoint> face_quadrature(FaceShape shape, int points);

/// Shape functions and their derivatives with respect to (s, t) at a point:
/// `n` has face_shape_nodes entries, `dn` is nodes x 2 (d/ds, d/dt).
void face_shape_functions(FaceShape shape, Scalar s, Scalar t, Vector& n, Matrix& dn);

/// Geometry of a face at one reference point.
struct FaceGeometryPoint {
  Vector3 x = Vector3::Zero();      ///< position [m]
  Vector3 xs = Vector3::Zero();     ///< dx/ds [m]
  Vector3 xt = Vector3::Zero();     ///< dx/dt [m]
  Vector3 area = Vector3::Zero();   ///< outward area vector a (see the file comment)
  Scalar jacobian = 0.0;            ///< |a| [m^2]
};

/// Evaluate the face geometry at (s, t) from the face's nodal coordinates
/// (3 x face nodes).
FaceGeometryPoint face_geometry(FaceShape shape, const Matrix& face_coords, Scalar s,
                                Scalar t, const Vector& n, const Matrix& dn);

/// Nodal coordinates (dim x face nodes) of local face `local_face` of
/// element `e`, in the face table's order.
Matrix element_face_coordinates(const Mesh& mesh, Index element, int local_face);

/// \f$\int_\Gamma N_a\,d\Gamma\f$ per face node: the consistent nodal weights
/// of a uniform flux.
Vector face_shape_integrals(FaceShape shape, const Matrix& face_coords, int points);

}  // namespace sparlab
